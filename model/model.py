"""
该代码实现了完整的语言模型架构，主要特点包括：

模型架构：

基于Transformer结构

支持Rotary Positional Encoding（RoPE）

可选混合专家（MoE）层

支持Grouped-Query Attention

使用RMSNorm和SwiGLU激活

关键组件：

RMSNorm：改进的层归一化方法

precompute_pos_cis：预计算旋转位置编码

Attention：支持Flash Attention优化的多头注意力

MoEGate：动态专家选择门控机制

MOEFeedForward：混合专家前馈网络

高效的KV缓存机制

生成功能：

支持流式生成和普通生成两种模式

实现温度采样和Top-p采样

包含重复惩罚机制

增量解码优化（KV缓存）

工程优化：

训练/推理模式分离

MoE推理模式优化

类型转换优化（保持计算精度）

内存高效设计（避免不必要的复制）

每个组件都有详细的中文注释，解释了：

设计原理（如RoPE的数学公式）

参数含义

张量形状变化

关键实现细节

不同模式下的处理逻辑（训练/推理）

性能优化考虑

注释覆盖了从底层数学运算到高层架构设计的所有关键点，便于理解模型工作原理和进行二次开发。


"""
# 导入必要的库和模块
import math
import struct
import inspect
import time

from .LMConfig import LMConfig  # 自定义的模型配置类
from typing import Any, Optional, Tuple, List  # 类型提示
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel  # HuggingFace预训练模型基类
from transformers.modeling_outputs import CausalLMOutputWithPast  # 语言模型输出结构


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization (LLaMA的改进版LayerNorm)
    特点：计算量比标准LayerNorm更小，适合大模型
    
    Args:
        dim (int): 输入特征维度
        eps (float): 防止除零的小值
    """
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数

    def forward(self, x):
        # 公式：weight * x / sqrt(mean(x^2) + eps)
        return self.weight * (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    """预计算旋转位置编码（RoPE）的复数形式
    
    Args:
        dim: 每个注意力头的维度（需要是偶数）
        end: 预计算的最大序列长度
        theta: 旋转角度基数，默认1000000（LLaMA的设置）
        
    Returns:
        pos_cis: 复数形式的位置编码 [seq_len, dim//2]
    """
    # 计算频率向量（按公式θ_i = theta^(−2i/d)）
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # 位置序列 [0, 1, ..., end-1]
    freqs = torch.outer(t, freqs).float()  # 外积得到位置-频率矩阵 [seq_len, dim//2]
    
    # 将角度转换为复数形式（cosθ + i sinθ）
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # 使用极坐标形式生成复数
    return pos_cis


def apply_rotary_emb(xq, xk, pos_cis):
    """应用旋转位置编码到query和key上
    
    Args:
        xq: query张量 [batch, seq_len, n_heads, head_dim]
        xk: key张量 [batch, seq_len, n_kv_heads, head_dim]
        pos_cis: 预计算的位置编码复数 [seq_len, head_dim//2]
        
    Returns:
        旋转后的query和key张量
    """
    def unite_shape(pos_cis, x):
        """调整位置编码形状以匹配输入张量"""
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        # 生成与输入张量匹配的形状 [1, seq_len, 1, head_dim//2]
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    # 将最后两个维度转换为复数形式（将head_dim分成实部和虚部）
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 调整位置编码形状并进行复数乘法（旋转操作）
    pos_cis = unite_shape(pos_cis, xq_)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)  # 旋转后转回实数形式并展平
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """重复key/value张量以匹配多头注意力的数量
    用于Grouped-Query Attention场景，当n_kv_heads < n_heads时
    
    Args:
        x: 输入的key/value张量 [batch, seq_len, n_kv_heads, head_dim]
        n_rep: 每个key/value头需要重复的次数
        
    Returns:
        重复后的张量 [batch, seq_len, n_kv_heads * n_rep, head_dim]
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # 通过扩展和重塑实现高效重复
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """多头注意力机制模块，支持Grouped-Query Attention和Flash Attention"""
    def __init__(self, args: LMConfig):
        super().__init__()
        # 注意力头参数设置
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0, "n_heads必须能被n_kv_heads整除"
        self.n_local_heads = args.n_heads        # 总注意力头数
        self.n_local_kv_heads = self.n_kv_heads  # Key/Value头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每个KV头重复次数
        self.head_dim = args.dim // args.n_heads # 每个头的维度

        # 初始化线性变换层
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # 正则化参数
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # 是否使用Flash Attention加速
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # 初始化因果注意力掩码（上三角矩阵）
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False):
        # 输入形状: [batch_size, seq_len, dim]
        bsz, seq_len, _ = x.shape
        
        # 线性投影得到Q/K/V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # 重塑为多头格式
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        # KV缓存机制（用于增量解码）
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None  # 保存当前KV供后续使用

        # 调整张量形状为注意力计算格式 [bsz, n_heads, seq_len, head_dim]
        xq, xk, xv = (
            xq.transpose(1, 2),  # [bsz, n_heads, seq_len, head_dim]
            repeat_kv(xk, self.n_rep).transpose(1, 2),  # 重复KV头并转置
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        # 使用Flash Attention加速（PyTorch 2.0+）
        if self.flash and seq_len != 1:
            dropout_p = self.dropout if self.training else 0.0
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,    # Flash Attention自动处理因果掩码
                dropout_p=dropout_p,
                is_causal=True
            )
        else:  # 普通注意力实现
            # 计算注意力分数 [bsz, n_heads, seq_len, seq_len]
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores += self.mask[:, :, :seq_len, :seq_len]  # 应用因果掩码
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        # 重塑输出并应用最终线性层
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))
        return output, past_kv


class FeedForward(nn.Module):
    """标准前馈网络（SwiGLU激活）"""
    def __init__(self, config: LMConfig):
        super().__init__()
        # 计算隐藏层维度（遵循LLaMA的设计）
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3)  # SwiGLU的缩放因子
            # 对齐到multiple_of参数（如64的倍数）
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        
        # 网络层定义
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)  # 门控分支
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)  # 输出层
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)  # 前馈分支
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU激活函数：swish(w1(x)) * w3(x)
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MoEGate(nn.Module):
    """混合专家（MoE）门控机制
    实现Top-K专家选择及负载均衡辅助损失
    
    Args:
        config: 模型配置参数
    """
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok  # 每个token使用的专家数
        self.n_routed_experts = config.n_routed_experts  # 总专家数

        # 门控函数配置
        self.scoring_func = config.scoring_func  # 得分计算方式（如softmax）
        self.alpha = config.aux_loss_alpha       # 辅助损失系数
        self.seq_aux = config.seq_aux            # 是否使用序列级辅助损失

        self.norm_topk_prob = config.norm_topk_prob  # 是否归一化Top-K概率
        self.gating_dim = config.dim                 # 门控维度
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """参数初始化（Kaiming均匀分布）"""
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        # 输入形状: [batch_size, seq_len, dim]
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)  # [batch*seq_len, dim]
        
        # 计算专家得分（原始logits）
        logits = F.linear(hidden_states, self.weight, None)  # [batch*seq_len, n_experts]
        
        # 得分归一化（默认使用softmax）
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'不支持的评分函数: {self.scoring_func}')

        # 选择Top-K专家
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        # 概率归一化（当top_k > 1时）
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # 计算辅助损失（负载均衡损失）
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            
            if self.seq_aux:  # 序列级辅助损失
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                # 统计每个batch中专家被选中的次数
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                              torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                # 计算KL散度损失
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:  # Token级辅助损失
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)  # 专家选择频率
                Pi = scores_for_aux.mean(0)    # 平均门控概率
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0

        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """混合专家前馈网络
    包含多个专家网络和共享专家（可选）
    
    Args:
        config: 模型配置参数
    """
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        # 初始化专家网络
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)  # 门控模块
        
        # 可选共享专家（用于稳定训练）
        if config.n_shared_experts is not None:
            self.shared_experts = FeedForward(config)

    def forward(self, x):
        identity = x  # 残差连接
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        
        # 通过门控选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])  # [batch*seq_len, dim]
        flat_topk_idx = topk_idx.view(-1)  # 展平后的专家索引
        
        # 训练模式下的前向传播
        if self.training:
            # 复制输入以匹配专家数量 [batch*seq_len*top_k, dim]
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            
            # 并行处理所有专家
            for i, expert in enumerate(self.experts):
                # 处理分配给当前专家的所有token
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)
            
            # 加权求和并恢复形状
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:  # 推理模式优化
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        
        # 添加共享专家输出
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        
        self.aux_loss = aux_loss  # 保存辅助损失供后续使用
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """优化的推理模式前向传播
        减少不必要的计算，提升推理速度
        
        Args:
            x: 输入张量 [batch*seq_len, dim]
            flat_expert_indices: 专家索引 [batch*seq_len*top_k]
            flat_expert_weights: 专家权重 [batch*seq_len*top_k, 1]
        """
        expert_cache = torch.zeros_like(x)  # 初始化专家输出缓存
        idxs = flat_expert_indices.argsort()  # 按专家ID排序
        # 计算每个专家处理的token数量（累计和）
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok  # 原始token索引
        
        # 按专家顺序处理
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue  # 没有token分配给当前专家
            
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]  # 当前专家处理的token索引
            expert_tokens = x[exp_token_idx]  # 取出对应token
            
            # 专家前向传播并加权
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            
            # 将结果累加到缓存
            expert_cache.scatter_add_(0, 
                                     exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                                     expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    """Transformer Block结构
    包含自注意力层和前馈层（可能是MoE）
    
    Args:
        layer_id: 层编号
        config: 模型配置
    """
    def __init__(self, layer_id: int, config: LMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        
        # 模块初始化
        self.attention = Attention(config)  # 自注意力模块
        self.layer_id = layer_id
        
        # 归一化层
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        
        # 前馈网络（标准或MoE）
        self.feed_forward = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, x, pos_cis, past_key_value=None, use_cache=False):
        # 自注意力部分
        h_attn, past_kv = self.attention(
            self.attention_norm(x),  # 预归一化
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        h = x + h_attn  # 残差连接
        
        # 前馈网络部分
        out = h + self.feed_forward(self.ffn_norm(h))  # 残差连接
        return out, past_kv


class MiniMindLM(PreTrainedModel):
    """完整的语言模型结构"""
    config_class = LMConfig  # 关联配置类

    def __init__(self, params: LMConfig = None):
        self.params = params or LMConfig()
        super().__init__(self.params)
        
        # 初始化参数
        self.vocab_size, self.n_layers = params.vocab_size, params.n_layers
        
        # 嵌入层
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        
        # 堆叠Transformer层
        self.layers = nn.ModuleList([
            MiniMindBlock(l, params) for l in range(self.n_layers)
        ])
        
        # 输出层
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight  # 权重共享
        
        # 注册预计算的位置编码（不参与持久化保存）
        self.register_buffer("pos_cis",
                           precompute_pos_cis(dim=params.dim//params.n_heads, theta=params.rope_theta),
                           persistent=False)
        self.OUT = CausalLMOutputWithPast()  # 输出容器

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **args):
        # 处理输入
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = args.get('start_pos', 0)  # 用于增量解码的起始位置
        
        # 嵌入层
        h = self.dropout(self.tok_embeddings(input_ids))
        
        # 获取当前序列对应的位置编码
        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.size(1)]
        past_kvs = []  # 保存当前层的KV缓存
        
        # 逐层处理
        for l, layer in enumerate(self.layers):
            h, past_kv = layer(
                h, pos_cis,
                past_key_value=past_key_values[l],
                use_cache=use_cache
            )
            past_kvs.append(past_kv)
        
        # 最终归一化和输出投影
        logits = self.output(self.norm(h))
        
        # 收集MoE的辅助损失
        aux_loss = sum(l.feed_forward.aux_loss for l in self.layers if isinstance(l.feed_forward, MOEFeedForward))
        
        # 构造输出对象
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT

    @torch.inference_mode()
    def generate(self, input_ids, eos_token_id=2, max_new_tokens=1024, temperature=0.75, top_p=0.90,
                 stream=False, rp=1., use_cache=True, pad_token_id=0, **args):
        """文本生成函数
        
        Args:
            input_ids: 输入token IDs
            eos_token_id: 结束符ID
            max_new_tokens: 最大生成长度
            temperature: 温度参数（控制随机性）
            top_p: 核心采样概率
            stream: 是否流式生成
            rp: 重复惩罚系数
            use_cache: 是否使用KV缓存
            pad_token_id: 填充token ID
        """
        # 流式生成模式
        if stream:
            return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)

        # 普通生成模式（批处理）
        generated = []
        for i in range(input_ids.size(0)):
            # 去除padding并生成
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            out = self._stream(non_pad, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)
            # 收集生成结果
            tokens_list = [tokens[:, -1:] for tokens in out]
            gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad
            full_sequence = torch.cat([non_pad, gen], dim=-1)
            generated.append(full_sequence)
        
        # 对齐不同序列的长度（添加padding）
        max_length = max(seq.size(1) for seq in generated)
        generated = [
            torch.cat([seq, torch.full((1, max_length - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)], dim=-1)
            for seq in generated
        ]
        return torch.cat(generated, dim=0)

    def _stream(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args):
        """流式生成核心逻辑"""
        start, first_seq, past_kvs = input_ids.shape[1], True, None
        while input_ids.shape[1] < max_new_tokens - 1:
            # 前向传播
            if first_seq or not use_cache:
                out, first_seq = self(input_ids, past_key_values=past_kvs, use_cache=use_cache, **args), False
            else:
                out = self(input_ids[:, -1:], past_key_values=past_kvs, use_cache=use_cache,
                         start_pos=input_ids.shape[1] - 1, **args)
            
            # 处理输出
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values
            
            # 重复惩罚（降低已出现token的概率）
            logits[:, list(set(input_ids.tolist()[0]))] /= rp
            logits /= (temperature + 1e-9)  # 应用温度
            
            # Top-p（核）采样
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # 创建掩码移除低概率token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # 采样下一个token
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)
            
            yield input_ids[:, start:]  # 流式返回新生成的token
            
            # 遇到终止符则停止
            if input_ids_next.item() == eos_token_id:
                break