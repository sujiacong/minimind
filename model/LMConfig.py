"""
完整带注释代码说明：
类定义：

继承自PretrainedConfig，确保与Hugging Face生态兼容

model_type用于Transformers自动模型加载时识别模型类型

基础参数：

dim到flash_attn控制模型主体架构：

n_kv_heads支持分组查询注意力（GQA）

multiple_of优化内存访问模式

rope_theta调节位置编码的频率范围

flash_attn启用高效注意力计算

MOE参数：

通过use_moe开关控制是否启用

路由机制参数：

num_experts_per_tok决定每个token使用的专家数

n_routed_experts设置专家池大小

scoring_func控制路由权重计算方式

辅助训练参数：

aux_loss_alpha平衡主任务损失和专家负载均衡损失

seq_aux决定辅助损失的计算粒度

实现细节：

所有参数最终赋值给实例变量

通过super().__init__处理父类参数

hidden_dim的自动计算逻辑在模型构建时实现

设计特点：

清晰的参数分组（基础参数/MOE参数）

默认值设置适合中等规模模型

类型提示增强代码可读性

兼容Transformers的预训练/微调流程
"""
# 导入所需库
from transformers import PretrainedConfig  # Hugging Face Transformers库中的预训练配置基类
from typing import List  # 用于类型提示

# 定义语言模型配置类，继承自PretrainedConfig
class LMConfig(PretrainedConfig):
    """
    MiniMInd语言模型的配置类，用于存储模型结构和训练相关的所有配置参数
    继承自Hugging Face的PretrainedConfig，支持与Transformers库的兼容性
    """
    
    # 模型类型标识，用于自动类注册时识别模型类型
    model_type = "minimind"

    def __init__(
        self,
        # 基础模型架构参数
        dim: int = 512,                   # 模型隐藏层维度（特征向量维度）
        n_layers: int = 8,                # Transformer的层数（解码器块数量）
        n_heads: int = 8,                # 多头注意力机制中的注意力头数量
        n_kv_heads: int = 2,             # 键/值投影的注意力头数量（用于分组查询注意力）
        vocab_size: int = 6400,           # 词汇表大小（输入词嵌入维度）
        hidden_dim: int = None,           # FFN层的隐藏维度（如果未指定则根据multiple_of计算）
        multiple_of: int = 64,           # 确保FFN隐藏维度是该值的整数倍（用于内存对齐优化）
        norm_eps: float = 1e-5,           # Layer Normalization层中的epsilon值（防止除零错误）
        max_seq_len: int = 8192,          # 模型支持的最大序列长度
        rope_theta: int = 1e6,           # RoPE（旋转位置编码）的基频参数（theta值）
        dropout: float = 0.0,             # 全连接层的dropout比率（0.0表示不使用dropout）
        flash_attn: bool = True,         # 是否使用Flash Attention优化（加速注意力计算）
        
        ####################################################
        # 以下是MOE（混合专家）的专有配置
        # 当use_moe=False时，以下参数无效
        ####################################################
        use_moe: bool = False,            # 是否启用混合专家（MoE）架构
        num_experts_per_tok: int = 2,     # 每个token选择使用的专家数量（top-k值）
        n_routed_experts: int = 4,       # 参与路由选择的专家总数（非共享专家数量）
        n_shared_experts: bool = True,    # 是否使用共享专家（跨层共享的专家模块）
        scoring_func: str = 'softmax',    # 专家评分函数类型（如softmax, topk等）
        aux_loss_alpha: float = 0.1,      # 辅助损失权重（用于平衡专家负载均衡）
        seq_aux: bool = True,             # 是否在序列级别计算辅助损失（True）还是批次级别（False）
        norm_topk_prob: bool = True,      # 是否对top-k专家的概率进行归一化处理
        **kwargs,                         # 父类PretrainedConfig的额外参数
    ):
        """
        初始化模型配置参数
        
        参数说明：
        - 基础参数控制模型的主体架构
        - MOE参数仅在use_moe=True时生效，用于构建混合专家系统
        """
        
        # 设置基础模型参数
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of   # 保证FFN层维度对齐，提升计算效率
        self.norm_eps = norm_eps          # 数值稳定性参数，防止归一化时的除零错误
        self.max_seq_len = max_seq_len    # 决定位置编码的最大长度
        self.rope_theta = rope_theta      # 控制RoPE编码的频率基值
        self.dropout = dropout            # 正则化参数，防止过拟合
        self.flash_attn = flash_attn      # 是否使用优化的Flash Attention实现

        ####################################################
        # 设置MOE相关参数
        ####################################################
        self.use_moe = use_moe            # MOE开关，False时以下参数不生效
        self.num_experts_per_tok = num_experts_per_tok  # 每个token路由选择的专家数
        self.n_routed_experts = n_routed_experts       # 可路由选择的专家总数
        self.n_shared_experts = n_shared_experts        # 是否使用跨层共享的专家
        self.scoring_func = scoring_func  # 专家选择的评分函数（控制路由逻辑）
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失系数（平衡主损失和负载均衡）
        self.seq_aux = seq_aux            # True: 序列级辅助损失 False: 批次级
        self.norm_topk_prob = norm_topk_prob  # 是否对top-k概率进行归一化（总和为1）

        # 调用父类初始化方法（处理PretrainedConfig的标准参数）
        super().__init__(**kwargs)

        # 注意：当hidden_dim未指定时，实际值会在模型构建时根据公式计算：
        # hidden_dim = multiple_of * ((2/3 * 4 * dim) // multiple_of)