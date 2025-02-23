"""
MiniMind 语言模型服务端代码
基于FastAPI框架实现，支持流式响应和普通响应两种模式
"""

# 导入标准库模块
import argparse  # 用于解析命令行参数
import json      # 用于JSON数据处理
import os        # 处理文件路径和操作系统相关功能
import sys       # 系统相关功能，用于修改模块导入路径
import time      # 时间相关操作
import warnings  # 控制警告信息

# 导入第三方库
import torch          # PyTorch深度学习框架
import uvicorn        # ASGI服务器，用于运行FastAPI应用
from fastapi import FastAPI, HTTPException  # FastAPI核心组件
from fastapi.responses import StreamingResponse  # 流式响应支持
from pydantic import BaseModel  # 数据验证和设置管理
from transformers import AutoTokenizer, AutoModelForCausalLM  # Hugging Face模型工具

# 添加项目根目录到系统路径，确保可以导入自定义模块
__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入自定义模块
from model.LMConfig import LMConfig       # 模型配置类
from model.model import MiniMindLM        # 核心模型类
from model.model_lora import apply_lora, load_lora  # LoRA相关功能

# 忽略警告信息（生产环境中不建议使用）
warnings.filterwarnings('ignore')

# 初始化FastAPI应用实例
app = FastAPI(title="MiniMind API Server", description="MiniMind语言模型推理接口")

def init_model(args):
    """
    初始化模型和分词器
    参数:
        args: 命令行参数对象，包含模型配置信息
    返回:
        (model, tokenizer): 加载好的模型实例和分词器
    """
    # 加载基础分词器（使用预训练的MiniMind分词器）
    tokenizer = AutoTokenizer.from_pretrained('../model/minimind_tokenizer')
    
    # 判断加载模式（0: 加载原生PyTorch权重，1: 使用transformers加载）
    if args.load == 0:
        # 构造模型路径相关参数
        moe_path = '_moe' if args.use_moe else ''  # 是否使用混合专家模型
        modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason'}  # 模型模式映射表
        # 构造模型检查点路径
        ckp = f'../{args.out_dir}/{modes[args.model_mode]}_{args.dim}{moe_path}.pth'

        # 初始化自定义模型结构
        model = MiniMindLM(LMConfig(
            dim=args.dim,              # 模型维度
            n_layers=args.n_layers,    # 层数
            max_seq_len=args.max_seq_len,  # 最大序列长度
            use_moe=args.use_moe       # 是否使用混合专家
        ))

        # 加载模型权重（排除mask相关参数）
        state_dict = torch.load(ckp, map_location=device)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=True)

        # 如果启用了LoRA（低秩适应）
        if args.lora_name != 'None':
            apply_lora(model)  # 应用LoRA结构
            load_lora(model, f'../{args.out_dir}/{args.lora_name}_{args.dim}.pth')  # 加载LoRA权重
    else:
        # 使用transformers直接加载预训练模型
        model = AutoModelForCausalLM.from_pretrained(
            './MiniMind2',
            trust_remote_code=True  # 信任远程代码（自定义模型需要）
        )
    
    # 打印模型参数量统计信息
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
    
    # 将模型设置为评估模式并转移到指定设备
    return model.eval().to(device), tokenizer

class ChatRequest(BaseModel):
    """
    API请求数据模型（遵循OpenAI API格式）
    """
    model: str = "minimind"        # 模型名称（固定为minimind）
    messages: list                 # 对话历史列表，格式：[{"role": "user", "content": "..."}, ...]
    temperature: float = 0.7       # 温度参数（0-2），控制生成随机性，值越小输出越确定
    top_p: int = 0.92             # 核采样参数（0-1），控制生成多样性
    max_tokens: int = 8192         # 生成的最大token数
    stream: bool = False           # 是否使用流式传输

def generate_stream_response(messages, temperature, top_p, max_tokens):
    """
    生成流式响应内容
    参数:
        messages: 对话历史
        temperature: 温度参数
        top_p: 核采样参数
        max_tokens: 最大生成token数
    返回:
        生成器，逐个产生事件流数据块
    """
    try:
        # 应用聊天模板格式化输入（保留最近的max_tokens个token）
        new_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )[-max_tokens:]
        
        # 将文本转换为token序列
        x = tokenizer(new_prompt).data['input_ids']
        x = torch.tensor(x, dtype=torch.long, device=device)[None, ...]  # 增加batch维度
        
        with torch.no_grad():  # 禁用梯度计算
            # 生成文本流
            res_y = model.generate(
                x,
                eos_token_id=tokenizer.eos_token_id,  # 结束符ID
                max_new_tokens=max_tokens,           # 最大新生成token数
                temperature=temperature,             # 温度参数
                top_p=top_p,                         # 核采样参数
                stream=True,                         # 流式生成模式
                rp=1.,                              # 重复惩罚参数（未使用）
                pad_token_id=tokenizer.pad_token_id  # 填充符ID
            )
            
            history_idx = 0  # 记录已生成内容的位置
            for y in res_y:  # 遍历生成的token序列
                # 解码当前生成的token（跳过特殊token）
                answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                
                # 过滤无效字符（UTF-8替换字符）和空内容
                if (answer and answer[-1] == '�') or not answer:
                    continue
                
                # 计算新增内容片段
                delta = answer[history_idx:]
                history_idx = len(answer)  # 更新已处理位置
                
                # 构造符合OpenAI格式的响应数据
                json_data = {
                    'id': f'chatcmpl-{int(time.time())}',  # 唯一ID（时间戳）
                    'object': 'chat.completion.chunk',     # 对象类型
                    'created': int(time.time()),           # 创建时间
                    'model': 'minimind',                   # 模型名称
                    'choices': [{
                        'index': 0,
                        'delta': {'content': delta},       # 增量内容
                        'finish_reason': None              # 未结束
                    }]
                }
                yield f"data: {json.dumps(json_data)}\n\n"  # SSE格式数据
                
    except Exception as e:
        # 异常处理，返回错误信息
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    处理聊天补全请求（兼容OpenAI API格式）
    参数:
        request: ChatRequest实例，包含请求参数
    返回:
        StreamingResponse或标准JSON响应
    """
    try:
        if request.stream:  # 流式响应模式
            return StreamingResponse(
                generate_stream_response(
                    messages=request.messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens
                ),
                media_type="text/event-stream"  # 指定媒体类型为事件流
            )
        else:  # 普通响应模式
            # 格式化输入（与流式处理相同）
            new_prompt = tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True
            )[-request.max_tokens:]
            
            # 准备输入tensor
            x = tokenizer(new_prompt).data['input_ids']
            x = torch.tensor(x, dtype=torch.long, device=device)[None, ...]
            
            with torch.no_grad():
                # 生成完整响应（非流式）
                res_y = model.generate(
                    x,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stream=False,  # 关闭流式生成
                    rp=1.,
                    pad_token_id=tokenizer.pad_token_id
                )
                # 解码生成的token（跳过输入部分）
                answer = tokenizer.decode(res_y.squeeze()[x.shape[1]:].tolist(), skip_special_tokens=True)
            
            # 构造标准响应格式
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "minimind",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop"  # 完成原因
                }]
            }

    except Exception as e:
        # 异常处理，返回500错误
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Server for MiniMind")
    parser.add_argument('--out_dir', default='out', type=str, 
                      help='模型权重输出目录')
    parser.add_argument('--lora_name', default='None', type=str,
                      help='LoRA权重名称，None表示不使用')
    parser.add_argument('--dim', default=512, type=int,
                      help='模型维度')
    parser.add_argument('--n_layers', default=8, type=int,
                      help='模型层数')
    parser.add_argument('--max_seq_len', default=8192, type=int,
                      help='最大序列长度')
    parser.add_argument('--use_moe', default=False, type=bool,
                      help='是否使用混合专家模型')
    parser.add_argument('--load', default=0, type=int,
                      help='加载模式：0-加载原生PyTorch权重，1-使用transformers加载')
    parser.add_argument('--model_mode', default=1, type=int,
                      help='模型模式：0-预训练，1-SFT微调，2-RLHF微调，3-推理模式')

    # 自动选择设备（优先使用CUDA）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 初始化模型和分词器
    model, tokenizer = init_model(parser.parse_args())
    
    # 启动FastAPI服务器
    uvicorn.run(app, host="0.0.0.0", port=8998)



    """
    代码说明要点：
架构设计：基于FastAPI实现REST API，支持流式和非流式响应，兼容OpenAI API格式

模型管理：

支持两种加载方式：原生PyTorch权重和transformers格式

支持LoRA适配器加载

支持混合专家（MoE）架构

核心功能：

使用transformers的apply_chat_template处理对话历史格式

支持temperature、top_p等生成参数调节

智能处理生成内容中的特殊字符

性能优化：

自动选择CUDA设备

使用torch.no_grad()减少内存占用

流式生成降低响应延迟

安全性：

使用Pydantic进行输入验证

完善的异常处理机制

限制最大生成token数防止资源耗尽

可以通过命令行参数灵活配置模型参数，示例启动命令：

bash
复制
python server.py --dim 512 --n_layers 8 --model_mode 1 --use_moe False
    """