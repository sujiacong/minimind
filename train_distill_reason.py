# 导入必要的库
import os  # 文件和目录操作
import platform  # 获取操作系统信息
import argparse  # 命令行参数解析
import time  # 时间相关功能
import math  # 数学运算
import warnings  # 警告处理

import pandas as pd  # 数据处理
import torch  # PyTorch 深度学习框架
import torch.nn.functional as F  # PyTorch 神经网络函数
import torch.distributed as dist  # 分布式训练支持
from contextlib import nullcontext  # 上下文管理器

from torch import optim, nn  # 优化器和神经网络模块
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载器和分布式采样器
from transformers import AutoTokenizer, AutoModelForCausalLM  # Hugging Face Transformers 库中的分词器和预训练模型
from model.model import MiniMindLM  # 自定义的语言模型
from model.LMConfig import LMConfig  # 模型配置类
from model.dataset import SFTDataset  # 自定义的数据集类

warnings.filterwarnings('ignore')  # 忽略警告信息


def Logger(content):
    """
    日志记录函数，仅在非分布式模式或主进程中打印日志。
    """
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    计算当前步的学习率，使用余弦退火调度策略。
    
    参数:
    - current_step: 当前训练步数
    - total_steps: 总训练步数
    - lr: 初始学习率
    
    返回:
    - 当前步的学习率
    """
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    """
    训练一个epoch的函数。
    
    参数:
    - epoch: 当前epoch编号
    - wandb: WandB 实例，用于日志记录
    """
    # 获取思考标签和答案标签的token ID
    start_of_think_ids = tokenizer('<think>').input_ids
    end_of_think_ids = tokenizer('</think>').input_ids
    start_of_answer_ids = tokenizer('<answer>').input_ids
    end_of_answer_ids = tokenizer('</answer>').input_ids
    
    loss_fct = nn.CrossEntropyLoss(reduction='none')  # 定义损失函数
    start_time = time.time()  # 记录开始时间
    
    for step, (X, Y, loss_mask) in enumerate(train_loader):  # 遍历数据集
        X = X.to(args.device)  # 将输入数据移动到指定设备
        Y = Y.to(args.device)  # 将目标数据移动到指定设备
        loss_mask = loss_mask.to(args.device)  # 将损失掩码移动到指定设备
        
        # 计算当前步的学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        
        # 更新优化器的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        with ctx:  # 使用上下文管理器（自动混合精度）
            res = model(X)  # 前向传播
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),  # 展平logits
                Y.view(-1)  # 展平目标
            ).view(Y.size())  # 恢复形状
            
            # 获取特殊标记的token ID
            sp_ids = torch.isin(Y.view(-1),
                                torch.tensor(start_of_think_ids + end_of_think_ids
                                             + start_of_answer_ids + end_of_answer_ids
                                             ).to(args.device))
            
            # 在特殊标记位置增加额外惩罚
            loss_mask = loss_mask.view(-1)
            loss_mask_sum = loss_mask.sum()
            loss_mask[sp_ids] = 10
            loss_mask = loss_mask.view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask_sum
            loss += res.aux_loss  # 添加辅助损失
            loss = loss / args.accumulation_steps  # 平均梯度累积
        
        scaler.scale(loss).backward()  # 反向传播
        
        if (step + 1) % args.accumulation_steps == 0:  # 每累积一定步数更新参数
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪
            
            scaler.step(optimizer)  # 更新参数
            scaler.update()
            
            optimizer.zero_grad(set_to_none=True)  # 清空梯度
        
        if step % args.log_interval == 0:  # 每隔一定步数记录日志
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
            
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})
        
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):  # 每隔一定步数保存模型
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/reason_{lm_config.dim}{moe_path}.pth'
            
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    """
    初始化模型和分词器。
    
    参数:
    - lm_config: 模型配置
    
    返回:
    - model: 初始化后的模型
    - tokenizer: 分词器
    """
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')  # 加载分词器
    model = MiniMindLM(lm_config)  # 初始化模型
    
    moe_path = '_moe' if lm_config.use_moe else ''  # 是否使用MoE
    ckp = f'./out/rlhf_{lm_config.dim}{moe_path}.pth'  # 模型检查点路径
    
    state_dict = torch.load(ckp, map_location=args.device)  # 加载预训练模型权重
    model.load_state_dict(state_dict, strict=False)  # 加载权重到模型
    
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')  # 打印模型参数量
    
    model = model.to(args.device)  # 将模型移动到指定设备
    return model, tokenizer


def init_distributed_mode():
    """
    初始化分布式训练环境。
    """
    if not ddp: return
    
    global ddp_local_rank, DEVICE
    
    dist.init_process_group(backend="nccl")  # 初始化分布式进程组
    ddp_rank = int(os.environ["RANK"])  # 获取全局rank
    ddp_local_rank = int(os.environ["LOCAL_RANK"])  # 获取本地rank
    ddp_world_size = int(os.environ["WORLD_SIZE"])  # 获取world size
    DEVICE = f"cuda:{ddp_local_rank}"  # 设置设备
    torch.cuda.set_device(DEVICE)  # 设置CUDA设备


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Distill Reasoning")  # 创建命令行解析器
    
    # 添加命令行参数
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=1024, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="./dataset/r1_mix_1024.jsonl")

    args = parser.parse_args()  # 解析命令行参数
    
    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)  # 初始化模型配置
    args.save_dir = os.path.join(args.out_dir)  # 设置保存目录
    os.makedirs(args.save_dir, exist_ok=True)  # 创建保存目录
    os.makedirs(args.out_dir, exist_ok=True)  # 创建输出目录
    tokens_per_iter = args.batch_size * lm_config.max_seq_len  # 计算每迭代的token数量
    torch.manual_seed(1337)  # 设置随机种子
    device_type = "cuda" if "cuda" in args.device else "cpu"  # 设置设备类型
    
    args.wandb_run_name = f"MiniMind-Distill-Reasoning-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"  # 设置WandB运行名称
    
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()  # 设置上下文管理器
    ddp = int(os.environ.get("RANK", -1)) != -1  # 判断是否为分布式训练
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()  # 初始化分布式模式
        args.device = torch.device(DEVICE)  # 设置设备
    
    if args.use_wandb and (not ddp or ddp_local_rank == 0):  # 初始化WandB
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None
    
    model, tokenizer = init_model(lm_config)  # 初始化模型和分词器
    
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)  # 初始化训练数据集
    train_sampler = DistributedSampler(train_ds) if ddp else None  # 初始化分布式采样器
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )  # 初始化数据加载器
    
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))  # 初始化梯度缩放器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)  # 初始化优化器
    
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}  # 忽略某些参数
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])  # 包装为分布式模型
    
    iter_per_epoch = len(train_loader)  # 计算每个epoch的迭代次数
    
    for epoch in range(args.epochs):  # 开始训练
        train_epoch(epoch, wandb)