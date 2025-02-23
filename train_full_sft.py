# 导入必要的库和模块
import os  # 文件和目录操作
import platform  # 获取平台信息
import argparse  # 解析命令行参数
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
from transformers import AutoTokenizer, AutoModelForCausalLM  # Hugging Face 的预训练模型和分词器
from model.model import MiniMindLM  # 自定义的 MiniMind 模型
from model.LMConfig import LMConfig  # 模型配置类
from model.dataset import SFTDataset  # 自定义的数据集类

warnings.filterwarnings('ignore')  # 忽略警告信息


def Logger(content):
    """日志记录函数，仅在非分布式模式或主进程中打印日志"""
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """动态调整学习率的函数，使用余弦退火策略"""
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    """训练一个 epoch 的函数"""
    loss_fct = nn.CrossEntropyLoss(reduction='none')  # 定义交叉熵损失函数
    start_time = time.time()  # 记录开始时间
    
    for step, (X, Y, loss_mask) in enumerate(train_loader):  # 遍历数据加载器
        X = X.to(args.device)  # 将输入数据移动到指定设备（CPU 或 GPU）
        Y = Y.to(args.device)  # 将标签数据移动到指定设备
        loss_mask = loss_mask.to(args.device)  # 将损失掩码移动到指定设备
        
        # 动态调整当前步的学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:  # 更新优化器的学习率
            param_group['lr'] = lr

        with ctx:  # 使用自动混合精度上下文管理器
            res = model(X)  # 前向传播
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),  # 展平 logits 和标签
                Y.view(-1)
            ).view(Y.size())  # 计算每个 token 的损失
            
            # 应用损失掩码并计算平均损失
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss  # 添加辅助损失
            loss = loss / args.accumulation_steps  # 梯度累积

        scaler.scale(loss).backward()  # 反向传播并缩放梯度

        if (step + 1) % args.accumulation_steps == 0:  # 梯度累积达到设定步数时更新参数
            scaler.unscale_(optimizer)  # 取消缩放以进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪

            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 更新缩放因子

            optimizer.zero_grad(set_to_none=True)  # 清空梯度

        if step % args.log_interval == 0:  # 按照设定间隔打印日志
            spend_time = time.time() - start_time  # 计算已花费时间
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):  # 使用 WandB 记录日志
                wandb.log({"loss": loss,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):  # 按照设定间隔保存模型
            model.eval()  # 切换到评估模式
            moe_path = '_moe' if lm_config.use_moe else ''  # 根据是否使用 MoE 决定保存路径后缀
            ckp = f'{args.save_dir}/full_sft_{lm_config.dim}{moe_path}.pth'  # 构建保存路径

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):  # 如果是分布式模型
                state_dict = model.module.state_dict()  # 获取模块的状态字典
            else:
                state_dict = model.state_dict()  # 获取模型的状态字典

            torch.save(state_dict, ckp)  # 保存模型状态字典
            model.train()  # 切换回训练模式


def init_model(lm_config):
    """初始化模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')  # 加载预训练分词器
    model = MiniMindLM(lm_config)  # 初始化 MiniMind 模型
    moe_path = '_moe' if lm_config.use_moe else ''  # 根据是否使用 MoE 决定加载路径后缀
    ckp = f'./out/pretrain_{lm_config.dim}{moe_path}.pth'  # 构建预训练模型路径
    state_dict = torch.load(ckp, map_location=args.device)  # 加载预训练模型权重
    model.load_state_dict(state_dict, strict=False)  # 加载模型权重，允许部分加载
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')  # 打印模型参数量
    model = model.to(args.device)  # 将模型移动到指定设备
    return model, tokenizer


def init_distributed_mode():
    """初始化分布式训练模式"""
    if not ddp: return  # 如果不是分布式模式，直接返回
    
    global ddp_local_rank, DEVICE  # 定义全局变量

    dist.init_process_group(backend="nccl")  # 初始化分布式进程组
    ddp_rank = int(os.environ["RANK"])  # 获取分布式训练的全局排名
    ddp_local_rank = int(os.environ["LOCAL_RANK"])  # 获取本地排名
    ddp_world_size = int(os.environ["WORLD_SIZE"])  # 获取世界大小
    DEVICE = f"cuda:{ddp_local_rank}"  # 设置设备为本地 GPU
    torch.cuda.set_device(DEVICE)  # 设置当前使用的 GPU


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")  # 创建命令行解析器
    parser.add_argument("--out_dir", type=str, default="out")  # 输出目录
    parser.add_argument("--epochs", type=int, default=1)  # 训练轮数
    parser.add_argument("--batch_size", type=int, default=32)  # 批次大小
    parser.add_argument("--learning_rate", type=float, default=5e-5)  # 学习率
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")  # 设备选择
    parser.add_argument("--dtype", type=str, default="bfloat16")  # 数据类型
    parser.add_argument("--use_wandb", action="store_true")  # 是否使用 WandB
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT")  # WandB 项目名称
    parser.add_argument("--num_workers", type=int, default=1)  # 数据加载器的工作线程数
    parser.add_argument("--ddp", action="store_true")  # 是否使用分布式训练
    parser.add_argument("--accumulation_steps", type=int, default=1)  # 梯度累积步数
    parser.add_argument("--grad_clip", type=float, default=1.0)  # 梯度裁剪阈值
    parser.add_argument("--warmup_iters", type=int, default=0)  # 预热迭代次数
    parser.add_argument("--log_interval", type=int, default=100)  # 日志打印间隔
    parser.add_argument("--save_interval", type=int, default=100)  # 模型保存间隔
    parser.add_argument('--local_rank', type=int, default=-1)  # 本地排名
    parser.add_argument('--dim', default=512, type=int)  # 模型维度
    parser.add_argument('--n_layers', default=8, type=int)  # 模型层数
    parser.add_argument('--max_seq_len', default=512, type=int)  # 最大序列长度
    parser.add_argument('--use_moe', default=False, type=bool)  # 是否使用 MoE
    parser.add_argument("--data_path", type=str, default="./dataset/sft_mini_512.jsonl")  # 数据集路径

    args = parser.parse_args()  # 解析命令行参数

    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)  # 初始化模型配置
    args.save_dir = os.path.join(args.out_dir)  # 设置保存目录
    os.makedirs(args.save_dir, exist_ok=True)  # 创建保存目录
    os.makedirs(args.out_dir, exist_ok=True)  # 创建输出目录
    tokens_per_iter = args.batch_size * lm_config.max_seq_len  # 每个迭代处理的 token 数量
    torch.manual_seed(1337)  # 设置随机种子
    device_type = "cuda" if "cuda" in args.device else "cpu"  # 设置设备类型

    args.wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"  # 设置 WandB 运行名称

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()  # 设置自动混合精度上下文管理器
    ddp = int(os.environ.get("RANK", -1)) != -1  # 判断是否为分布式训练
    ddp_local_rank, DEVICE = 0, "cuda:0"  # 初始化分布式训练变量
    if ddp:
        init_distributed_mode()  # 初始化分布式训练模式
        args.device = torch.device(DEVICE)  # 设置设备

    if args.use_wandb and (not ddp or ddp_local_rank == 0):  # 初始化 WandB
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
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}  # 忽略某些参数和缓冲区
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])  # 包装模型为分布式数据并行

    iter_per_epoch = len(train_loader)  # 每个 epoch 的迭代次数
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)  # 训练每个 epoch