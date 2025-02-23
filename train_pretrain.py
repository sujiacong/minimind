"""
MiniMind 预训练脚本
支持分布式数据并行训练(DDP)、混合精度训练、梯度累积等技术
包含完整的训练循环、模型保存、日志记录等功能
"""

# 导入系统相关库
import os
import platform
import argparse  # 命令行参数解析
import time
import math
import warnings  # 警告处理

# 数据处理相关库
import pandas as pd

# PyTorch 核心库
import torch
import torch.distributed as dist  # 分布式训练
from torch import optim, nn  # 优化器和神经网络模块
from torch.nn.parallel import DistributedDataParallel  # DDP模型包装
from torch.optim.lr_scheduler import CosineAnnealingLR  # 学习率调度器
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载

# 上下文管理
from contextlib import nullcontext

# Transformers 相关组件
from transformers import AutoTokenizer  # 分词器

# 自定义模块
from model.model import MiniMindLM  # 核心模型
from model.LMConfig import LMConfig  # 模型配置
from model.dataset import PretrainDataset  # 预训练数据集

# 忽略警告信息
warnings.filterwarnings('ignore')


def Logger(content):
    """
    分布式安全的日志打印函数
    参数:
        content: 要打印的内容
    说明:
        在DDP环境下只允许rank 0进程打印，避免重复输出
    """
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    自定义学习率调度函数（线性warmup + 余弦退火）
    参数:
        current_step: 当前步骤数
        total_steps: 总步骤数
        lr: 基础学习率
    返回:
        计算后的学习率值
    公式说明:
        lr = base_lr/10 + 0.5*base_lr*(1 + cos(π*current_step/total_steps))
        实现从0.1*lr开始线性增长到lr，之后进行余弦退火
    """
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    """
    单个训练epoch的执行函数
    参数:
        epoch: 当前epoch序号
        wandb: wandb对象，用于日志记录（可为None）
    """
    loss_fct = nn.CrossEntropyLoss(reduction='none')  # 使用none reduction以便后续mask处理
    start_time = time.time()
    
    # 遍历训练数据
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 数据转移到设备
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 更新学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 混合精度上下文管理
        with ctx:
            # 前向传播
            res = model(X)
            # 计算损失（考虑mask）
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()  # 应用mask并归一化
            loss += res.aux_loss  # 添加辅助损失（如MoE的负载平衡损失）
            loss = loss / args.accumulation_steps  # 梯度累积归一化

        # 反向传播（使用梯度缩放）
        scaler.scale(loss).backward()

        # 梯度累积步骤判断
        if (step + 1) % args.accumulation_steps == 0:
            # 梯度裁剪前先取消缩放
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 参数更新
            scaler.step(optimizer)
            scaler.update()

            # 梯度清零（高效的内存处理）
            optimizer.zero_grad(set_to_none=True)

        # 日志记录间隔判断
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,  # 恢复实际损失值
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            # Wandb日志记录（仅在主进程）
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr'],
                    "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                })

        # 模型保存间隔判断（仅在主进程）
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()  # 切换到评估模式
            moe_path = '_moe' if lm_config.use_moe else ''  # MoE模型特殊标记
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'  # 保存路径

            # 处理DDP模型的state_dict
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()  # 恢复训练模式


def init_model(lm_config):
    """
    初始化模型和分词器
    参数:
        lm_config: 语言模型配置对象
    返回:
        model: 初始化后的模型
        tokenizer: 分词器对象
    """
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')  # 加载预训练分词器
    model = MiniMindLM(lm_config).to(args.device)  # 初始化模型并转移到设备
    # 打印模型参数量
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


def init_distributed_mode():
    """
    初始化分布式训练环境
    说明:
        - 设置NCCL后端
        - 获取分布式训练环境变量
        - 设置当前设备
    """
    if not ddp: return
    global ddp_local_rank, DEVICE

    # 初始化进程组
    dist.init_process_group(backend="nccl")
    # 获取环境变量
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    # 设置当前CUDA设备
    torch.cuda.set_device(DEVICE)


# 分布式训练启动命令示例：
# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    # ==================== 参数解析 ====================
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    # 输入输出参数
    parser.add_argument("--out_dir", type=str, default="out", help="输出目录")
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl", help="训练数据路径")

    # 训练参数
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="单卡批量大小")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], 
                       help="混合精度训练类型")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--accumulation_steps", type=int, default=8, 
                       help="梯度累积步数（实际批量大小 = batch_size * accumulation_steps * num_gpus）")

    # 设备参数
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                       help="训练设备（单卡训练时使用）")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")

    # 分布式参数
    parser.add_argument("--ddp", action="store_true", help="启用分布式训练")
    parser.add_argument('--local_rank', type=int, default=-1, help="自动设置的DDP参数，无需手动指定")

    # 模型参数
    parser.add_argument('--dim', default=512, type=int, help="模型隐藏层维度")
    parser.add_argument('--n_layers', default=8, type=int, help="模型层数")
    parser.add_argument('--max_seq_len', default=512, type=int, help="最大序列长度")
    parser.add_argument('--use_moe', default=False, type=bool, help="是否使用混合专家(MoE)结构")

    # 日志参数
    parser.add_argument("--use_wandb", action="store_true", help="启用Wandb日志记录")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="Wandb项目名称")
    parser.add_argument("--log_interval", type=int, default=100, help="日志记录间隔（步数）")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔（步数）")

    args = parser.parse_args()


    # ==================== 初始化配置 ====================
    # 模型配置
    lm_config = LMConfig(
        dim=args.dim,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        use_moe=args.use_moe
    )
    
    # 创建输出目录
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # 计算tokens_per_iter（用于统计吞吐量）
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    
    # 设置随机种子
    torch.manual_seed(1337)
    
    # 设备类型判断（cuda/cpu）
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 混合精度上下文配置
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # ==================== 分布式训练初始化 ====================
    ddp = int(os.environ.get("RANK", -1)) != -1  # 是否处于DDP环境
    ddp_local_rank, DEVICE = 0, "cuda:0"  # 默认值
    
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    # ==================== Wandb初始化 ====================
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # ==================== 模型和数据初始化 ====================
    model, tokenizer = init_model(lm_config)
    
    # 数据集和数据加载器
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None  # DDP采样器
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,        # 锁页内存加速数据传输
        drop_last=False,        # 不丢弃最后不完整的batch
        shuffle=False,          # 由sampler处理shuffle
        num_workers=args.num_workers,
        sampler=train_sampler  # DDP采样器
    )

    # ==================== 优化器配置 ====================
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))  # 梯度缩放器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)  # AdamW优化器

    # ==================== DDP模型包装 ====================
    if ddp:
        # 忽略特定参数（如位置编码）的同步
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        # 使用DistributedDataParallel包装模型
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # ==================== 训练循环 ====================
    iter_per_epoch = len(train_loader)  # 每个epoch的迭代次数
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)  # 执行训练epoch