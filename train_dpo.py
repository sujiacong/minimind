# 导入必要的库和模块
import os  # 操作系统接口
import platform  # 获取操作系统信息
import argparse  # 解析命令行参数
import time  # 时间处理
import math  # 数学运算
import warnings  # 控制警告信息

import pandas as pd  # 数据处理
import torch  # PyTorch深度学习框架
import torch.nn.functional as F  # PyTorch神经网络功能函数
import torch.distributed as dist  # 分布式训练支持
from contextlib import nullcontext  # 上下文管理器，用于兼容性

from torch import optim, nn  # PyTorch优化器和神经网络模块
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载器和分布式采样器
from transformers import AutoTokenizer, AutoModelForCausalLM  # Hugging Face Transformers库中的分词器和预训练模型
from model.model import MiniMindLM  # 自定义语言模型
from model.LMConfig import LMConfig  # 自定义语言模型配置类
from model.dataset import DPODataset  # 自定义数据集类

warnings.filterwarnings('ignore')  # 忽略所有警告信息


def Logger(content):
    """日志记录函数，仅在非分布式模式或主进程中打印日志"""
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """根据当前步数和总步数计算学习率，使用余弦退火调度策略"""
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def logits_to_probs(logits, labels):
    """
    将logits转换为概率值
    
    参数:
    - logits: 形状为(batch_size, seq_len, vocab_size)的张量，表示每个位置上每个词汇的概率分布
    - labels: 形状为(batch_size, seq_len)的张量，表示每个位置上的真实标签
    
    返回:
    - probs: 形状为(batch_size, seq_len)的张量，表示每个位置上真实标签的概率
    """
    log_probs = F.log_softmax(logits, dim=2)  # 计算logits的对数softmax
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)  # 根据labels索引对应的概率值
    return probs


def dpo_loss(ref_probs, probs, beta):
    """
    计算DPO损失函数
    
    参数:
    - ref_probs: 参考模型生成的概率，形状为(batch_size, seq_len)
    - probs: 当前模型生成的概率，形状为(batch_size, seq_len)
    - beta: 超参数，控制损失函数的强度
    
    返回:
    - loss: 计算得到的平均损失值
    """
    ref_probs = ref_probs.mean(dim=1)  # 计算参考模型每个样本的平均概率
    probs = probs.mean(dim=1)  # 计算当前模型每个样本的平均概率

    batch_size = ref_probs.shape[0]
    chosen_ref_probs = ref_probs[:batch_size // 2]  # 分割成chosen和rejected两部分
    reject_ref_probs = ref_probs[batch_size // 2:]
    chosen_probs = probs[:batch_size // 2]
    reject_probs = probs[batch_size // 2:]

    pi_logratios = chosen_probs - reject_probs  # 计算当前模型的log ratio
    ref_logratios = chosen_ref_probs - reject_ref_probs  # 计算参考模型的log ratio
    logits = pi_logratios - ref_logratios  # 计算最终的logits
    loss = -F.logsigmoid(beta * logits)  # 计算损失
    return loss.mean()


def train_epoch(epoch, wandb):
    """训练一个epoch"""
    start_time = time.time()
    for step, batch in enumerate(train_loader):
        # 将输入数据移动到指定设备
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)

        # 合并chosen和rejected的数据
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # 计算当前的学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr  # 更新优化器的学习率

        with ctx:
            # 使用参考模型生成logits，并计算概率
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_probs = logits_to_probs(ref_logits, y)
            ref_probs = ref_probs * mask

            # 使用当前模型生成logits，并计算概率
            outputs = model(x)
            logits = outputs.logits
            probs = logits_to_probs(logits, y)
            probs = probs * mask

            # 计算损失
            loss = dpo_loss(ref_probs, probs, beta=0.1)
            loss = loss / args.accumulation_steps  # 平均梯度累积

        # 反向传播
        scaler.scale(loss).backward()

        # 如果达到了梯度累积步数，则进行优化步骤
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # 清空梯度

        # 打印日志
        if step % args.log_interval == 0:
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

            # 使用WandB记录日志
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 保存模型检查点
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/rlhf_{lm_config.dim}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    """初始化模型、参考模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model = MiniMindLM(lm_config)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'./out/full_sft_{lm_config.dim}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)

    # 初始化参考模型
    ref_model = MiniMindLM(lm_config)
    ref_model.load_state_dict(state_dict, strict=False)
    ref_model.eval()
    ref_model.requires_grad_(False)

    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    ref_model = ref_model.to(args.device)

    return model, ref_model, tokenizer


def init_distributed_mode():
    """初始化分布式训练模式"""
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")  # 初始化分布式进程组
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind RLHF")
    
    # 添加命令行参数
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-8)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-RLHF-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=3000, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="./dataset/dpo.jsonl")

    args = parser.parse_args()

    # 初始化语言模型配置
    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 设置WandB运行名称
    args.wandb_run_name = f"MiniMind-Full-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 设置上下文管理器
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    ddp = int(os.environ.get("RANK", -1)) != -1  # 是否是分布式训练
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    # 初始化WandB
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化模型、参考模型和分词器
    model, ref_model, tokenizer = init_model(lm_config)

    # 加载数据集
    train_ds = DPODataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # 初始化混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 分布式训练设置
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)