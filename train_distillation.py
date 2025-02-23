# 导入必要的库和模块
import os  # 文件和目录操作
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
from transformers import AutoTokenizer, AutoModelForCausalLM  # Hugging Face 变压器模型和分词器
from model.model import MiniMindLM  # 自定义的 MiniMind 模型
from model.LMConfig import LMConfig  # 模型配置类
from model.dataset import SFTDataset  # 自定义的数据集类

warnings.filterwarnings('ignore')  # 忽略警告信息


def Logger(content):
    """日志记录函数，仅在非分布式模式或主进程中打印日志"""
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """根据当前步数计算学习率，使用余弦退火调度器"""
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def distillation_loss_fn(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    """蒸馏损失函数，计算学生模型和教师模型之间的 KL 散度"""
    with torch.no_grad():
        # 计算教师模型的概率分布
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    # 计算学生模型的对数概率分布
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # 计算 KL 散度
    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    return (temperature ** 2) * kl


def train_epoch(epoch, wandb, alpha=0.0, temperature=1.0):
    """训练一个 epoch 的函数"""
    start_time = time.time()  # 记录开始时间

    # 如果有教师模型，将其设置为评估模式并禁用梯度计算
    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    # 遍历训练数据集
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)  # 将输入数据移动到指定设备
        Y = Y.to(args.device)  # 将标签数据移动到指定设备
        loss_mask = loss_mask.to(args.device)  # 将损失掩码移动到指定设备
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)  # 获取当前学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr  # 更新优化器的学习率

        # 学生模型前向传播
        with ctx:
            res = model(X)
            student_logits = res.logits  # 获取学生模型的输出 logits

        # 教师模型前向传播（只在 eval 和 no_grad 模式下）
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(X).logits
                vocab_size_student = student_logits.size(-1)  # N
                teacher_logits = teacher_logits[..., :vocab_size_student]  # 截取教师模型的 logits 到学生模型的词汇量大小

        # ========== 计算损失 ==========
        # 1) Ground-Truth CE Loss（交叉熵损失）
        loss_mask_flat = loss_mask.view(-1)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            Y.view(-1),
            ignore_index=0,
            reduction='none'
        )
        ce_loss = torch.sum(ce_loss * loss_mask_flat) / loss_mask_flat.sum()
        if lm_config_student.use_moe:
            ce_loss += res.aux_loss  # 如果使用 MoE 层，则添加辅助损失

        # 2) Distillation Loss（蒸馏损失）
        if teacher_model is not None:
            # 只在有效 token 位置做蒸馏
            distill_loss = distillation_loss_fn(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature
            )
        else:
            distill_loss = torch.tensor(0.0, device=args.device)

        # 3) 总损失 = alpha * CE + (1-alpha) * Distill
        loss = alpha * ce_loss + (1 - alpha) * distill_loss

        scaler.scale(loss).backward()  # 反向传播并缩放损失

        # 梯度累积
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪
            scaler.step(optimizer)  # 更新权重
            scaler.update()  # 更新缩放器
            optimizer.zero_grad(set_to_none=True)  # 清空梯度

        # 日志记录
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.4f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch,
                    args.epochs - 1,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                )
            )

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "loss": loss.item(),
                    "ce_loss": ce_loss.item(),
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                    "lr": optimizer.param_groups[-1]['lr'],
                    "last-time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                })

        # 模型保存
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/full_dist_{lm_config_student.dim}{moe_path}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(state_dict, ckp)
            model.train()


def init_student_model(lm_config):
    """初始化学生模型"""
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')  # 加载分词器
    model = MiniMindLM(lm_config)  # 初始化 MiniMind 模型
    moe_path = '_moe' if lm_config.use_moe else ''  # 根据是否使用 MoE 添加路径后缀
    ckp = f'./out/full_sft_{lm_config.dim}{moe_path}.pth'  # 模型检查点路径
    state_dict = torch.load(ckp, map_location=args.device)  # 加载预训练模型权重
    model.load_state_dict(state_dict, strict=False)  # 加载权重到模型中
    Logger(f'学生模型(LLM)总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)  # 将模型移动到指定设备

    return model, tokenizer


def init_teacher_model(lm_config):
    """初始化教师模型"""
    model = MiniMindLM(lm_config)  # 初始化 MiniMind 模型
    moe_path = '_moe' if lm_config.use_moe else ''  # 根据是否使用 MoE 添加路径后缀
    ckp = f'./out/full_sft_{lm_config.dim}{moe_path}.pth'  # 模型检查点路径
    state_dict = torch.load(ckp, map_location=args.device)  # 加载预训练模型权重
    model.load_state_dict(state_dict, strict=False)  # 加载权重到模型中
    Logger(f'教师模型(LLM)总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)  # 将模型移动到指定设备
    return model


def init_distributed_mode():
    """初始化分布式训练模式"""
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")  # 初始化分布式进程组
    ddp_rank = int(os.environ["RANK"])  # 获取全局 rank
    ddp_local_rank = int(os.environ["LOCAL_RANK"])  # 获取本地 rank
    ddp_world_size = int(os.environ["WORLD_SIZE"])  # 获取世界大小
    DEVICE = f"cuda:{ddp_local_rank}"  # 设置设备为本地 GPU
    torch.cuda.set_device(DEVICE)  # 设置当前使用的 GPU


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")  # 创建命令行参数解析器
    parser.add_argument("--out_dir", type=str, default="out")  # 输出目录
    parser.add_argument("--epochs", type=int, default=6)  # 训练轮数
    parser.add_argument("--batch_size", type=int, default=32)  # 批量大小
    parser.add_argument("--learning_rate", type=float, default=5e-6)  # 学习率
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")  # 设备类型
    parser.add_argument("--dtype", type=str, default="bfloat16")  # 数据类型
    parser.add_argument("--use_wandb", action="store_true")  # 是否使用 Weights & Biases 进行实验跟踪
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT")  # W&B 项目名称
    parser.add_argument("--num_workers", type=int, default=1)  # 数据加载器的工作线程数
    parser.add_argument("--ddp", action="store_true")  # 是否启用分布式数据并行
    parser.add_argument("--accumulation_steps", type=int, default=1)  # 梯度累积步数
    parser.add_argument("--grad_clip", type=float, default=1.0)  # 梯度裁剪阈值
    parser.add_argument("--warmup_iters", type=int, default=0)  # 预热迭代次数
    parser.add_argument("--log_interval", type=int, default=100)  # 日志记录间隔
    parser.add_argument("--save_interval", type=int, default=100)  # 模型保存间隔
    parser.add_argument('--local_rank', type=int, default=-1)  # 本地 rank
    parser.add_argument("--data_path", type=str, default="./dataset/sft_data.jsonl")  # 数据集路径

    args = parser.parse_args()  # 解析命令行参数
    # 定义学生模型和教师模型的配置
    lm_config_student = LMConfig(dim=512, n_layers=8, max_seq_len=512)
    lm_config_teacher = LMConfig(dim=768, n_layers=16, max_seq_len=512)
    max_seq_len = lm_config_student.max_seq_len  # 最大序列长度
    args.save_dir = os.path.join(args.out_dir)  # 保存目录
    os.makedirs(args.save_dir, exist_ok=True)  # 创建保存目录
    os.makedirs(args.out_dir, exist_ok=True)  # 创建输出目录
    tokens_per_iter = args.batch_size * max_seq_len  # 每次迭代的 token 数量
    torch.manual_seed(1337)  # 设置随机种子
    device_type = "cuda" if "cuda" in args.device else "cpu"  # 设置设备类型

    args.wandb_run_name = f"MiniMind-Dist-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()  # 设置上下文管理器
    ddp = int(os.environ.get("RANK", -1)) != -1  # 是否是分布式训练
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()  # 初始化分布式模式
        args.device = torch.device(DEVICE)  # 设置设备

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb  # 导入 Weights & Biases 库

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)  # 初始化 W&B 实验
    else:
        wandb = None

    # 初始化学生模型和教师模型
    model, tokenizer = init_student_model(lm_config_student)
    teacher_model = init_teacher_model(lm_config_teacher)

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=max_seq_len)  # 初始化训练数据集
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
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])  # 使用分布式数据并行

    iter_per_epoch = len(train_loader)  # 每个 epoch 的迭代次数
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)  # 训练一个 epoch