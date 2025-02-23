"""
各数据集的数据格式要求：
PretrainDataset：

每行一个JSON对象，包含"text"字段（原始文本）

示例：{"text": "这是一个示例句子..."}

SFTDataset：

每行一个JSON对象，包含"conversations"字段（对话列表）

对话轮次需严格交替user/assistant角色

示例：{"conversations": [{"content": "你好"}, {"content": "你好！有什么可以帮您？"}]}

DPODataset：

每行一个JSON对象，包含"chosen"和"rejected"字段

每个字段都是对话列表（格式同SFTDataset）

示例：{"chosen": [...], "rejected": [...]}

这些数据集类可以与PyTorch的DataLoader配合使用，实现不同训练阶段的高效数据加载。
"""
# 导入必要的库
import json  # 用于处理JSON数据格式
import random  # 随机数生成
import re  # 正则表达式处理
import pandas as pd  # 数据处理库
import numpy as np  # 科学计算库
from torch.utils.data import Dataset, DataLoader  # PyTorch数据加载工具
import torch  # PyTorch深度学习框架
from sklearn.model_selection import train_test_split  # 数据分割工具
import os  # 操作系统接口
import ast  # 抽象语法树操作

# 设置环境变量（禁用tokenizers的并行处理以避免警告）
os.environ["TOKENIZERS_PARALLELISM"] = "false"

###############################################
# 预训练数据集处理类
###############################################
class PretrainDataset(Dataset):
    """用于语言模型预训练的数据集类"""
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        初始化预训练数据集
        Args:
            data_path (str): 训练数据文件路径（jsonl格式）
            tokenizer: 预训练的分词器对象
            max_length (int): 输入序列的最大长度（默认512）
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)  # 加载数据

    def load_data(self, path):
        """加载并解析jsonl格式的预训练数据"""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):  # 从1开始计数行号
                try:
                    data = json.loads(line.strip())  # 解析每行JSON数据
                    samples.append(data)
                except json.JSONDecodeError:
                    print(f"警告：第{line_num}行JSON解析失败，已跳过")
        return samples

    def __len__(self):
        """返回数据集样本数量"""
        return len(self.samples)

    def __getitem__(self, index):
        """
        获取单个训练样本，格式化为：
        X: 输入token序列（前n-1个token）
        Y: 目标token序列（后n-1个token）
        loss_mask: 损失掩码（忽略padding部分的损失）
        """
        sample = self.samples[index]
        
        # 构建输入文本，添加特殊token
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"
        
        # 使用分词器处理文本
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,  # 最大长度截断
            padding='max_length',  # 填充到最大长度
            truncation=True,  # 启用截断
            return_tensors='pt'  # 返回PyTorch张量
        )
        
        input_ids = encoding.input_ids.squeeze()  # 移除多余的维度 [1, L] -> [L]
        
        # 创建损失掩码（忽略padding部分的损失）
        loss_mask = (input_ids != self.tokenizer.pad_token_id)
        
        # 构建训练数据（使用前n-1个token预测后n-1个token）
        # X: [0,1,...,n-2], Y: [1,2,...,n-1]
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐Y的位置
        
        return X, Y, loss_mask


###############################################
# 监督微调数据集处理类（SFT）
###############################################
class SFTDataset(Dataset):
    """用于监督式微调的对话数据集类"""
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        """
        初始化SFT数据集
        Args:
            jsonl_path (str): 对话数据文件路径（jsonl格式）
            tokenizer: 预训练的分词器对象
            max_length (int): 输入序列的最大长度（默认1024）
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        
        # 预计算特殊token的ID（用于动态损失掩码生成）
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids  # 回复开始标记
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids          # 回复结束标记

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        """加载对话数据集"""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())  # 解析对话数据
                    # 数据格式应包含'conversations'字段，每个对话轮次包含'content'
                    samples.append(data)
                except json.JSONDecodeError:
                    print(f"警告：第{line_num}行JSON解析失败，已跳过")
        return samples

    def _create_chat_prompt(self, conversations):
        """
        构建符合ChatML格式的对话提示
        Args:
            conversations: 对话列表，每个元素是包含'content'的字典
        Returns:
            str: 格式化后的对话文本
        """
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'  # 假设对话是严格的user/assistant交替
            messages.append({"role": role, "content": turn['content']})
        
        # 使用tokenizer的模板格式化对话
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # 返回字符串而不是token
            add_generation_prompt=False  # 不添加生成提示
        )

    def _generate_loss_mask(self, input_ids):
        """
        生成动态损失掩码（只在assistant回答部分计算损失）
        Args:
            input_ids: 分词后的token ID列表
        Returns:
            list: 损失掩码列表（1表示需要计算损失的位置）
        """
        loss_mask = [0] * len(input_ids)  # 初始化为全0
        i = 0
        
        # 遍历整个输入序列
        while i < len(input_ids):
            # 检查当前位置是否是assistant回复的开始（匹配bos_id）
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # 找到回复内容的起始位置（跳过bos_id本身）
                start = i + len(self.bos_id)
                end = start
                
                # 寻找对应的eos_id作为回复结束位置
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                
                # 设置从start+1到end位置的损失掩码为1（预测下一个token）
                # +1是因为我们预测的是下一个token，所以从start+1开始
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    if j < len(loss_mask):
                        loss_mask[j] = 1
                
                # 跳过已处理的eos_id部分
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1  # 继续遍历
        return loss_mask

    def __getitem__(self, index):
        """获取单个训练样本"""
        sample = self.samples[index]
        
        # 构建对话提示
        prompt = self._create_chat_prompt(sample['conversations'])
        
        # 分词并处理到固定长度
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]  # 截断
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))  # 填充
        
        # 生成动态损失掩码（只在assistant回复部分计算损失）
        loss_mask = self._generate_loss_mask(input_ids)
        
        # 构建训练数据（移位预测）
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐Y的位置
        
        return X, Y, loss_mask


###############################################
# DPO训练数据集处理类
###############################################
class DPODataset(Dataset):
    """用于直接偏好优化（DPO）训练的数据集类"""
    def __init__(self, file_path, tokenizer, max_length=4096):
        """
        初始化DPO数据集
        Args:
            file_path (str): 数据文件路径（jsonl格式）
            tokenizer: 预训练的分词器对象
            max_length (int): 输入序列最大长度（默认4096）
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        # 预计算特殊token的ID（用于动态损失掩码生成）
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids  # 回复开始标记
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids          # 回复结束标记
        
        # 加载数据
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                try:
                    obj = json.loads(line)
                    # 数据应包含'chosen'和'rejected'字段，每个都是对话列表
                    self.data.append(obj)
                except json.JSONDecodeError:
                    print(f"警告：解析行失败：{line[:50]}...")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """获取单个DPO训练样本（包含chosen和rejected对）"""
        item = self.data[index]
        chosen = item['chosen']   # 优选回复的对话列表
        rejected = item['rejected']  # 被拒绝回复的对话列表
        
        # 构建优选样本的对话提示
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, 
            tokenize=False,  # 返回字符串
            add_generation_prompt=False
        )
        
        # 构建拒绝样本的对话提示
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # 处理优选样本
        chosen_encoding = self.tokenizer(
            chosen_prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length'  # 填充到最大长度
        )
        
        # 处理拒绝样本
        rejected_encoding = self.tokenizer(
            rejected_prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length'
        )
        
        # 生成动态损失掩码（只在assistant回复部分计算损失）
        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)
        
        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        
        # 转换为PyTorch张量（移位预测格式）
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)
        
        return {
            'x_chosen': x_chosen,        # 优选样本输入
            'y_chosen': y_chosen,        # 优选样本目标
            'mask_chosen': mask_chosen,  # 优选样本损失掩码
            'x_rejected': x_rejected,    # 拒绝样本输入
            'y_rejected': y_rejected,    # 拒绝样本目标
            'mask_rejected': mask_rejected  # 拒绝样本损失掩码
        }

    def _generate_loss_mask(self, input_ids):
        """（与SFTDataset相同的方法）生成动态损失掩码"""
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 设置从start+1到结束位置的损失掩码
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    if j < len(loss_mask):
                        loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


# 主程序入口（示例代码保留为空）
if __name__ == "__main__":
    pass