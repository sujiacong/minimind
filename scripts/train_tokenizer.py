"""G
BPE Tokenizer训练脚本
功能：从头训练一个Byte-level BPE tokenizer，保存并测试其功能
"""

# 导入必要的库
import random  # 用于设置随机种子
from tqdm import tqdm  # 显示进度条（注意：当前代码未实际使用）
from transformers import AutoTokenizer  # 用于加载训练好的tokenizer
import json  # 处理JSON文件
from datasets import load_dataset  # 加载数据集（注意：当前代码未实际使用）
from tokenizers import (  # Tokenizer相关组件
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
import os  # 处理文件路径

# 设置随机种子保证可复现性
random.seed(42)

def train_tokenizer():
    """训练BPE tokenizer的主函数"""
    
    def read_texts_from_jsonl(file_path):
        """
        从JSONL文件中读取文本数据的生成器
        Args:
            file_path: JSONL文件路径
        Yields:
            文本字符串
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)  # 解析每行JSON
                yield data['text']  # 返回text字段内容

    # 数据集路径（需要根据实际情况修改）
    data_path = '../dataset/pretrain_hq.jsonl'

    # 初始化BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    # 设置预分词器为Byte-level（处理Unicode字符）
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 定义特殊token列表
    special_tokens = ["<unk>", "<s>", "</s>"]  # 依次为未知token，句子开始，句子结束

    # 配置BPE训练器参数
    trainer = trainers.BpeTrainer(
        vocab_size=6400,  # 目标词表大小
        special_tokens=special_tokens,  # 确保特殊token被包含
        show_progress=True,  # 显示训练进度
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()  # 初始字母表（基础字符）
    )

    # 获取文本数据迭代器
    texts = read_texts_from_jsonl(data_path)

    # 开始训练tokenizer（使用文本迭代器）
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 设置Byte-level解码器（与预分词器对应）
    tokenizer.decoder = decoders.ByteLevel()

    # 验证特殊token的索引是否正确
    try:
        assert tokenizer.token_to_id("<unk>") == 0  # 验证<unk>的ID是否为0
        assert tokenizer.token_to_id("<s>") == 1   # 验证<s>的ID是否为1
        assert tokenizer.token_to_id("</s>") == 2   # 验证</s>的ID是否为2
    except AssertionError:
        raise ValueError("特殊token的ID分配不正确！")

    # 保存tokenizer的路径配置
    tokenizer_dir = "../model/minimind_tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)  # 递归创建目录
    
    # 保存tokenizer文件
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))  # 完整配置
    tokenizer.model.save("../model/minimind_tokenizer")  # 模型参数

    # 手动创建tokenizer配置文件（适配Hugging Face格式）
    config = {
        # 基础配置
        "add_bos_token": False,        # 不自动添加句子开始token
        "add_eos_token": False,        # 不自动添加句子结束token
        "add_prefix_space": False,     # 不在开头添加空格
        "clean_up_tokenization_spaces": False,  # 不清理空格
        "legacy": True,                # 兼容旧版行为
        "model_max_length": 32768,     # 模型最大长度限制
        # 特殊token配置
        "bos_token": "<s>",            # 句子开始token
        "eos_token": "</s>",           # 句子结束token
        "pad_token": "<unk>",         # 填充token（使用<unk>）
        "unk_token": "<unk>",         # 未知token
        # 特殊token解码器配置
        "added_tokens_decoder": {
            "0": {  # <unk>的配置
                "content": "<unk>",
                "lstrip": False,       # 左侧不剥离空格
                "normalized": False,   # 不做标准化
                "rstrip": False,       # 右侧不剥离空格
                "single_word": False,  # 不是单词语token
                "special": True        # 标记为特殊token
            },
            "1": {  # <s>的配置
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {  # </s>的配置
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        # 对话模板配置（用于chat场景）
        "chat_template": (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set system_message = messages[0]['content'] %}"
            "{{ '<s>system\\n' + system_message + '</s>\\n' }}"
            "{% else %}"
            "{{ '<s>system\\n你是 MiniMind，是一个有用的人工智能助手。</s>\\n' }}"
            "{% endif %}"
            "{% for message in messages %}"
            "{% set content = message['content'] %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ content + '</s>' + '\\n' }}"
            "{% endif %}"
            "{% endfor %}"
        ),
        # 其他配置
        "additional_special_tokens": [],  # 没有额外特殊token
        "sp_model_kwargs": {},            # 空参数
        "spaces_between_special_tokens": False,  # 特殊token间不加空格
        "tokenizer_class": "PreTrainedTokenizerFast"  # tokenizer类型
    }

    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("Tokenizer训练完成并保存成功！")

def eval_tokenizer():
    """测试训练好的tokenizer"""
    # 加载训练好的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("../model/minimind_tokenizer")

    # 测试对话模板功能
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    # 应用对话模板（不进行tokenize）
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print("生成的对话模板：\n", new_prompt)

    # 验证词表大小
    actual_vocab_size = len(tokenizer)
    print('\n实际词表大小：', actual_vocab_size)

    # 测试编码/解码功能
    model_inputs = tokenizer(new_prompt)
    print('\n编码后的输入长度：', len(model_inputs['input_ids']))

    # 解码测试
    input_ids = model_inputs['input_ids']
    response = tokenizer.decode(input_ids, skip_special_tokens=False)
    print('\n解码结果与原始文本是否一致：', response == new_prompt)

def main():
    """主执行函数"""
    train_tokenizer()  # 训练tokenizer
    eval_tokenizer()   # 测试tokenizer

if __name__ == '__main__':
    main()


"""
代码说明要点：
文件结构：

包含完整的训练和测试流程

使用Hugging Face的tokenizers库进行训练

生成符合Hugging Face格式的配置文件

关键功能：

BPE训练：使用Byte-level BPE算法，支持Unicode字符

特殊token处理：明确处理<unk>, <s>, </s>三个特殊token

大上下文支持：配置model_max_length=32768

对话模板：内置符合ChatML格式的对话模板

验证机制：

特殊token索引检查

编码/解码一致性验证

实际词表大小检查

可配置参数：

vocab_size: 可调整词表大小

data_path: 训练数据路径

model_max_length: 最大上下文长度

建议运行前检查：

确保../dataset/pretrain_hq.jsonl存在且格式正确

确保输出目录../model/minimind_tokenizer有写入权限

根据实际需求调整词表大小和特殊token配置
"""