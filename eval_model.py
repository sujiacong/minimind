# 导入必要的库和模块
import argparse  # 用于解析命令行参数
import random  # 用于生成随机数
import time  # 用于时间相关的操作
import numpy as np  # 用于科学计算
import torch  # PyTorch 深度学习框架
import warnings  # 用于控制警告信息的显示
from transformers import AutoTokenizer, AutoModelForCausalLM  # Hugging Face Transformers 库中的分词器和模型
from model.model import MiniMindLM  # 自定义的 MiniMindLM 模型
from model.LMConfig import LMConfig  # 自定义的模型配置类
from model.model_lora import *  # LoRA 相关模块

# 忽略所有警告信息
warnings.filterwarnings('ignore')


def init_model(args):
    """
    初始化模型和分词器。
    
    参数:
    args (argparse.Namespace): 包含命令行参数的对象
    
    返回:
    model (torch.nn.Module): 初始化后的模型
    tokenizer (transformers.PreTrainedTokenizer): 分词器
    """
    # 加载预训练的分词器
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    
    if args.load == 0:
        # 如果使用原生 PyTorch 权重加载模型
        moe_path = '_moe' if args.use_moe else ''  # 根据是否使用 MoE 决定路径后缀
        modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason'}  # 不同模式对应的文件名前缀
        ckp = f'./{args.out_dir}/{modes[args.model_mode]}_{args.dim}{moe_path}.pth'  # 模型权重文件路径
        
        # 创建 MiniMindLM 模型实例
        model = MiniMindLM(LMConfig(
            dim=args.dim,  # 模型维度
            n_layers=args.n_layers,  # 模型层数
            max_seq_len=args.max_seq_len,  # 最大序列长度
            use_moe=args.use_moe  # 是否使用 MoE
        ))
        
        # 加载模型权重
        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=True)
        
        # 如果指定了 LoRA 名称，则应用 LoRA 并加载 LoRA 权重
        if args.lora_name != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.out_dir}/lora/{args.lora_name}_{args.dim}.pth')
    else:
        # 如果使用 Hugging Face Transformers 加载模型
        transformers_model_path = './MiniMind2'
        tokenizer = AutoTokenizer.from_pretrained(transformers_model_path)
        model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)
    
    # 打印模型参数量
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
    
    # 返回模型和分词器，并将模型设置为评估模式并移动到指定设备
    return model.eval().to(args.device), tokenizer


def get_prompt_datas(args):
    """
    获取提示数据，根据不同的模型模式和 LoRA 名称选择不同的提示数据。
    
    参数:
    args (argparse.Namespace): 包含命令行参数的对象
    
    返回:
    prompt_datas (list): 提示数据列表
    """
    if args.model_mode == 0:
        # 预训练模型的接龙能力（无法对话）
        prompt_datas = [
            '马克思主义基本原理',
            '人类大脑的主要功能',
            '万有引力原理是',
            '世界上最高的山峰是',
            '二氧化碳在空气中',
            '地球上最大的动物有',
            '杭州市的美食有'
        ]
    else:
        if args.lora_name == 'None':
            # 通用对话问题
            prompt_datas = [
                '请介绍一下自己。',
                '你更擅长哪一个学科？',
                '鲁迅的《狂人日记》是如何批判封建礼教的？',
                '我咳嗽已经持续了两周，需要去医院检查吗？',
                '详细的介绍光速的物理概念。',
                '推荐一些杭州的特色美食吧。',
                '请为我讲解“大语言模型”这个概念。',
                '如何理解ChatGPT？',
                'Introduce the history of the United States, please.'
            ]
        else:
            # 特定领域问题
            lora_prompt_datas = {
                'lora_identity': [
                    "你是ChatGPT吧。",
                    "你叫什么名字？",
                    "你和openai是什么关系？"
                ],
                'lora_medical': [
                    '我最近经常感到头晕，可能是什么原因？',
                    '我咳嗽已经持续了两周，需要去医院检查吗？',
                    '服用抗生素时需要注意哪些事项？',
                    '体检报告中显示胆固醇偏高，我该怎么办？',
                    '孕妇在饮食上需要注意什么？',
                    '老年人如何预防骨质疏松？',
                    '我最近总是感到焦虑，应该怎么缓解？',
                    '如果有人突然晕倒，应该如何急救？'
                ],
            }
            prompt_datas = lora_prompt_datas[args.lora_name]
    
    return prompt_datas


# 设置可复现的随机种子
def setup_seed(seed):
    """
    设置可复现的随机种子，确保每次运行结果一致。
    
    参数:
    seed (int): 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """
    主函数，负责解析命令行参数、初始化模型、获取提示数据并进行推理。
    """
    parser = argparse.ArgumentParser(description="Chat with MiniMind")
    parser.add_argument('--lora_name', default='None', type=str)
    parser.add_argument('--out_dir', default='out', type=str)
    parser.add_argument('--temperature', default=0.85, type=float)
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    # 此处max_seq_len（最大允许输入长度）并不意味模型具有对应的长文本的性能，仅防止QA出现被截断的问题
    # MiniMind2-moe (145M)：(dim=640, n_layers=8, use_moe=True)
    # MiniMind2-Small (26M)：(dim=512, n_layers=8)
    # MiniMind2 (104M)：(dim=768, n_layers=16)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    # 携带历史对话上下文条数
    # history_cnt需要设为偶数，即【用户问题, 模型回答】为1组；设置为0时，即当前query不携带历史上文
    # 模型未经过外推微调时，在更长的上下文的chat_template时难免出现性能的明显退化，因此需要注意此处设置
    parser.add_argument('--history_cnt', default=0, type=int)
    parser.add_argument('--stream', default=True, type=bool)
    parser.add_argument('--load', default=0, type=int, help="0: 原生torch权重，1: transformers加载")
    parser.add_argument('--model_mode', default=1, type=int, help="0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: Reason模型")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 初始化模型和分词器
    model, tokenizer = init_model(args)
    
    # 获取提示数据
    prompts = get_prompt_datas(args)
    
    # 选择测试模式：自动测试或手动输入
    test_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    
    messages = []
    for idx, prompt in enumerate(prompts if test_mode == 0 else iter(lambda: input('👶: '), '')):
        # 设置随机种子
        setup_seed(random.randint(0, 2048))
        # setup_seed(2025)  # 如需固定每次输出则换成【固定】的随机种子
        if test_mode == 0: print(f'👶: {prompt}')

        messages = messages[-args.history_cnt:] if args.history_cnt else []
        messages.append({"role": "user", "content": prompt})
        
        # 构建新的提示文本
        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )[-args.max_seq_len + 1:] if args.model_mode != 0 else (tokenizer.bos_token + prompt)
        
        answer = new_prompt
        
        # 进行推理
        with torch.no_grad():
            x = torch.tensor(tokenizer(new_prompt)['input_ids'], device=args.device).unsqueeze(0)
            outputs = model.generate(
                x,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_seq_len,
                temperature=args.temperature,
                top_p=args.top_p,
                stream=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
            print('🤖️: ', end='')
            try:
                if not args.stream:
                    print(tokenizer.decode(outputs.squeeze()[x.shape[1]:].tolist(), skip_special_tokens=True), end='')
                else:
                    history_idx = 0
                    for y in outputs:
                        answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                        if (answer and answer[-1] == '�') or not answer:
                            continue
                        print(answer[history_idx:], end='', flush=True)
                        history_idx = len(answer)
            except StopIteration:
                print("No answer")
            print('\n')
        
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()