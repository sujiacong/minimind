"""
将自定义PyTorch模型与HuggingFace Transformers格式互相转换的工具脚本
支持功能：
1. 将训练好的PyTorch模型转换为Transformers格式（包含模型和tokenizer）
2. 将Transformers格式模型转换回PyTorch格式
3. 参数统计和格式验证
"""

# 导入必要的库
import torch  # PyTorch深度学习框架
import warnings  # 警告处理模块
import sys  # 系统相关功能模块
import os  # 操作系统接口模块

# 设置当前包名为"scripts"，用于相对导入时正确解析路径
__package__ = "scripts"

# 将项目根目录添加到系统路径，确保可以导入自定义模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入HuggingFace Transformers相关组件
from transformers import AutoTokenizer, AutoModelForCausalLM  # 自动模型和分词器加载工具

# 导入自定义模块
from model.LMConfig import LMConfig  # 语言模型配置类
from model.model import MiniMindLM  # 自定义实现的MiniMind语言模型

# 忽略UserWarning类别的警告（通常用于清理输出）
warnings.filterwarnings('ignore', category=UserWarning)


def convert_torch2transformers(torch_path, transformers_path):
    """
    将PyTorch模型转换为Transformers格式的完整流程
    参数:
        torch_path (str): PyTorch模型权重文件路径(.pth)
        transformers_path (str): 转换后的Transformers模型保存路径
    """
    
    def export_tokenizer(transformers_path):
        """
        辅助函数：导出tokenizer到指定路径
        参数:
            transformers_path (str): tokenizer保存路径
        """
        # 从预训练路径加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained('../model/minimind_tokenizer')
        # 保存tokenizer到目标路径
        tokenizer.save_pretrained(transformers_path)
        print(f"Tokenizer已保存到: {transformers_path}")

    # 注册自定义配置到自动类系统，使Transformers库能自动识别我们的配置
    LMConfig.register_for_auto_class()
    
    # 将自定义模型类注册到自动模型类，使其可以通过AutoModelForCausalLM加载
    MiniMindLM.register_for_auto_class("AutoModelForCausalLM")

    # 初始化模型实例（需要提前定义lm_config变量）
    lm_model = MiniMindLM(lm_config)
    
    # 自动选择运行设备（优先使用GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载PyTorch模型权重
    # map_location参数确保权重加载到正确设备（CPU/GPU）
    state_dict = torch.load(torch_path, map_location=device)
    
    # 加载权重到模型（strict=False允许部分加载，适用于模型结构有微小差异的情况）
    lm_model.load_state_dict(state_dict, strict=False)
    
    # 计算可训练参数总数（展示模型规模）
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数统计:')
    print(f'- 总数: {model_params:,}')
    print(f'- 换算: {model_params / 1e6:.2f} 百万 / {model_params / 1e9:.2f} B')

    # 保存模型到指定路径
    # safe_serialization=False 使用传统PyTorch保存方式（兼容性更好）
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    
    # 导出tokenizer
    export_tokenizer(transformers_path)
    
    print(f"转换完成！Transformers格式模型已保存到: {transformers_path}")


def convert_transformers2torch(transformers_path, torch_path):
    """
    将Transformers格式模型转换回PyTorch格式
    参数:
        transformers_path (str): Transformers模型目录路径
        torch_path (str): 转换后的PyTorch模型保存路径(.pth)
    """
    # 从指定路径加载Transformers模型
    # trust_remote_code=True 允许执行自定义模型代码（需要信任模型来源时使用）
    model = AutoModelForCausalLM.from_pretrained(
        transformers_path,
        trust_remote_code=True
    )
    
    # 保存模型权重为PyTorch格式
    torch.save(model.state_dict(), torch_path)
    
    print(f"转换完成！PyTorch格式模型已保存到: {torch_path}")


# 以下函数当前未使用，作为示例保留
def push_to_hf(export_model_path):
    """
    将模型推送到HuggingFace Hub（示例函数，当前未启用）
    参数:
        export_model_path (str): 已导出的模型路径
    """
    def init_model():
        """初始化模型和tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained('../model/minimind_tokenizer')
        model = AutoModelForCausalLM.from_pretrained(
            export_model_path,
            trust_remote_code=True
        )
        return model, tokenizer

    model, tokenizer = init_model()
    # 实际推送需要配置HuggingFace账户和权限
    # model.push_to_hub(model_path)
    # tokenizer.push_to_hub(model_path, safe_serialization=False)
    print("推送功能需要配置HuggingFace账户后使用")


if __name__ == '__main__':
    # 主程序入口
    # 初始化语言模型配置
    lm_config = LMConfig(
        dim=512,        # 模型维度
        n_layers=8,     # 层数
        max_seq_len=8192,  # 最大序列长度
        use_moe=False   # 是否使用混合专家(MoE)结构
    )

    # 定义模型路径模板
    # 根据配置参数生成文件名：dim尺寸 + moe标识
    torch_path = f"../out/rlhf_{lm_config.dim}{'_moe' if lm_config.use_moe else ''}.pth"
    
    # Transformers格式模型保存路径
    transformers_path = '../MiniMind2-Small'

    # 执行转换操作（选择需要的转换方向）
    
    # 1. 将PyTorch模型转换为Transformers格式
    convert_torch2transformers(torch_path, transformers_path)
    
    # 2. 将Transformers格式转换回PyTorch格式（需要时取消注释）
    # convert_transformers2torch(transformers_path, torch_path)

    """
    关键功能说明
路径处理系统

通过sys.path.append将项目根目录加入路径，确保模块正确导入

使用动态路径组合保证跨平台兼容性

模型转换核心流程

PyTorch → Transformers：

注册自定义配置和模型类

权重加载与设备适配（自动选择GPU/CPU）

参数统计与验证

保存完整模型结构+权重+tokenizer

Transformers → PyTorch：

使用trust_remote_code加载自定义模型

仅保存模型权重（传统PyTorch格式）

安全与验证机制

strict=False允许部分权重加载（应对版本差异）

显式设备分配防止设备不匹配

参数统计验证模型完整性

扩展功能

预留HuggingFace Hub推送接口

动态路径命名（根据模型配置参数）

使用注意事项
路径依赖

确保../model/minimind_tokenizer存在有效tokenizer

输出目录(../out/)需要提前创建

设备兼容性

自动检测GPU可用性

加载权重时自动处理设备位置

版本兼容

使用safe_serialization=False保证旧版PyTorch兼容

当使用Transformers新特性时可能需要调整

安全警告

trust_remote_code=True需要确保模型来源可信

推送功能需要配置HF_TOKEN环境变量
    """