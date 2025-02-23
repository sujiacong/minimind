# 导入必要的PyTorch模块
import torch
from torch import optim, nn

# 定义LoRA网络结构
class LoRA(nn.Module):
    """LoRA（低秩适应）网络模块
    通过低秩分解实现参数高效调整，用于预训练模型的微调
    
    Args:
        in_features (int): 输入特征维度
        out_features (int): 输出特征维度
        rank (int): 低秩矩阵的秩，控制近似矩阵的维度
    """
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # 保存秩参数用于后续操作
        
        # 定义低秩分解矩阵A（向下投影）和B（向上投影）
        # 两个线性层均不包含偏置项，符合LoRA论文设计
        self.A = nn.Linear(in_features, rank, bias=False)  # 形状：(in_features, rank)
        self.B = nn.Linear(rank, out_features, bias=False)  # 形状：(rank, out_features)
        
        # 初始化参数
        # 矩阵A使用高斯初始化（与Transformer的常规初始化一致）
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B初始化为全零（确保初始状态不影响原始模型输出）
        self.B.weight.data.zero_()

    def forward(self, x):
        """前向传播过程
        实现低秩矩阵乘法：B(A(x))
        
        Args:
            x (torch.Tensor): 输入张量，形状为(batch_size, in_features)
            
        Returns:
            torch.Tensor: 输出张量，形状为(batch_size, out_features)
        """
        return self.B(self.A(x))  # 顺序执行A和B的线性变换


def apply_lora(model, rank=16):
    """将LoRA适配器应用到模型的线性层
    
    Args:
        model (nn.Module): 需要应用LoRA的目标模型
        rank (int, optional): LoRA的秩，默认16
    """
    # 遍历模型的所有命名模块
    for name, module in model.named_modules():
        # 筛选条件：1. 是线性层 2. 权重矩阵是方阵（适用于自注意力等场景）
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 创建LoRA适配器实例，保持输入输出维度相同
            lora = LoRA(
                in_features=module.weight.shape[0],
                out_features=module.weight.shape[1],
                rank=rank
            ).to(model.device)  # 保持设备一致性
            
            # 将LoRA适配器挂载到原始模块上
            setattr(module, "lora", lora)
            
            # 保存原始前向传播方法
            original_forward = module.forward
            
            # 定义新的前向传播方法（显式绑定参数避免闭包陷阱）
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                """组合原始输出和LoRA适配输出的新前向传播
                
                Args:
                    x (torch.Tensor): 输入张量
                    layer1 (function): 原始前向传播函数（显式绑定）
                    layer2 (LoRA): LoRA适配器实例（显式绑定）
                    
                Returns:
                    torch.Tensor: 原始输出与LoRA输出的和
                """
                return layer1(x) + layer2(x)  # 残差连接结构
            
            # 替换模块的前向传播方法
            module.forward = forward_with_lora


def load_lora(model, path):
    """加载LoRA权重到模型中
    
    Args:
        model (nn.Module): 已应用LoRA的模型
        path (str): LoRA权重文件路径
    """
    # 加载完整状态字典到对应设备
    state_dict = torch.load(path, map_location=model.device)
    
    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 检查是否包含LoRA适配器
        if hasattr(module, 'lora'):
            # 提取与该模块对应的LoRA参数
            # 键名处理示例：将"layer.0.lora.A.weight"转换为"A.weight"
            lora_state = {
                k.replace(f'{name}.lora.', ''): v 
                for k, v in state_dict.items() 
                if f'{name}.lora.' in k
            }
            # 加载匹配的参数到当前模块的LoRA适配器
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """保存模型的LoRA权重
    
    Args:
        model (nn.Module): 已应用LoRA的模型
        path (str): 保存路径
    """
    state_dict = {}
    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 检查是否包含LoRA适配器
        if hasattr(module, 'lora'):
            # 获取LoRA参数并添加层级前缀
            # 示例：将"A.weight"转换为"layer.0.lora.A.weight"
            lora_state = {
                f'{name}.lora.{k}': v 
                for k, v in module.lora.state_dict().items()
            }
            state_dict.update(lora_state)  # 合并到总状态字典
    
    # 保存完整LoRA状态字典
    torch.save(state_dict, path)


"""
使用说明：
1. 首先对模型调用apply_lora进行适配
   model = TransformerModel()
   apply_lora(model, rank=8)

2. 训练完成后使用save_lora保存适配器参数
   save_lora(model, "lora_weights.pth")

3. 加载时先应用相同结构的LoRA，再加载参数
   apply_lora(new_model, rank=8)
   load_lora(new_model, "lora_weights.pth")

注意事项：
- 仅适用于权重矩阵为方阵的线性层（适用于自注意力层的适配）
- 加载权重时需要保持相同的rank设置
- 原始模型参数默认冻结，需要手动设置requires_grad=True进行全参数训练
- 显式绑定参数是为了避免Python闭包陷阱（late binding问题）
"""