# 导入OpenAI官方库，用于与OpenAI API或兼容的API服务进行交互
# 注意: 这里实际可能用于连接本地部署的LLM服务
from openai import OpenAI

# 初始化OpenAI客户端配置
# 注意: 这里的配置适用于本地测试环境，生产环境需要调整
client = OpenAI(
    api_key="none",  # 认证密钥，本地测试时可置空或设为无效值
    base_url="http://localhost:8998/v1"  # 指向本地部署的LLM服务端点
)

# 流式输出开关配置
stream = True  # True表示启用流式响应，实时逐字输出；False则等待完整响应

# 初始化原始对话历史记录
# 注意: 该列表用于存储多轮对话的完整上下文
conversation_history_origin = []

# 创建对话历史的副本（当前逻辑下等同于空列表）
# 注意: 当前实现每次循环都会重置，实际可能造成上下文丢失
conversation_history = conversation_history_origin.copy()

# 启动无限循环对话系统
while True:
    # 重置对话历史（当前逻辑每次对话都是独立上下文）
    # ⚠️潜在问题: 这会导致无法进行多轮对话，每次都是新对话
    # 如需保留上下文，应删除这行，并在循环外维护对话历史
    conversation_history = conversation_history_origin.copy()
    
    # 获取用户输入
    query = input('[Q]: ')  # 控制台提示符获取用户问题
    
    # 将用户输入加入对话历史
    conversation_history.append({
        "role": "user",     # 标识消息发送者身份
        "content": query    # 消息文本内容
    })
    
    # 调用大语言模型生成回复
    response = client.chat.completions.create(
        model="minimind",  # 指定使用的模型名称（根据实际部署的模型调整）
        messages=conversation_history,  # 传入完整的对话上下文
        stream=stream  # 是否启用流式传输模式
    )
    
    # 处理非流式响应（当stream=False时执行）
    if not stream:
        # 直接从响应对象中获取完整回复内容
        assistant_res = response.choices[0].message.content
        # 打印带格式的助理回复
        print('[A]: ', assistant_res)
    # 处理流式响应（当stream=True时执行）
    else:
        print('[A]: ', end='')  # 打印前缀但不换行
        assistant_res = ''  # 初始化回复内容容器
        # 逐块处理流式响应
        for chunk in response:
            # 获取当前数据块内容（可能为空字符串）
            delta_content = chunk.choices[0].delta.content or ""
            # 实时打印流式输出内容（不换行）
            print(delta_content, end="")
            # 拼接完整回复内容
            assistant_res += delta_content
    
    # 将AI回复加入对话历史
    # 注意: 由于当前每次循环重置历史，该操作实际只在当次循环有效
    conversation_history.append({
        "role": "assistant",  # 标识AI身份
        "content": assistant_res  # AI生成的完整回复
    })
    
    # 打印两个换行作为对话分隔符
    print('\n\n')

# 代码改进建议:
# 1. 若需多轮对话，应将conversation_history_origin改为在循环外维护
#    并移除循环内的copy()操作
# 2. 可增加错误处理机制（如API调用失败、网络中断等）
# 3. 可添加退出指令检测（如输入'exit'时退出循环）
# 4. 建议将API配置参数（如base_url）提取为配置文件或环境变量
# 5. 可增加对话历史长度管理，防止超出模型token限制