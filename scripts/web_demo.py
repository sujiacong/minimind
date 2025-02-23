# 导入必要的库
import random  # 用于生成随机数
import re  # 正则表达式处理
import time  # 时间相关操作
import numpy as np  # 数值计算库
import streamlit as st  # 网页应用框架
import torch  # PyTorch深度学习框架

# 配置Streamlit页面
st.set_page_config(
    page_title="MiniMind",  # 页面标题
    initial_sidebar_state="collapsed"  # 初始侧边栏状态（折叠）
)

# 自定义CSS样式，用于美化界面元素
st.markdown("""
    <style>
        /* 操作按钮样式 */
        .stButton button {
            border-radius: 50% !important;  /* 圆形按钮 */
            width: 32px !important;         /* 固定宽度 */
            height: 32px !important;        /* 固定高度 */
            padding: 0 !important;          /* 移除内边距 */
            background-color: transparent !important;  /* 透明背景 */
            border: 1px solid #ddd !important;  /* 边框样式 */
            display: flex !important;       /* 弹性布局 */
            align-items: center !important;  /* 垂直居中 */
            justify-content: center !important;  /* 水平居中 */
            font-size: 14px !important;      /* 字体大小 */
            color: #666 !important;         /* 字体颜色 */
            margin: 5px 10px 5px 0 !important;  /* 外边距 */
        }
        /* 按钮悬停效果 */
        .stButton button:hover {
            border-color: #999 !important;   /* 悬停边框颜色 */
            color: #333 !important;          /* 悬停字体颜色 */
            background-color: #f5f5f5 !important;  /* 悬停背景色 */
        }
        /* 主内容区上边距调整 */
        .stMainBlockContainer > div:first-child {
            margin-top: -50px !important;
        }
        /* 底部边距调整 */
        .stApp > div:last-child {
            margin-bottom: -35px !important;
        }
        /* 重置按钮基础样式 */
        .stButton > button {
            all: unset !important;           /* 重置所有默认样式 */
            box-sizing: border-box !important;  /* 盒模型设置 */
            border-radius: 50% !important;   /* 圆形按钮 */
            width: 18px !important;          /* 更小的尺寸 */
            height: 18px !important;
            min-width: 18px !important;
            min-height: 18px !important;
            max-width: 18px !important;
            max-height: 18px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #888 !important;         /* 更浅的字体颜色 */
            cursor: pointer !important;      /* 鼠标指针样式 */
            transition: all 0.2s ease !important;  /* 过渡动画 */
            margin: 0 2px !important;        /* 外边距调整 */
        }
    </style>
""", unsafe_allow_html=True)  # 允许不安全HTML

# 全局变量初始化
system_prompt = []  # 系统提示语列表
device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动检测设备类型

def process_assistant_content(content):
    """
    处理助手返回内容中的特殊标记，转换为HTML元素
    参数:
        content (str): 原始助手返回内容
    返回:
        str: 处理后的HTML内容
    """
    # 仅对R1版本模型进行处理
    if 'R1' not in MODEL_PATHS[selected_model][1]:
        return content

    # 处理完整包含<think>标签的情况
    if '<think>' in content and '</think>' in content:
        content = re.sub(
            r'(<think>)(.*?)(</think>)',
            r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理内容（展开）</summary>\2</details>',
            content,
            flags=re.DOTALL
        )

    # 处理只有开始标签的情况
    if '<think>' in content and '</think>' not in content:
        content = re.sub(
            r'<think>(.*?)$',
            r'<details open style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理中...</summary>\1</details>',
            content,
            flags=re.DOTALL
        )

    # 处理只有结束标签的情况
    if '<think>' not in content and '</think>' in content:
        content = re.sub(
            r'(.*?)</think>',
            r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理内容（展开）</summary>\1</details>',
            content,
            flags=re.DOTALL
        )

    return content

@st.cache_resource  # Streamlit缓存装饰器，避免重复加载模型
def load_model_tokenizer(model_path):
    """
    加载预训练模型和分词器
    参数:
        model_path (str): 模型本地路径
    返回:
        tuple: (模型对象, 分词器对象)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True  # 信任远程代码（自定义模型需要）
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,  # 使用完整的分词器功能
        trust_remote_code=True
    )
    model = model.eval().to(device)  # 设置为评估模式并移动到指定设备
    return model, tokenizer

def clear_chat_messages():
    """清空聊天会话历史"""
    del st.session_state.messages
    del st.session_state.chat_messages

def init_chat_messages():
    """初始化聊天消息，处理历史消息显示"""
    if "messages" in st.session_state:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "assistant":
                # 显示助手消息，带删除按钮
                with st.chat_message("assistant", avatar=image_url):
                    st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
                    # 为每条消息添加删除按钮
                    if st.button("🗑", key=f"delete_{i}"):
                        # 删除对应的用户和助手消息
                        st.session_state.messages.pop(i)
                        st.session_state.messages.pop(i - 1)
                        st.session_state.chat_messages.pop(i)
                        st.session_state.chat_messages.pop(i - 1)
                        st.rerun()  # 立即刷新界面
            else:
                # 用户消息右对齐显示
                st.markdown(
                    f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px;  background-color: #ddd; border-radius: 10px; color: black;">{message["content"]}</div></div>',
                    unsafe_allow_html=True)
    else:
        # 初始化空的消息列表
        st.session_state.messages = []
        st.session_state.chat_messages = []
    return st.session_state.messages

# 辅助功能函数
def regenerate_answer(index):
    """重新生成指定位置的回答"""
    st.session_state.messages.pop()
    st.session_state.chat_messages.pop()
    st.rerun()

def delete_conversation(index):
    """删除指定位置的对话"""
    st.session_state.messages.pop(index)
    st.session_state.messages.pop(index - 1)
    st.session_state.chat_messages.pop(index)
    st.session_state.chat_messages.pop(index - 1)
    st.rerun()

# 侧边栏配置区域
st.sidebar.title("模型设定调整")
st.sidebar.text("【注】训练数据偏差，增加上下文记忆时\n多轮对话（较单轮）容易出现能力衰减")

# 对话历史数量滑块（步长2）
st.session_state.history_chat_num = st.sidebar.slider(
    "Number of Historical Dialogues", 
    0, 6, 0, step=2
)

# 生成参数配置
st.session_state.max_new_tokens = st.sidebar.slider(
    "Max Sequence Length", 
    256, 8192, 8192, step=1
)
st.session_state.top_p = st.sidebar.slider(
    "Top-P", 
    0.8, 0.99, 0.85, step=0.01
)
st.session_state.temperature = st.sidebar.slider(
    "Temperature", 
    0.6, 1.2, 0.85, step=0.01
)

# 模型路径映射字典
MODEL_PATHS = {
    "MiniMind2-R1 (0.1B)": ["../MiniMind2-R1", "MiniMind2-R1"],
    "MiniMind2-Small-R1 (0.02B)": ["../MiniMind2-Small-R1", "MiniMind2-Small-R1"],
    "MiniMind2 (0.1B)": ["../MiniMind2", "MiniMind2"],
    "MiniMind2-MoE (0.15B)": ["../MiniMind2-MoE", "MiniMind2-MoE"],
    "MiniMind2-Small (0.02B)": ["../MiniMind2-Small", "MiniMind2-Small"],
    "MiniMind-V1 (0.1B)": ["../minimind-v1", "MiniMind-V1"],
    "MiniMind-V1-MoE (0.1B)": ["../minimind-v1-moe", "MiniMind-V1-MoE"],
    "MiniMind-V1-Small (0.02B)": ["../minimind-v1-small", "MiniMind-V1-Small"],
}

# 模型选择下拉框（默认选中第三个选项）
selected_model = st.sidebar.selectbox(
    'Models', 
    list(MODEL_PATHS.keys()), 
    index=2
)
model_path = MODEL_PATHS[selected_model][0]  # 获取选择的模型路径

# 页面头部区域
image_url = "https://www.modelscope.cn/api/v1/studio/gongjy/MiniMind/repo?Revision=master&FilePath=images%2Flogo2.png&View=true"
slogan = f"Hi, I'm {MODEL_PATHS[selected_model][1]}"  # 动态生成标语

# 使用HTML构建页面头部布局
st.markdown(
    f'<div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0; padding: 0;">'
    '<div style="font-style: italic; font-weight: 900; margin: 0; padding-top: 4px; display: flex; align-items: center; justify-content: center; flex-wrap: wrap; width: 100%;">'
    f'<img src="{image_url}" style="width: 45px; height: 45px; "> '  # 品牌Logo
    f'<span style="font-size: 26px; margin-left: 10px;">{slogan}</span>'  # 动态标语
    '</div>'
    '<span style="color: #bbb; font-style: italic; margin-top: 6px; margin-bottom: 10px;">内容完全由AI生成，请务必仔细甄别<br>Content AI-generated, please discern with care</span>'
    '</div>',
    unsafe_allow_html=True
)

def setup_seed(seed):
    """设置随机种子保证结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 确保卷积操作结果确定
    torch.backends.cudnn.benchmark = False  # 关闭自动优化

def main():
    """主程序逻辑"""
    # 加载模型和分词器
    model, tokenizer = load_model_tokenizer(model_path)

    # 初始化会话消息
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    messages = st.session_state.messages

    # 显示历史消息
    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            with st.chat_message("assistant", avatar=image_url):
                # 处理并显示助手消息
                st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
                # 添加删除按钮
                if st.button("×", key=f"delete_{i}"):
                    st.session_state.messages = st.session_state.messages[:i - 1]
                    st.session_state.chat_messages = st.session_state.chat_messages[:i - 1]
                    st.rerun()
        else:
            # 用户消息右对齐显示
            st.markdown(
                f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px;  background-color: gray; border-radius: 10px; color:white; ">{message["content"]}</div></div>',
                unsafe_allow_html=True)

    # 用户输入处理
    prompt = st.chat_input(key="input", placeholder="给 MiniMind 发送消息")

    # 处理重新生成逻辑
    if hasattr(st.session_state, 'regenerate') and st.session_state.regenerate:
        prompt = st.session_state.last_user_message
        regenerate_index = st.session_state.regenerate_index
        # 清除重新生成相关状态
        delattr(st.session_state, 'regenerate')
        delattr(st.session_state, 'last_user_message')
        delattr(st.session_state, 'regenerate_index')

    if prompt:
        # 显示用户消息
        st.markdown(
            f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px;  background-color: gray; border-radius: 10px; color:white; ">{prompt}</div></div>',
            unsafe_allow_html=True)
        # 更新消息记录
        messages.append({"role": "user", "content": prompt})
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        # 生成助手回复
        with st.chat_message("assistant", avatar=image_url):
            placeholder = st.empty()  # 占位符用于流式显示
            random_seed = random.randint(0, 2**32-1)  # 生成随机种子
            setup_seed(random_seed)  # 设置随机种子

            # 构造历史对话上下文
            st.session_state.chat_messages = system_prompt + st.session_state.chat_messages[-(st.session_state.history_chat_num + 1):]
            # 应用聊天模板
            new_prompt = tokenizer.apply_chat_template(
                st.session_state.chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )[-(st.session_state.max_new_tokens - 1):]  # 截断到最大长度

            # 生成回答
            x = torch.tensor(tokenizer(new_prompt)['input_ids'], device=device).unsqueeze(0)
            with torch.no_grad():
                res_y = model.generate(
                    x, 
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=st.session_state.max_new_tokens,
                    temperature=st.session_state.temperature,
                    top_p=st.session_state.top_p,
                    stream=True  # 启用流式生成
                )
                try:
                    for y in res_y:
                        answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                        # 跳过无效字符
                        if (answer and answer[-1] == '�') or not answer:
                            continue
                        # 实时更新显示内容
                        placeholder.markdown(process_assistant_content(answer), unsafe_allow_html=True)
                except StopIteration:
                    print("No answer")

                # 后处理生成的回答
                assistant_answer = answer.replace(new_prompt, "")
                # 更新消息记录
                messages.append({"role": "assistant", "content": assistant_answer})
                st.session_state.chat_messages.append({"role": "assistant", "content": assistant_answer})

                # 为最新消息添加删除按钮
                with st.empty():
                    if st.button("×", key=f"delete_{len(messages) - 1}"):
                        st.session_state.messages = st.session_state.messages[:-2]
                        st.session_state.chat_messages = st.session_state.chat_messages[:-2]
                        st.rerun()

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    main()