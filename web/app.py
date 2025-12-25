import streamlit as st
import requests
import time
import json

# 设置页面宽屏模式，看起来更像专业大模型
st.set_page_config(page_title="EasyTune Pro", layout="wide")

API_URL = "http://localhost:8000"

st.title("🚀 EasyTune - 大模型微调云平台")

# 初始化 session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'task_id' not in st.session_state:
    st.session_state['task_id'] = "" # 默认空

# 使用 Tabs 分隔功能
tab1, tab2 = st.tabs(["🏗️ 训练控制台", "💬 模型对话对比"])

# ==================== Tab 1: 训练控制台 ====================
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("1. 上传数据")
        uploaded_file = st.file_uploader("上传JSON数据集", type="json")
        if uploaded_file is not None:
            if st.button("📤 上传并预处理", key="upload_btn"):
                uploaded_file.seek(0)
                files = {"file": uploaded_file}
                res = requests.post(f"{API_URL}/upload", files=files)
                if res.status_code == 200:
                    st.session_state['file_id'] = res.json()['file_id']
                    st.success("✅ 上传成功！")

    with col2:
        st.subheader("2. 微调配置")
        # 建议 Epoch 默认设大一点
        epoch = st.slider("训练轮数 (Epochs)", 1, 50, 20)
        
        if st.button("🚀 开始微调任务", type="primary"):
            if 'file_id' not in st.session_state:
                st.error("请先上传数据！")
            else:
                payload = {"file_id": st.session_state['file_id'], "epoch": epoch}
                res = requests.post(f"{API_URL}/train", params=payload)
                st.session_state['task_id'] = res.json()['task_id']
                st.info(f"任务已提交，ID: {st.session_state['task_id']}")

        # 状态监控
        if st.session_state['task_id']:
            st.divider()
            st.write(f"当前任务 ID: `{st.session_state['task_id']}`")
            status_box = st.empty()
            
            # 简单的轮询逻辑
            while True:
                try:
                    res = requests.get(f"{API_URL}/status/{st.session_state['task_id']}")
                    status = res.json().get('status', 'unknown')
                    
                    if status == "running":
                        status_box.info("⏳ 正在全力训练中... (请查看后端控制台日志)")
                        time.sleep(2)
                    elif status == "success":
                        status_box.success("🎉 训练完成！请切换到【模型对话对比】标签页进行测试。")
                        break
                    elif status == "failed":
                        status_box.error("❌ 训练失败，请检查后端日志。")
                        break
                    else:
                        time.sleep(2)
                except:
                    break

# ==================== Tab 2: 模型对话对比 ====================
with tab2:
    st.subheader("🤖 微调效果验证")
    
    # 侧边栏控制
    with st.sidebar:
        st.header("🎮 对话控制")
        use_lora = st.toggle("🔥 启用微调模型 (LoRA)", value=False)
        
        current_task = st.text_input("任务 ID (自动填入)", value=st.session_state['task_id'])
        if use_lora and not current_task:
            st.warning("⚠️ 请先在训练台完成训练，或手动输入Task ID")
        
        if st.button("🧹 清空对话历史"):
            st.session_state['chat_history'] = []

    # 聊天记录显示
    for role, text in st.session_state['chat_history']:
        with st.chat_message(role):
            st.write(text)

    # 输入框
    if prompt := st.chat_input("输入你的问题（试试问'你是谁'）..."):
        # 1. 显示用户输入
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state['chat_history'].append(("user", prompt))

        # 2. 调用后端
        with st.chat_message("assistant"):
            with st.spinner("模型正在思考... (首次切换模型可能需要几秒加载)"):
                try:
                    payload = {
                        "query": prompt,
                        "task_id": current_task if current_task else None,
                        "use_lora": use_lora
                    }
                    res = requests.post(f"{API_URL}/chat", json=payload)
                    response_text = res.json()['response']
                    
                    # 这里的 UI 优化：显示当前用的是什么模型
                    model_tag = "【🔥微调版】" if use_lora else "【🧊基座版】"
                    final_text = f"{model_tag} {response_text}"
                    
                    st.write(final_text)
                    st.session_state['chat_history'].append(("assistant", final_text))
                    
                except Exception as e:
                    st.error(f"连接失败: {e}")