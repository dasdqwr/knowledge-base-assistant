import streamlit as st
import requests
import uuid


# 后端地址
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="知识库助手", page_icon="📚")
st.title("📚 个人知识库助手")

# ========== 初始化会话状态 ==========
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# ========== 侧边栏 ==========
with st.sidebar:
    st.write(f"会话 ID: `{st.session_state.thread_id[:8]}...`")
    if st.button("🔄 新会话"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# ========== 显示历史消息 ==========
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ========== 输入框 ==========
if prompt := st.chat_input("请输入问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 调后端
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "message": prompt,
                        "thread_id": st.session_state.thread_id,
                    },
                    timeout=60,
                )
                answer = resp.json()["answer"]
            except Exception as e:
                answer = f"请求失败: {e}"
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})