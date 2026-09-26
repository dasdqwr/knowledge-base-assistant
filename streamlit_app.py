import json
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
        def stream_from_backend():
            with requests.post(
                    f"{API_URL}/chat/stream",
                    json={
                        "message": prompt,
                        "thread_id": st.session_state.thread_id,
                    },
                    stream=True,
                    timeout=120,
            ) as resp:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data["type"] == "token":
                            yield data["content"]
                        elif data["type"] == "done":
                            break
                        elif data["type"] == "error":
                            yield f"\n[错误: {data['content']}]"
                            break

        answer = st.write_stream(stream_from_backend())

    st.session_state.messages.append({"role": "assistant", "content": answer})