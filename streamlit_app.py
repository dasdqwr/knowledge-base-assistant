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

    st.divider()
    st.write("### 📄 文档管理")

    # 上传
    uploaded_file = st.file_uploader(
        "上传文档",
        type=["pdf", "txt", "md", "docx"],
        help="支持 PDF、TXT、Markdown、docx",
    )
    if uploaded_file is not None:
        if st.button("上传并重建索引"):
            with st.spinner("处理中，可能需要几十秒..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    resp = requests.post(f"{API_URL}/upload", files=files, timeout=300)
                    if resp.status_code == 200:
                        st.success(f"✅ {uploaded_file.name} 上传成功")
                    else:
                        st.error(f"❌ 上传失败: {resp.text}")
                except Exception as e:
                    st.error(f"❌ 请求失败: {e}")

    st.divider()

    # 文档列表 + 删除
    try:
        resp = requests.get(f"{API_URL}/documents", timeout=10)
        docs = resp.json().get("documents", [])
    except Exception:
        docs = []

    if docs:
        st.caption(f"共 {len(docs)} 个文档")
        for doc in docs:
            col1, col2 = st.columns([4, 1])
            col1.write(f"📄 {doc['name']}  \n<small>({doc['size_kb']} KB)</small>", unsafe_allow_html=True)

            # 删除按钮
            if col2.button("🗑️", key=f"del_{doc['name']}", help="删除"):
                st.session_state[f"confirm_delete_{doc['name']}"] = True

            # 确认删除
            if st.session_state.get(f"confirm_delete_{doc['name']}"):
                st.warning(f"确认删除 **{doc['name']}** ？")
                c1, c2 = st.columns(2)
                if c1.button("✅ 确认", key=f"yes_{doc['name']}"):
                    try:
                        resp = requests.post(
                            f"{API_URL}/delete",
                            params={"filename": doc["name"]},
                            timeout=300,
                        )
                        if resp.status_code == 200:
                            st.success(f"已删除 {doc['name']}")
                            st.session_state[f"confirm_delete_{doc['name']}"] = False
                            st.rerun()
                        else:
                            st.error(f"删除失败: {resp.text}")
                    except Exception as e:
                        st.error(f"请求失败: {e}")
                if c2.button("❌ 取消", key=f"no_{doc['name']}"):
                    st.session_state[f"confirm_delete_{doc['name']}"] = False
                    st.rerun()
    else:
        st.caption("暂无文档")

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