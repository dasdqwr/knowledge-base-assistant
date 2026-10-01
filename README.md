# 个人知识库问答助手

基于 LangChain + LangGraph 的 RAG 知识库助手，支持混合检索、Rerank 重排序、持久化记忆和 Web 聊天界面。

> 一个从零搭建的 RAG 项目，记录完整开发过程。重点不是“完美”，而是“踩过的坑和优化思路”。

---

## ✨ 功能

- 🔍 **混合检索**：BM25 + 向量检索 + RRF 融合，解决向量检索对精确 API 名不敏感的问题
- 🎯 **Rerank 重排序**：用 `bge-reranker-v2-m3` 精排，目标文档从第 5 名提升到第 1 名
- 🧠 **持久化记忆**：PostgreSQL + `AsyncPostgresSaver`，跨会话记住用户
- 🤖 **Agent 工具调用**：知识库检索 + 计算器双工具，自主决策
- 💬 **Web 界面**：FastAPI + Streamlit，浏览器可访问
- ⚡ **SSE 流式输出**：打字机效果，逐字显示回答
- 📄 **文档上传**：支持 PDF / TXT / MD / DOCX，运行时重建索引

---

## 🛠️ 技术栈

| 层面 | 技术 |
|:---|:---|
| 框架 | LangChain |
| 后端 | FastAPI、Uvicorn |
| 前端 | Streamlit |
| 数据库 | PostgreSQL（记忆）、Chroma（向量） |
| 嵌入模型 | BAAI/bge-m3 |
| 重排模型 | BAAI/bge-reranker-v2-m3 |
| 分词 | jieba + 正则 |
| LLM | DeepSeek（对话）、SiliconFlow（嵌入） |

---

## 📁 项目结构

```
knowledge-base-assistant/
├── src/
│   ├── __init__.py
│   ├── config.py           # 配置（绝对路径）
│   ├── model.py            # 模型初始化（含 @lru_cache）
│   ├── loader.py           # 文档加载与切分
│   ├── vectorstore.py      # 向量存储
│   ├── rag_chain.py        # 混合检索 + 分词
│   └── agent.py            # Agent 构建
├── app.py                  # FastAPI 后端
├── streamlit_app.py        # Streamlit 前端
├── run.py                  # 后端启动入口
├── main.py                 # 命令行入口
├── scripts/                # 调试脚本
├── data/                   # 文档目录
├── .streamlit/
│   └── config.toml         # Streamlit 配置
├── .env.example            # 环境变量模板
├── .gitignore
└── requirements.txt
```

---

## 🚀 快速开始

### 1. 环境准备

**前置要求**：Python 3.10+、Conda（或 venv）

```bash
# 创建环境
conda create -n langchain_env python=3.12
conda activate langchain_env

# 安装依赖
pip install -r requirements.txt
```

如果 `requirements.txt` 不存在，手动安装核心依赖：

```bash
pip install langchain langchain-openai langchain-community langchain-huggingface
pip install langchain-text-splitters langgraph langgraph-checkpoint-postgres
pip install chromadb "psycopg[binary,pool]"
pip install sentence-transformers rank-bm25 jieba
pip install fastapi uvicorn streamlit requests
pip install python-dotenv docx2txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

然后编辑 `.env`，填入你自己的配置：

```env
# ========== 对话模型（DeepSeek） ==========
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# ========== 向量化模型（SiliconFlow） ==========
SILICONFLOW_API_KEY=your_siliconflow_api_key
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

# ========== PostgreSQL（持久化记忆） ==========
DB_URL=postgresql://your_user:your_password@127.0.0.1:5432/your_db?sslmode=disable

# ========== HuggingFace 镜像（中国大陆用户建议开启） ==========
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

**⚠️ 注意事项**：

| 配置项 | 说明 |
|:---|:---|
| `DEEPSEEK_API_KEY` | DeepSeek 的 API Key |
| `SILICONFLOW_API_KEY` | SiliconFlow 的 API Key |
| `DB_URL` | 换成你自己的用户名、密码、数据库名 |
| `DB_URL` 必须用 `127.0.0.1` | Windows 上 `localhost` 会解析到 IPv6，连不上 |
| `HF_ENDPOINT` | 中国大陆用户保留，其他地区可删除 |

### 3. 启动 PostgreSQL

**方式一：Ubuntu / Debian 本地安装**

```bash
sudo apt install postgresql postgresql-contrib
sudo -u postgres psql
```

在 psql 里创建用户和数据库：

```sql
CREATE USER langchain_user WITH PASSWORD 'your_password';
CREATE DATABASE langchain_db OWNER langchain_user;
GRANT ALL PRIVILEGES ON DATABASE langchain_db TO langchain_user;
\q
```

配置远程连接：

```bash
# 修改监听地址
sudo nano /etc/postgresql/18/main/postgresql.conf
# 找到并改成：listen_addresses = '*'

# 添加认证规则
sudo nano /etc/postgresql/18/main/pg_hba.conf
# 在末尾添加：host all all 0.0.0.0/0 scram-sha-256

sudo systemctl restart postgresql
```

**方式二：Windows 本地安装**

去 https://www.postgresql.org/download/windows/ 下载安装，记住密码，安装后服务会自动启动。

**方式三：虚拟机**

在 VMware / VirtualBox 里装 Ubuntu，再按方式一配置。Windows 主机通过端口转发访问虚拟机的 5432 端口。

### 4. 放入文档

把 PDF / TXT / MD / DOCX 文件放到 `data/` 目录：

```bash
cp your_docs/*.pdf data/
```

首次启动后端时会自动加载、切分、建索引。

### 5. 启动后端

```bash
python run.py
```

看到以下输出说明成功：

```
🚀 正在启动服务，初始化 Agent...
✅ 初始化完成
INFO:     Application startup complete.
```

### 6. 启动前端

**新开一个终端**（后端那个别关）：

```bash
streamlit run streamlit_app.py
```

### 7. 访问

浏览器打开 http://127.0.0.1:8501

---

## 💡 核心技术点

### 1. 混合检索：解决“明明有关键词却搜不到”

**问题**：向量检索擅长语义理解，但对 `init_chat_model` 这种精确 API 名不敏感。

**方案**：BM25 + 向量双路召回，用 RRF 融合。

```python
def build_hybrid_retriever(vectorstore, chunks):
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    bm25_retriever = BM25Retriever.from_documents(
        chunks, preprocess_func=preprocess
    )
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.7, 0.3],
    )
```

**自定义分词器**（关键）：

```python
def preprocess(text: str) -> list[str]:
    """英文标识符整体保留，中文用 jieba 分词"""
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|\d+', text)
    chinese_text = re.sub(r'[a-zA-Z0-9_\s]+', ' ', text)
    tokens.extend(jieba.lcut(chinese_text))
    return tokens
```

### 2. Rerank 重排序：把最相关的排到第一

**问题**：混合检索后，真正讲某个 API 的块排不上来，因为其他块“顺带提到”了关键词。

**方案**：用 CrossEncoder 精排。

```python
def rerank_and_format(question):
    docs = retriever.invoke(question)[:10]
    pairs = [[question, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in ranked[:TOP_K]]
    return "\n\n".join(doc.page_content for doc in top_docs)
```

### 3. 持久化记忆：跨会话记住用户

**方案**：`AsyncPostgresSaver` + `AsyncConnectionPool`。

**关键**：异步调用（`ainvoke`）必须配异步 checkpointer，否则报 `NotImplementedError`。

### 4. SSE 流式输出：打字机效果

**后端**：用 `StreamingResponse` 返回 SSE，`agent.astream()` 逐块推送。

**前端**：用 `st.write_stream()` 逐字渲染。

### 5. 启动优化：从 30 秒到 3.3 秒

| 优化 | 效果 |
|:---|:---|
| `@lru_cache` 缓存模型 | 同一进程内不重复加载 |
| 拆分 `build_retriever` / `build_agent` | 上传文档时只重建 agent，复用连接池 |
| 延迟加载 Reranker | 启动时不加载，第一次检索才加载 |

---

## 📌 后续计划

- [ ] 文件删除功能
- [ ] 来源引用（回答里标注文档来源）
- [ ] Redis 缓存检索结果
- [ ] RAGAS 评估体系
- [ ] Docker 部署
- [ ] 多用户隔离 + 权限控制

---

## 📄 License

MIT