# 个人知识库问答助手

基于 LangChain + LangGraph 的 RAG 知识库助手，支持混合检索、Rerank 重排序、持久化记忆和 Web 聊天界面。

> 一个从零搭建的 RAG 项目，记录完整开发过程。重点不是"完美"，而是"踩过的坑和优化思路"。

---

## ✨ 功能

- 🔍 **混合检索**：BM25 + 向量检索 + RRF 融合，解决向量检索对精确 API 名不敏感的问题
- 🎯 **Rerank 重排序**：用 `bge-reranker-v2-m3` 精排，把真正相关的块排到第一
- ⚡ **索引复用**：按内容哈希判断，文档没变就不重新嵌入（首次 3.2s → 之后 0.15s）
- 🧠 **持久化记忆**：PostgreSQL + `AsyncPostgresSaver`，跨会话记住用户
- 💾 **检索缓存**：Redis 缓存检索结果，重复提问不再走一遍检索
- 🤖 **Agent 工具调用**：知识库检索 + 计算器双工具，自主决策
- 💬 **Web 界面**：FastAPI + Streamlit，浏览器可访问
- ⚡ **SSE 流式输出**：打字机效果，逐字显示回答
- 📄 **文档管理**：支持 PDF / TXT / MD / DOCX，上传 / 列表 / 删除，运行时重建索引
- ⌨️ **CLI 工具**：`kb ask / chat / upload / delete / rebuild / serve / web / info`

---

## 🛠️ 技术栈

| 层面 | 技术 |
|:---|:---|
| 框架 | LangChain + LangGraph |
| 后端 | FastAPI、Uvicorn |
| 前端 | Streamlit |
| 命令行 | Typer |
| 数据库 | PostgreSQL（对话记忆）、Chroma（向量）、Redis（检索缓存） |
| 嵌入模型 | BAAI/bge-m3（SiliconFlow API） |
| 重排模型 | BAAI/bge-reranker-v2-m3（本地 CrossEncoder） |
| 分词 | jieba + 正则 |
| LLM | DeepSeek（对话） |

---

## 📁 项目结构

```
knowledge-base-assistant/
├── src/
│   ├── config.py           # 配置、路径、检索参数、支持的文档后缀
│   ├── utf8.py             # 控制台 UTF-8 引导（Windows 必需，见"踩坑"）
│   ├── model.py            # 模型初始化（含 @lru_cache + 延迟加载 Reranker）
│   ├── loader.py           # 文档加载与分层切分
│   ├── vectorstore.py      # 向量库读写
│   ├── index_meta.py       # 索引新鲜度判断（清单 + 内容哈希）
│   ├── rag_chain.py        # 混合检索 + 分词器
│   ├── cache.py            # Redis 检索缓存
│   ├── document.py         # 文档列表 / 保存 / 删除（含路径穿越防护）
│   └── agent.py            # Agent 构建、索引复用、句柄管理
├── app.py                  # FastAPI 后端
├── streamlit_app.py        # Streamlit 前端
├── cli.py                  # Typer CLI（kb 命令）
├── run.py                  # 后端启动入口
├── main.py                 # 早期命令行入口（已被 cli.py 取代，保留作参考）
├── scripts/                # 调试与诊断脚本
├── data/                   # 文档目录
├── .streamlit/config.toml  # Streamlit 配置
├── .env.example            # 环境变量模板
├── pyproject.toml          # 依赖与工具配置
└── .gitignore
```

> 索引状态文件 `index_manifest.json` 会在首次构建时自动生成，已被 gitignore。

---

## 🚀 快速开始

### 1. 环境准备

**前置要求**：Python 3.10+、Conda（或 venv）

```bash
conda create -n langchain_env python=3.12
conda activate langchain_env

# 安装项目依赖（依赖声明在 pyproject.toml 里）
pip install -e .
```

想手动装核心依赖的话：

```bash
pip install langchain langchain-openai langchain-community langchain-huggingface
pip install langchain-text-splitters langgraph langgraph-checkpoint-postgres
pip install chromadb "psycopg[binary,pool]" redis
pip install sentence-transformers rank-bm25 jieba
pip install fastapi uvicorn streamlit requests typer
pip install python-dotenv docx2txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

然后编辑 `.env`：

```env
# ========== 对话模型（DeepSeek） ==========
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# ========== 向量化模型（SiliconFlow） ==========
SILICONFLOW_API_KEY=your_siliconflow_api_key
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

# ========== PostgreSQL（持久化记忆） ==========
DB_URL=postgresql://your_user:your_password@127.0.0.1:5432/your_db?sslmode=disable

# ========== Redis（检索缓存） ==========
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# ========== HuggingFace 镜像（中国大陆用户建议开启） ==========
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

**⚠️ 注意事项**：

| 配置项 | 说明 |
|:---|:---|
| `DB_URL` 必须用 `127.0.0.1` | Windows 上 `localhost` 会解析到 IPv6，连不上 |
| `HF_ENDPOINT` | 中国大陆用户保留，其他地区可删除 |
| Redis 未配置时 | 缓存功能会打印警告并降级为"每次都走检索"，不影响主流程 |

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

> ⚠️ **这台虚拟机没开时，端口 5432 上的 `vmnat.exe` 仍会接受 TCP 连接**，
> 表现为连接静默超时而不是"拒绝连接"，很容易误判成代码问题。用
> `python scripts/check_db.py` 可以在 5 秒内区分这两种情况。

### 4. 启动 Redis（可选，用于检索缓存）

```bash
# Ubuntu
sudo apt install redis-server && sudo systemctl start redis-server

# Windows：用 WSL 或 Memurai
```

### 5. 放入文档

把 PDF / TXT / MD / DOCX 文件放到 `data/` 目录：

```bash
cp your_docs/*.pdf data/
```

也可以在 Web 界面或 CLI 里上传。

### 6. 启动后端

```bash
python run.py
```

看到以下输出说明成功：

```
🚀 正在启动服务，初始化 Agent...
加载文档...
📄 切分为 277 个块
♻️ 复用已有向量库：索引新鲜，277 个块     ← 首次会是"✅ 向量库重建完成"
⏱️ 构建检索器: 0.15s
⏱️ 连接池: 0.00s
⏱️ 构建 Agent: 2.63s
⏱️ 总耗时: 2.78s
INFO:     Application startup complete.
```

### 7. 启动前端

**新开一个终端**（后端那个别关）：

```bash
streamlit run streamlit_app.py
```

### 8. 访问

浏览器打开 http://127.0.0.1:8501

---

## ⌨️ CLI 用法

安装后（`pip install -e .`）在任意目录可用 `kb` 命令：

```bash
kb ask "init_chat_model 怎么用？"    # 单次提问
kb chat                             # 交互式对话
kb upload data/doc.pdf              # 上传文档并重建索引
kb delete doc.pdf                   # 删除文档并重建索引
kb rebuild                          # 检查/重建索引
kb rebuild --force                  # 无条件重建（忽略内容哈希）
kb serve                            # 启动后端
kb web                              # 启动前端
kb info                             # 查看配置与文档列表
```

---

## 💡 核心技术点

### 1. 混合检索：解决"明明有关键词却搜不到"

**问题**：向量检索擅长语义理解，但对 `init_chat_model` 这种精确 API 名不敏感。

**方案**：BM25 + 向量双路召回，用 RRF 融合。

```python
def build_hybrid_retriever(vectorstore, all_chunks):
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    bm25_retriever = BM25Retriever.from_documents(all_chunks, preprocess_func=preprocess)
    bm25_retriever.k = RETRIEVAL_K
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=ENSEMBLE_WEIGHTS,      # [0.8, 0.2]，BM25 权重更高
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

**问题**：混合检索后，真正讲某个 API 的块排不上来，因为其他块"顺带提到"了关键词。

**方案**：用 CrossEncoder 精排。

```python
docs = retriever.invoke(query)[:RERANK_CANDIDATES]
pairs = [[query, d.page_content] for d in docs]
scores = reranker.predict(pairs)
ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
top_docs = [d for d, _ in ranked[:TOP_K]]
```

### 3. 索引复用：不要每次启动都重新嵌入

**问题**：早期实现每次启动都无条件 `rmtree(chroma_db)` + 全量重建，
277 个块要发 277 次嵌入请求、约 3.2 秒。

**方案**：用内容哈希判断索引是否新鲜，新鲜就直接加载。

```
index_manifest.json 的存在  ==  "索引完整且与当前 data/ 一致"
```

签名包含三部分：

| 类别 | 内容 |
|:---|:---|
| 文件 | 文件名 + 大小 + mtime + **内容 SHA256** |
| 切分参数 | `CHUNK_SIZE`、`CHUNK_OVERLAP`、`code_version`、`schema_version` |
| 嵌入模型 | `EMBEDDING_MODEL` |

**为什么用内容哈希而不是 mtime**：`mtime` 会被"复制时保留时间戳"骗过
（`git checkout`、`robocopy /COPYALL`、解压都会保持原时间）。`data/` 只有
152 KB，哈希成本可忽略。

**关键设计：清单的写入时机**

```
需要重建 → 先删清单 → 再删库 → 嵌入构建 → 全部成功后写清单
```

清单在 `build_vectorstore` 返回**之后**才落盘，所以它永远不会为半成品背书。
任何中途崩溃（Ctrl+C、网络断）都会留下"没有清单"的状态，下次必然重建。
这比记一个 `build_ok=false` 标志可靠 —— 进程被杀时你根本没机会写那个标志。

**效果**：首次 3.2s → 复用 0.15s（0 次嵌入调用）。

### 4. 持久化记忆：跨会话记住用户

**方案**：`AsyncPostgresSaver` + `AsyncConnectionPool`。

**关键**：异步调用（`ainvoke`）必须配异步 checkpointer，否则报 `NotImplementedError`。

### 5. SSE 流式输出：打字机效果

**后端**：用 `StreamingResponse` 返回 SSE，`agent.astream()` 逐块推送。

**前端**：用 `st.write_stream()` 逐字渲染。

### 6. 启动优化

| 优化 | 效果 |
|:---|:---|
| `@lru_cache` 缓存模型 | 同一进程内不重复加载 |
| 延迟加载 Reranker | 启动时不加载，第一次检索才加载 |
| 索引复用（见技术点 3） | 第二次起跳过全部嵌入调用 |
| 拆分 `build_retriever` / `build_agent` | 上传文档时只重建 agent，复用连接池 |

---

## 🕳️ 踩过的坑

### Windows 控制台 GBK 导致服务起不来

`print("🚀 ...")` 在中文 Windows 上会抛 `UnicodeEncodeError`：

```
sys.stdout.encoding = 'gbk'
sys.stdout.errors   = 'strict'      ← 无法编码就抛异常
sys.stderr.encoding = 'gbk'
sys.stderr.errors   = 'backslashreplace'   ← 转义成 \u2705，不抛
```

GBK 没有 emoji。而 `print` 异常会穿透 FastAPI 的 `lifespan`，让**整个服务启动失败** ——
一条日志语句把进程弄死。项目里有 80+ 处 emoji 输出，启动路径上就有 6 处。

**修法**：`src/utf8.py` 把标准流切到 UTF-8，并挂在 `src/config.py` 上。
因为 `config` 是每个入口最早导入的 src 模块，一处生效、全项目覆盖（含 `scripts/`）。

注意 stdout 和 stderr 的**错误处理器不同**：stdout 是 `strict`（会抛），
stderr 是 `backslashreplace`（转义后继续）。所以写向 stderr 的 emoji 不致命，
写向 stdout 的致命。

### Windows 文件句柄导致删不掉向量库

支持索引复用后，同进程内会"先加载、后删除"。而 Chroma 持有
`chroma.sqlite3` 的句柄和 `data_level0.bin` 的**内存映射**，Windows 不允许
删除被占用的文件，`rmtree` 报 `PermissionError: [WinError 32]`。

追查到的持有链：

```
Chroma → _client → Client → _system → System
                                         ├─ sqlite → chroma.sqlite3
                                         └─ mmap   → data_level0.bin
System 被 SharedSystemClient._identifier_to_system（类属性字典）按住
```

**`del` 掉自己的变量没用**：Chroma 对象会被 gc 回收，但 System 仍被那张
类级字典引用，句柄继续开着。`gc.collect()` 也救不了（这不是循环引用问题）。

**修法**：`release_all_vectorstores()` 先逐个 `client.close()`
（走 `_release_system()` → `system.stop()`，真正拆掉 Rust 层的内存映射），
再 `clear_system_cache()` 兜底。

⚠️ **只调 `clear_system_cache()` 不够** —— 它只是丢掉 Python 引用，
而内存映射由 Rust 层持有，Python 的 GC 回收不到那里。实测只清缓存时
`rmtree` 依然报 `WinError 32`。

### psycopg 异步模式与 ProactorEventLoop 不兼容

Windows 默认事件循环是 `ProactorEventLoop`，psycopg 异步模式不支持：

```
InterfaceError: Psycopg cannot use the 'ProactorEventLoop' to run in async mode
```

**修法**：入口处显式指定 SelectorEventLoop。

```python
asyncio.run(server.serve(), loop_factory=asyncio.SelectorEventLoop)
```

### Redis 用同步客户端而不是 redis.asyncio

Windows + SelectorEventLoop 下 `redis.asyncio` 首次连接容易超时。
同步客户端稳定，用 `asyncio.to_thread` 丢到线程池即可不阻塞事件循环。

---

## 🔍 诊断脚本

```bash
python scripts/check_db.py          # PostgreSQL 连通性（含协议握手预检）
python scripts/check_redis.py       # Redis 连通性
python scripts/check_async_redis.py # redis.asyncio 可用性对照
python scripts/debug_retrieval.py   # 检索效果对比（混合检索 vs 加 Rerank）
```

`check_db.py` 值得单独说：如果 5432 端口是虚拟机端口转发（`vmnat.exe`），
它会完成 TCP 握手却不转发数据，`psycopg.connect` 会**静默挂起**。
所以这个脚本先自己做一次 PostgreSQL 协议握手，5 秒内给出结论。

---

## 📌 后续计划

- [x] 文件删除功能
- [x] Redis 缓存检索结果
- [x] 索引复用（避免每次全量重建）
- [x] 来源引用（检索结果里带 `[来源: 文件名]`）
- [ ] 索引增量更新（只重新嵌入变化的块，而不是全量重建）
- [ ] 单元测试（pytest 配置已就绪，`tests/` 目录待建）
- [ ] RAGAS 评估体系（`kb evaluate` 目前会提示"尚未实现"）
- [ ] Docker 部署
- [ ] 多用户隔离 + 权限控制

### 已知限制

- **重建期间并发请求不设防**：`index_lock` 只串行化重建之间，不阻塞查询。
  重建窗口内的请求可能失败。彻底解决需要"影子索引 + 原子切换"。
- **重建失败会留下不可用的索引**：`build_retriever` 先删库再重建，
  中途失败则服务活着但查不了，需重启。可通过 `GET /rebuild/status` 感知。
- **`thread_id` 由客户端提供**，没有鉴权。本地开发无妨，上公网前必须加隔离。

---

## 📄 License

MIT
