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
| 嵌入模型 | BAAI/bge-m3 —— **可切换**：本地 sentence-transformers（默认）或 SiliconFlow API |
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
│   ├── loader.py           # 文档加载与切分调度
│   ├── md_chunker.py       # Markdown 切分（代码块原子化，见技术点 3）
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
├── scripts/                # 诊断与评估脚本
│   └── eval_chunks.py      # 分块质量报告（基线对比）
├── tests/                  # pytest 单测
│   ├── test_md_chunker.py  # 切分器边界用例 + 真实数据不变量
│   ├── eval_chunk_baseline.json   # 改切分前的质量基线
│   └── eval_chunk_after.json      # 改切分后的快照
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

# ========== 向量化：后端二选一 ==========
# local = 本地 sentence-transformers，零成本、离线、无限流（默认推荐）
# api   = 走下面的 OpenAI 兼容端点，按量计费（余额不足会报 402）
EMBEDDING_BACKEND=local
LOCAL_EMBEDDING_DEVICE=cpu

# 仅 EMBEDDING_BACKEND=api 时才需要
SILICONFLOW_API_KEY=your_siliconflow_api_key
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

# ========== PostgreSQL（持久化记忆） ==========
DB_URL=postgresql://your_user:your_password@127.0.0.1:5432/your_db?sslmode=disable

# ========== Redis（检索缓存） ==========
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# ========== HuggingFace 缓存与镜像 ==========
# 模型体积大（bge-m3 约 2.2 GB、reranker 约 2.2 GB），建议放空间充足的盘
HF_HOME=D:\hf-cache
# 中国大陆必须开镜像，否则直连 huggingface.co 会超时（WinError 10060）
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

**⚠️ 注意事项**：

| 配置项 | 说明 |
|:---|:---|
| `DB_URL` 必须用 `127.0.0.1` | Windows 上 `localhost` 会解析到 IPv6，连不上 |
| `EMBEDDING_BACKEND` | `local` 首次会下载模型（约 2.2 GB），之后完全离线；`api` 按量计费 |
| `HF_HOME` | 模型缓存目录。**必须在 huggingface/torch 导入之前生效**，放 `.env` 由 `config.py` 加载 |
| `HF_ENDPOINT` | 中国大陆用户保留，否则下载会超时 |
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

### 3. 代码块原子化切分：保证代码围栏永不被切断

**问题**：原来的流程是「`MarkdownHeaderTextSplitter` 按标题切 → 超长块交给
`RecursiveCharacterTextSplitter`」。第二步的 separators 是
`["\n\n","\n","。","！","？","；","，",""]` —— **完全不认识 ``` 围栏**，
于是在代码块中间下刀。实测 227 个块里 **41 个（18.1%）** 围栏只剩一半：

```
## 5.7 反射
```java          ← 开
...              ← 这个块里没有闭合围栏
```

危害不是"看起来不整齐"：半截代码的嵌入向量偏离语义中心，Rerank 会给它低分，
于是正确答案被排到后面。

**先试了 LangChain 自带的两个切分器，都不够用**（都实测过）：

| 切分器 | 结果 |
|:---|:---|
| `MarkdownHeaderTextSplitter` | 围栏感知安全，但不管超长块；交给递归切分器后围栏就破了 |
| `ExperimentalMarkdownSyntaxTextSplitter` | 思路好（代码块单独成块 + `Code` 元数据），但：围栏正则锚定行首，缩进的 ``` 识别不了（仍有 10 个坏块）；未闭合围栏时返回空串**静默丢掉文件剩余内容**；没有尺寸聚合（539 块里 175 个碎片块） |

所以借它的思路，自己写 `src/md_chunker.py`。

**方案**：把文档看作有序的「原子」序列，按规则装箱。

```
header  标题行（h1-h3）。当前块已有正文/代码则封口；否则继续累积
        （连续标题不会各自生成碎片块）
code    围栏代码块，整块不可切。下一个代码块到达时才封口（懒封口），
        这样代码块下面那句解释能并入同一块
prose   正文段落（按空行切）。只有正文才交给递归切分器 —— 正文里没有围栏，安全
hr      水平分割线，作为块边界，本身不进块
```

**为什么能做到 0 坏块**：切分器从头到尾没有"看见"围栏内部的文本 —— 代码块是
被整块搬进块里的。这不是调参调到 0，是结构上不可能坏。

围栏识别比 CommonMark 更宽松：除标准的 0-3 空格缩进，还容忍列表标记
（`- ```java`）和更深缩进。**宁可把缩进代码块误判成围栏（块变大），
也不能漏判围栏（块被切坏）。**

未闭合围栏的处理：剩余内容整体当作代码块保留，并打 `unclosed_code` 元数据。
**不伪造一个闭合围栏**，让评估报告能看见"源文件有问题"。

**效果**（`scripts/eval_chunks.py --compare`，同一份 data/）：

| 指标 | 改前 | 改后 |
|:---|:---|:---|
| 坏围栏块 | 41（18.1%） | **0** |
| 块数 | 227 | 277 |
| 过小块（<50） | 1 | 2 |
| 超长块 | 0 | 5（代码块不可切，已知取舍） |
| 块大小中位数 | 351 | 280 |
| 每块最多代码块数 | 7 | **1** |

超长的 5 个块就是最长的 5 个代码块（最大 1381 字符）。取舍很明确：
**一块完整的长代码，好过两块残缺的短代码。**

### 4. 索引复用：不要每次启动都重新嵌入

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

### 5. 持久化记忆：跨会话记住用户

**方案**：`AsyncPostgresSaver` + `AsyncConnectionPool`。

**关键**：异步调用（`ainvoke`）必须配异步 checkpointer，否则报 `NotImplementedError`。

### 6. SSE 流式输出：打字机效果

**后端**：用 `StreamingResponse` 返回 SSE，`agent.astream()` 逐块推送。

**前端**：用 `st.write_stream()` 逐字渲染。

### 7. 启动优化

| 优化 | 效果 |
|:---|:---|
| `@lru_cache` 缓存模型 | 同一进程内不重复加载 |
| 延迟加载 Reranker | 启动时不加载，第一次检索才加载 |
| 索引复用（见技术点 4） | 第二次起跳过全部嵌入调用 |
| 拆分 `build_retriever` / `build_agent` | 上传文档时只重建 agent，复用连接池 |

### 8. 嵌入后端可切换 + 计费错误翻译

**为什么单独处理嵌入**：它是 RAG 里唯一"每次都要花钱"的环节 —— 建索引时每个块要
嵌入一次，用户每提一个问题，问题本身也要嵌入一次才能检索。所以它出错时影响面最大，
而原始错误长这样：

```
openai.APIStatusError: Error code: 402 -
{'code': 30001, 'message': 'Sorry, your account balance is insufficient'}
```

这句话对使用者没有任何可操作性。`src/embeddings.py` 做两件事：

**① 后端可切换**（`EMBEDDING_BACKEND`）

| 后端 | 特点 |
|:---|:---|
| `local` | 本地 `sentence-transformers`，零成本、离线、无限流；首次需下载模型（约 2.2 GB） |
| `api` | 走 OpenAI 兼容端点（SiliconFlow 等），按量计费、需要网络 |

本地嵌入用 `normalize_embeddings=True`，与 API 端保持一致的余弦口径。

**② 错误分类与翻译**（`GuardedEmbeddings` 包装层）

| 类型 | 判据 | 提示内容 |
|:---|:---|:---|
| billing | HTTP 402 / `30001` / `insufficient` | 余额不足 + 充值或改 `EMBEDDING_BACKEND=local` |
| auth | HTTP 401 / `invalid api key` | 检查 key 是否过期或被禁用 |
| ratelimit | HTTP 429 / `rate limit` | 等待重试，或切本地绕开限流 |
| network | 超时 / 连接失败 | 检查网络/代理 |

无法归类的异常**原样抛出**，不做臆测；提示里始终附原始错误摘要（截断 300 字符）
方便排查，并用 `raise ... from e` 保留异常链。

**为什么包在嵌入对象上，而不是在每个调用点 try/except**：嵌入有两个调用点
（建索引的 `embed_documents`、检索的 `embed_query`），而且都被埋在 Chroma 内部，
外部很难逐个包裹。包装嵌入对象本身是唯一同时覆盖两条路径的位置。

**延迟加载**：`GuardedEmbeddings` 构造时不加载模型，首次真正嵌入时才加载。
实测 `load_vectorstore()` 只要 0.11s（不触发模型加载），首次检索才付模型加载代价。

**启动耗时实测拆分**（本机 CPU）：

| 阶段 | 耗时 |
|:---|:---|
| import `src.agent`（含 torch/transformers） | **12.9s** |
| `build_retriever()`（复用索引） | 0.86s |
| 首次检索（触发本地模型加载） | 0.35s |
| 后续检索 | 0.01s |

那 12.9s 是既有开销 —— 用 `EMBEDDING_BACKEND=api` 时同样是 12.5s，
因为 `langchain_text_splitters` 的导入链会拉进 `sentence_transformers` → `torch`。
与"用本地还是 API 嵌入"无关。

**切换后端必须重建索引**：就算模型同名，本地推理与 API 返回的向量也存在细微差异，
混用会让相似度不可比。所以 `embedding_backend` 也进了索引签名（见技术点 4）。

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

### HF_HOME / HF_ENDPOINT 必须在导入之前生效

`huggingface_hub` 在**导入时**就把缓存路径和端点算成常量：

```python
import huggingface_hub.constants as C
C.HF_HUB_CACHE   # D:\hf-cache\hub —— 导入那一刻就固定了
C.ENDPOINT       # https://hf-mirror.com
```

之后再改 `os.environ` 不会生效。踩过三次：

1. **下载走了直连**，报 `WinError 10060`（连接超时）—— 因为 `HF_ENDPOINT` 只在
   `run.py` / `cli.py` 里设置，而脚本直接 import 时没人设它。
2. **模型下到一半**，留下缺文件的缓存（缺 `1_Pooling/config.json`），
   加载时报一个看着完全无关的错：
   `TypeError: Pooling.__init__() missing 1 required positional argument: 'embedding_dimension'`
   —— 不是版本不兼容，是缓存不完整。
3. **大文件下载 401**：新版 `huggingface_hub` 默认走 Xet 存储后端
   （`cas-server.xethub.hf.co`），在镜像/未登录场景下返回
   `401 Unauthorized`。解法是 `HF_HUB_DISABLE_XET=1` 走普通 HTTP。
4. **每次启动都联网检查模型更新**：镜像超时重试会让一次本该复用的启动
   从 **1.1 秒变成 126 秒**。模型已缓存后设 `HF_HUB_OFFLINE=1`，
   启动回到 1 秒级（需要下载新模型时临时改回 0）。

**修法**：把 `HF_HOME` / `HF_ENDPOINT` / `HF_HUB_DISABLE_XET` 放进 `.env`，
由 `src/config.py` 的 `load_dotenv()` 加载；同时保证 `src/loader.py`
**先 import `src.config`** 再 import langchain / torch 相关模块
（文件顶部有注释说明原因）。

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

## 📊 分块质量评估

切分是 RAG 的第一道工序，且它的错误会污染后面所有指标（检索 Recall 偏低，
可能是因为检索器差，也可能是因为块本身残缺）。所以单独用一层静态检查量化它。

```bash
python scripts/eval_chunks.py                       # 打报告
python scripts/eval_chunks.py --save tests/x.json   # 存基线快照
python scripts/eval_chunks.py --compare tests/x.json    # 与基线对比
```

不看任何 API，4 组指标：

| 组 | 指标 |
|:---|:---|
| 规模 | 文档数、块数、每篇分布 |
| 大小 | min/中位/P90/max、过小块（<50）、超长块（>CHUNK_SIZE） |
| 结构 | **代码围栏配对率**、坏块分类、代码字符占比、标题元数据覆盖 |
| 冗余 | 完全重复块、近似重复对（Jaccard ≥ 0.85） |

对比时有两道保护，避免读出误导性的差值：

1. **切分参数不一致就拒绝比较**（退出码 2）—— 块边界变了，差值没有意义
2. **数据文件变了会明确警告** —— 否则会把"换了文档"的效果误算到"改了切分器"头上

对应单测：

```bash
python -m pytest tests/test_md_chunker.py -q
```

17 个用例，覆盖裸围栏 / 带语言标识 / 列表缩进围栏 / 波浪线 / 未闭合围栏 /
代码内 `#` 注释 / 代码内空行 / 超长代码块，外加真实文档的"围栏全部成对"
和"非空白字符不丢失"两条不变量。

---

## 📌 后续计划

- [x] 文件删除功能
- [x] Redis 缓存检索结果
- [x] 索引复用（避免每次全量重建）
- [x] 来源引用（检索结果里带 `[来源: 文件名]`）
- [x] 代码块原子化切分（坏围栏 18.1% → 0%）
- [x] 分块质量评估（`scripts/eval_chunks.py`，支持基线对比）
- [x] 单元测试（`tests/test_md_chunker.py` + `tests/test_embeddings.py`，43 个用例）
- [x] 嵌入后端可切换（本地 / API），计费与限流错误给可操作提示
- [ ] 索引增量更新（只重新嵌入变化的块，而不是全量重建）
- [ ] 检索层评估（Recall@k / MRR，对比 纯向量 / 混合 / 混合+Rerank）
- [ ] RAGAS 生成层评估（`kb evaluate` 目前会提示"尚未实现"）
- [ ] Docker 部署
- [ ] 多用户隔离 + 权限控制

### 已知限制

- **超长代码块**：代码块不可切，超过 `CHUNK_SIZE` 时整块保留。
  当前 277 块里有 5 块超长（最大 1412 字符）。取舍是"一块完整的长代码
  好过两块残缺的短代码"。
- **源文档重复块**：`langchain.md` 里同一段 `load_dotenv` 代码出现在两章，
  隔离成独立块后被识别为完全重复（2 块）。这是源文档的问题，不是切分的。
- **本地嵌入的启动成本**：`import src.agent` 要 12.9s（torch/transformers 导入链，
  与后端无关），首次检索再付 0.35s 模型加载。模型加载已是延迟的，
  剩下的导入开销要拆 `langchain_text_splitters` 的导入链才能省掉。
- **当前默认本地模型是轻量版**：`BAAI/bge-small-zh-v1.5`（512 维，92 MB）。
  `BAAI/bge-m3`（1024 维，2.2 GB）质量更好，但权重下载在当前网络下反复失败
  （Xet 401 / 镜像卡在 190 MB）。下载好后只改 `.env` 一行 `EMBEDDING_MODEL` 即可，
  索引会自动重建。
- **嵌入签名只含模型名与后端**：如果换了 API 服务商但模型名相同，
  清单不会察觉（两家返回的向量可能有细微差异）。要更严可以把 `base_url` 也纳入签名。
- **重建期间并发请求不设防**：`index_lock` 只串行化重建之间，不阻塞查询。
  重建窗口内的请求可能失败。彻底解决需要"影子索引 + 原子切换"。
- **重建失败会留下不可用的索引**：`build_retriever` 先删库再重建，
  中途失败则服务活着但查不了，需重启。可通过 `GET /rebuild/status` 感知。
  清单在删库前就被移除，所以失败后不会被误判为"索引新鲜"。
- **`thread_id` 由客户端提供**，没有鉴权。本地开发无妨，上公网前必须加隔离。

---

## 📄 License

MIT
