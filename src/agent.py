import time

from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.tools import tool
from pathlib import Path
from langchain.agents import create_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from src.cache import make_key, get_cached, set_cached
from src.config import TOP_K, CHROMA_DIR, DB_URL
from src.loader import load_documents, split_documents
from src.model import get_model, get_reranker
from src.rag_chain import build_hybrid_retriever
from src.vectorstore import build_vectorstore

def build_retriever():
    # ① 加载 + 切分（BM25 需要 chunks）
    print("加载文档...")
    docs = load_documents()
    if not docs:
        raise FileNotFoundError("data/ 目录下没有文档")
    chunks = split_documents(docs)
    # ② 全量重建向量库
    if Path(CHROMA_DIR).exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
    vectorstore = build_vectorstore(chunks)
    # ③ 混合检索器
    retriever = build_hybrid_retriever(vectorstore, chunks)
    return retriever


async def build_agent(retriever, pool):
    """用给定的 retriever 和 pool 创建 Agent"""
    reranker = get_reranker()

    # ④ 工具
    @tool
    async def search_knowledge_base(query: str) -> str:
        """搜索知识库，返回相关文档片段。当用户询问文档内容时使用。"""
        print(f"🔍 工具被调用，query = {query}")
        # 查缓存
        cache_key = make_key("retrieval", query)
        print(f"🔑 cache_key = {cache_key}")
        cached = await get_cached(cache_key)
        if cached:
            print(f"✅ 命中缓存: {query[:30]}")
            return cached
        # 没命中，走完整检索
        docs = retriever.invoke(query)[:10]
        if not docs:
            return "没有找到相关内容。"

        # Rerank 精排
        pairs = [[query, d.page_content] for d in docs]
        #逐个打分，返回一个分数列表：分数越高，说明这个块和查询越相关。
        scores = reranker.predict(pairs)
        #按分数降序排序
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [d for d, _ in ranked[:TOP_K]]

        result = "\n\n".join(
            f"[来源: {d.metadata.get('source', '未知')}]\n{d.page_content}"
            for d in top_docs
        )

        # 写入缓存
        success = await set_cached(cache_key, result, ttl=3600)
        if success:
            print(f"💾 已写入缓存: {cache_key}")

        return result

    @tool
    def calculator(expression: str) -> str:
        """计算数学表达式，例如 '(25 + 17) * 3'"""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"计算失败: {e}"

    await pool.open()  # 显式打开连接池
    # 持久化 checkpointer
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()  # 首次运行建表

    # ⑥ 创建 Agent
    model = get_model()
    return create_agent(
        model=model,
        tools=[search_knowledge_base, calculator],
        checkpointer=checkpointer,
        middleware=[  # ← 新增
            SummarizationMiddleware(
                model=model,  # 用同一个模型做摘要
                trigger=("tokens", 8000),  # 超过 8000 token 触发
                keep=("messages", 20),  # 保留最近 20 条
            ),
        ],
        system_prompt="""你是一个知识库助手。
    - 用户询问文档内容时，使用 search_knowledge_base 工具
    - 需要计算时，使用 calculator 工具
    - 如果知识库没有相关信息，诚实告知
    - 用中文回答""",
    )

async def build_agent_with_memory():
    """完整初始化：检索器 + 连接池 + Agent"""
    t0 = time.perf_counter()
    retriever = build_retriever()
    t1 = time.perf_counter()
    print(f"⏱️ 构建检索器: {t1 - t0:.2f}s")

    pool = AsyncConnectionPool(
        conninfo=DB_URL,
        min_size=0,
        max_size=20,
        timeout=10,  # ← 获取连接超时 10 秒
        open=False,  # 关键：不在构造函数中打开
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    t2 = time.perf_counter()
    print(f"⏱️ 连接池: {t2 - t1:.2f}s")

    agent = await build_agent(retriever, pool)
    t3 = time.perf_counter()
    print(f"⏱️ 构建 Agent: {t3 - t2:.2f}s")

    print(f"⏱️ 总耗时: {t3 - t0:.2f}s")
    return agent, pool