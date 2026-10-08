import ast
import gc
import operator
import shutil
import time

from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.tools import tool
from pathlib import Path
from langchain.agents import create_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from src.cache import make_key, get_cached, set_cached
from src.config import TOP_K, CHROMA_DIR, DB_URL, RERANK_CANDIDATES
from src.index_meta import (
    clear_manifest,
    current_params,
    file_stats,
    is_index_fresh,
    write_manifest,
)
from src.loader import load_documents, split_documents
from src.model import get_model, get_reranker
from src.rag_chain import build_hybrid_retriever
from src.vectorstore import build_vectorstore, load_vectorstore


# ========== 向量库句柄管理 ==========
# 为什么需要这个：Chroma 会把 HNSW 的 .bin 文件映射进内存（或持有打开句柄），
# 只要 vectorstore 对象还活着，Windows 就不允许删除那些文件。
# 旧代码每次启动都无条件 rmtree，那时进程里的 vectorstore 还没建出来，
# 所以从没暴露过这个问题；一旦支持"复用"，同进程里就会先加载再删除，
# rmtree 立刻报 PermissionError: [WinError 32]。
# 所以重建前必须显式释放句柄。

_active_vectorstores: list = []


def _clear_shared_system_cache() -> bool:
    """清空 chromadb 的进程级 System 缓存，返回是否成功。

    追查到的持有链（这才是真正让句柄不释放的东西）：

        Chroma → _client → Client → _system → System
                                                   ├─ sqlite 连接 → chroma.sqlite3
                                                   └─ mmap        → data_level0.bin
        System 被 SharedSystemClient._identifier_to_system
        这个「类属性字典」按住，key 是持久化目录的绝对路径。

    所以只 del 掉自己的 vectorstore 没用：Chroma 对象会被 gc 回收，
    但 System 仍被那张类级字典引用着，句柄继续开着，rmtree 报 WinError 32。

    clear_system_cache() 把 _identifier_to_system / _identifier_to_refcount
    两张表整个丢弃（= {}）。它**不**调用 system.stop()，只是丢掉 Python 引用。

    注意：仅靠它不足以释放句柄。HNSW 的 .bin 是内存映射，由 Rust 层持有，
    Python 的 GC 回收不到那里 —— 实测「只清缓存」时 rmtree 依然对
    data_level0.bin 报 WinError 32。真正拆掉映射的是 client.close()
    走的 _release_system() → system.stop()。所以本函数只作兜底。
    """
    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
        return True
    except Exception as e:
        print(f"⚠️ 清理 chromadb System 缓存失败（{type(e).__name__}: {e}）")
        return False


def release_all_vectorstores() -> int:
    """释放向量库句柄，返回显式 close 掉的客户端个数。

    两条路都要走，因为它们的作用**不同**，实测缺一不可：

    1. client.close() —— 走 chromadb 的正常释放路径 _release_system()，
       引用计数归零时调用 system.stop()，真正拆掉 Rust 层的 HNSW 内存映射。
       **这一步不能省**：clear_system_cache() 只是把 Python 引用丢掉，
       而内存映射由 C/Rust 层持有，Python 的 GC 回收不了它 ——
       实测只清缓存会在 rmtree 时对 data_level0.bin 报 WinError 32。

    2. clear_system_cache() —— 兜底清空整张进程级缓存，
       覆盖"没被登记过的" System（例如别的代码路径直接 Chroma(...) 建的）。

    技术债说明：第①步用 getattr(vs, "_client") 访问 langchain_chroma 的
    私有字段。之所以接受这个债：它是唯一能触发 system.stop() 的出口，
    而 stop() 是释放内存映射的必要条件。用 hasattr 探测，库改名时不至于崩，
    会退化成只走第②步（那时第②步的不足会暴露出来，但至少有日志可查）。

    幂等，多调几次无害 —— 资源清理宁可多做一次，不能漏。
    """
    global _active_vectorstores

    released = 0
    for vs in _active_vectorstores:
        client = getattr(vs, "_client", None)
        if client is None:
            continue
        try:
            client.close()          # → _release_system() → system.stop()
            released += 1
        except Exception as e:
            print(f"⚠️ 关闭向量库客户端失败（{type(e).__name__}: {e}）")
    _active_vectorstores = []

    _clear_shared_system_cache()

    return released


def remove_chroma_dir(retries: int = 10, delay: float = 0.3) -> None:
    """可靠地删除向量库目录。

    三层保险：
    1. 先释放句柄（release_all_vectorstores → 清 chromadb 的进程级
       System 缓存），再 gc 一轮，让底层映射文件真正关闭；
    2. Windows 上句柄释放往往不是瞬时的，所以带退避重试，
       而不是一次失败就抛出去。

    改不动就抛最后一个异常，让调用方看到真实原因。
    """
    if not Path(CHROMA_DIR).exists():
        return

    release_all_vectorstores()
    gc.collect()

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            shutil.rmtree(CHROMA_DIR)
            if attempt > 1:
                print(f"   第 {attempt} 次尝试删除成功")
            return
        except Exception as e:
            last_err = e
            gc.collect()
            time.sleep(delay)

    raise last_err


def build_retriever(force: bool = False):
    """构建检索器。

    默认会先判断现有向量库能否复用（内容未变就不重新嵌入），
    需要重建时才删库 + 全量构建。详见 src/index_meta.py。

    force=True 时无条件重建，忽略新鲜度判断。
    """
    # ① 加载 + 切分（BM25 需要 chunks，所以本地这步省不掉；
    #    好在实测很便宜：3 个文档 277 个块只要 0.02 秒）
    print("加载文档...")
    docs = load_documents()
    if not docs:
        raise FileNotFoundError("data/ 目录下没有文档")
    chunks = split_documents(docs)

    # ② 判断向量库能否复用
    vectorstore = None
    if force:
        print("重建向量库：指定了 force")
    else:
        snapshot = file_stats()
        params = current_params()
        fresh, reason, refresh_meta = is_index_fresh(snapshot, params)

        if fresh and Path(CHROMA_DIR).exists():
            try:
                vectorstore = load_vectorstore()
                _active_vectorstores.append(vectorstore)
                print(f"♻️ 复用已有向量库：{reason}")
                if refresh_meta:
                    # 内容没变，只是 mtime/size 动了，刷新清单元数据即可
                    write_manifest(params, snapshot, len(chunks))
            except Exception as e:
                print(f"⚠️ 加载已有向量库失败（{type(e).__name__}: {e}），改为重建")
                vectorstore = None
        else:
            if not fresh:
                print(f"重建向量库：{reason}")
            else:
                print("重建向量库：清单显示新鲜但向量库目录不存在")

    # ③ 需要重建时：先删凭证，再删库，最后重建
    if vectorstore is None:
        # 顺序很重要：先让清单失效，再动向量库本身。
        # 这样中途崩溃会留下"没有清单"的状态，下次必然重建，
        # 不会把半成品索引当成有效索引一直用下去。
        clear_manifest()
        remove_chroma_dir()  # ← 释放句柄后清空旧索引
        vectorstore = build_vectorstore(chunks)  # ← 从零写新索引
        _active_vectorstores.append(vectorstore)
        # 构建成功，才重新发"索引完整"的凭证
        write_manifest(current_params(), file_stats(), len(chunks))
        print(f"✅ 向量库重建完成，{len(chunks)} 个块")

    # ④ 混合检索器
    retriever = build_hybrid_retriever(vectorstore, chunks)
    return retriever


# ========== 安全算术求值 ==========
# 原来 calculator 工具用 eval(expression)。LLM 会被提示注入影响，
# 而 eval 能执行任意表达式（`__import__('os').system(...)`），
# 等于把 RCE 交给模型。这里改成 ast 白名单求值：
# 只允许数字与 + - * / // % ** 括号，用 operator 模块绑定运算，
# 完全不经过 eval，所以不存在绕过。
#
# 保留原始实现供对照：
#     return str(eval(expression))

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
# ast 节点白名单：表达式本身 + 三种字面量/节点 + 运算符节点
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Load,
) + tuple(_BIN_OPS) + tuple(_UNARY_OPS)

# 指数上限：防止 `9**9**9` 这种把 CPU 挂死的表达式
_MAX_POW = 1000


def safe_eval_arithmetic(expression: str) -> float:
    """只计算纯算术表达式。任何函数调用、名字、属性访问都会抛 ValueError。"""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"表达式语法错误: {e.msg}") from e

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"表达式里不允许出现 {type(node).__name__}，只支持数字和 + - * / // % **"
            )
        # 只接受真正的 int / float。
        # 注意 isinstance(True, int) 是 True（bool 是 int 的子类），
        # 所以必须显式排除 bool，否则 "True" 会被当成数字放行。
        # 同时 complex 也不接受（ast.parse("1j") 会产出 complex 常量）。
        if isinstance(node, ast.Constant) and (
            isinstance(node.value, bool)
            or not isinstance(node.value, (int, float))
        ):
            raise ValueError(f"只允许数字，收到: {node.value!r}")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            op = _BIN_OPS[type(node.op)]
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and (
                abs(right) > _MAX_POW or abs(left) > _MAX_POW
            ):
                raise ValueError(f"指数过大（上限 {_MAX_POW}）")
            return op(left, right)
        if isinstance(node, ast.UnaryOp):
            return _UNARY_OPS[type(node.op)](_eval(node.operand))

        raise ValueError(f"无法计算的节点: {type(node).__name__}")

    return _eval(tree)


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
        docs = retriever.invoke(query)[:RERANK_CANDIDATES]
        if not docs:
            return "没有找到相关内容。"

        # Rerank 精排
        pairs = [[query, d.page_content] for d in docs]
        #逐个打分，返回一个分数列表：分数越高，说明这个块和查询越相关。
        scores = reranker.predict(pairs)
        #按分数降序排序
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [d for d, _ in ranked[:TOP_K]]

        # 来源只取文件名。metadata['source'] 是绝对路径，直接塞进上下文会
        # 把本机目录结构（用户名、盘符）喂给模型，还可能被它复述出来。
        result = "\n\n".join(
            f"[来源: {Path(d.metadata.get('source', '未知')).name}]\n{d.page_content}"
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
            return str(safe_eval_arithmetic(expression))
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

async def build_agent_with_memory(force: bool = False):
    """完整初始化：检索器 + 连接池 + Agent

    force=True 时无条件重建向量库（见 build_retriever）。
    """
    t0 = time.perf_counter()
    retriever = build_retriever(force=force)
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