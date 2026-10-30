"""嵌入（Embedding）的后端选择与错误翻译。

为什么要单独一层
----------------
嵌入是 RAG 唯一"每次都要花钱"的环节：建索引时每个块要嵌入一次，用户每提一个
问题，问题本身也要嵌入一次才能检索。所以它出错时影响面最大，而原始错误长这样：

    openai.APIStatusError: Error code: 402 -
    {'code': 30001, 'message': 'Sorry, your account balance is insufficient'}

对使用者来说这句话没有可操作性。本模块做两件事：

  1. 后端可切换：EMBEDDING_BACKEND = "api"（SiliconFlow 等 OpenAI 兼容端点）
     或 "local"（本地 sentence-transformers，零成本、离线、无速率限制）
  2. 把计费/鉴权/限流/网络四类错误翻译成可操作的中文提示，
     其余异常原样抛出（不吞、不猜）。

注意：切换后端会让已有向量库失效（不同实现的向量存在细微差异），
所以 EMBEDDING_BACKEND 也进了索引签名（见 src/index_meta.py），
换后端会自动触发重建。
"""
from __future__ import annotations

import threading

from langchain_core.embeddings import Embeddings

from src.config import (
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    LOCAL_EMBEDDING_DEVICE,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
)


class EmbeddingServiceError(RuntimeError):
    """嵌入服务不可用。消息已翻译成可操作的提示。"""


# ---------------------------------------------------------------------------
# 错误分类
# ---------------------------------------------------------------------------
def classify_error(exc: BaseException) -> str | None:
    """把异常归类成 billing / auth / ratelimit / network，无法归类返回 None。"""
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()

    # 余额 / 配额。SiliconFlow 用 30001 表示余额不足，HTTP 是 402。
    if status == 402 or "30001" in text or "insufficient" in text or "balance" in text:
        return "billing"
    if status == 401 or "unauthorized" in text or "invalid api key" in text:
        return "auth"
    if status == 429 or "rate limit" in text or "too many requests" in text:
        return "ratelimit"
    if status in (408, 504) or any(
        k in text for k in ("timeout", "timed out", "connection", "getaddrinfo", "ssl")
    ):
        return "network"
    return None


_MAX_RAW = 300


def _raw(exc: BaseException) -> str:
    """原始错误摘要，附在提示末尾方便排查，但截断避免刷屏。"""
    s = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    return s if len(s) <= _MAX_RAW else s[:_MAX_RAW] + "…"


def translate_error(exc: BaseException, *, phase: str) -> str:
    """生成面向使用者的提示。phase 例如 '建索引' 或 '检索'。"""
    kind = classify_error(exc)
    backend = EMBEDDING_BACKEND
    model = EMBEDDING_MODEL

    if kind == "billing":
        return (
            f"嵌入服务余额不足，{phase}失败。\n"
            f"  当前后端：{backend}（模型 {model}）\n"
            f"  可选处理：\n"
            f"    1) 给服务商账户充值后重试\n"
            f"    2) 改用本地嵌入，零成本离线运行：在 .env 里设置\n"
            f"         EMBEDDING_BACKEND=local\n"
            f"       （首次会下载模型，之后不再联网）\n"
            f"  原始错误：{_raw(exc)}"
        )
    if kind == "auth":
        return (
            f"嵌入服务鉴权失败，{phase}失败。\n"
            f"  当前后端：{backend}\n"
            f"  请检查 .env 里的 API Key 是否正确、是否已过期或被禁用，"
            f"或改用 EMBEDDING_BACKEND=local。\n"
            f"  原始错误：{_raw(exc)}"
        )
    if kind == "ratelimit":
        return (
            f"嵌入服务触发限流，{phase}失败。\n"
            f"  当前后端：{backend}\n"
            f"  可选处理：\n"
            f"    1) 等几十秒后重试\n"
            f"    2) 用 EMBEDDING_BACKEND=local 彻底绕开服务商限流\n"
            f"  原始错误：{_raw(exc)}"
        )
    if kind == "network":
        return (
            f"连接嵌入服务失败（网络问题），{phase}失败。\n"
            f"  当前后端：{backend}，地址：{SILICONFLOW_BASE_URL}\n"
            f"  请检查网络/代理，或改用 EMBEDDING_BACKEND=local。\n"
            f"  原始错误：{_raw(exc)}"
        )
    # 无法归类 —— 保持原始异常信息，不做臆测
    return f"嵌入调用失败，{phase}失败（未识别的错误类型）。\n  原始错误：{_raw(exc)}"


# ---------------------------------------------------------------------------
# 后端构造
# ---------------------------------------------------------------------------
def _build_api_embeddings() -> Embeddings:
    """OpenAI 兼容端点（SiliconFlow 等）。"""
    if not SILICONFLOW_API_KEY:
        raise EmbeddingServiceError(
            "缺少 SILICONFLOW_API_KEY，无法使用 API 嵌入。\n"
            "  请在 .env 里配置，或改用 EMBEDDING_BACKEND=local。"
        )
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=SILICONFLOW_API_KEY,
        base_url=SILICONFLOW_BASE_URL,
    )


def local_error_message(exc: BaseException) -> str:
    """把本地模型加载失败翻译成可操作提示。

    实际踩过的两个坑：
      · 直连 huggingface.co 超时（WinError 10060）→ 没配 HF_ENDPOINT
      · 下载中断留下不完整缓存 → 报一个看着无关的 Pooling 错误：
        TypeError: Pooling.__init__() missing ... 'embedding_dimension'
    """
    import os

    text = str(exc).lower()
    home = os.environ.get("HF_HOME", "~/.cache/huggingface")
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

    # 离线模式下模型没缓存（HF_HUB_OFFLINE=1 时不会联网）
    if "offline" in text or "localentrynotfound" in text:
        return (
            f"本地嵌入模型不在缓存里，而当前处于离线模式。\n"
            f"  当前模型：{EMBEDDING_MODEL}，缓存目录：{home}\n"
            f"  处理：把 .env 里的 HF_HUB_OFFLINE 改成 0（或删掉该行）后重试，"
            f"让它联网下载一次；下载完成后再改回 1 以加快启动。\n"
            f"  原始错误：{_raw(exc)}"
        )
    # 缓存不完整（下载中断的典型症状）
    if any(k in text for k in ("pooling", "embedding_dimension", "1_pooling", "no such file")):
        return (
            f"本地嵌入模型加载失败：缓存可能不完整（通常是下载中断）。\n"
            f"  当前模型：{EMBEDDING_MODEL}\n"
            f"  缓存目录：{home}\n"
            f"  处理：删除 {home}\\hub\\models--{EMBEDDING_MODEL.replace('/', '--')} 后重新运行，"
            f"让它完整重下。\n"
            f"  原始错误：{_raw(exc)}"
        )
    if any(k in text for k in ("10060", "timed out", "timeout", "connection",
                              "couldn't connect", "getaddrinfo")):
        return (
            f"下载本地嵌入模型失败（网络问题）。\n"
            f"  当前模型：{EMBEDDING_MODEL}\n"
            f"  当前镜像：{endpoint}\n"
            f"  处理：在 .env 里设置 HF_ENDPOINT=https://hf-mirror.com 后重试；\n"
            f"       注意该变量必须在 import 之前生效（放 .env 由 config 加载即可）。\n"
            f"  原始错误：{_raw(exc)}"
        )
    return (
        f"本地嵌入模型加载失败。\n"
        f"  当前模型：{EMBEDDING_MODEL}，缓存目录：{home}\n"
        f"  原始错误：{_raw(exc)}"
    )


def _build_local_embeddings() -> Embeddings:
    """本地 sentence-transformers。零成本、离线、无限流。"""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as e:  # pragma: no cover - 依赖缺失时给明确指引
        raise EmbeddingServiceError(
            "本地嵌入需要 langchain-huggingface 和 sentence-transformers。\n"
            "  安装：pip install langchain-huggingface sentence-transformers"
        ) from e

    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            # 与 API 端保持一致的归一化，避免余弦相似度口径不同
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"device": LOCAL_EMBEDDING_DEVICE},
        )
    except Exception as e:
        # 下载失败 / 缓存不完整 都会在这里抛，翻译成可操作提示
        raise EmbeddingServiceError(local_error_message(e)) from e


# ---------------------------------------------------------------------------
# 错误翻译 + 延迟加载包装
# ---------------------------------------------------------------------------
class GuardedEmbeddings(Embeddings):
    """延迟构造 + 错误翻译的嵌入包装。

    两个职责：

    1. **延迟加载**：构造时不加载模型，第一次真正嵌入时才加载。
       本地 bge 模型要 import torch + 初始化权重，实测约 10 秒；而 Chroma 在
       构造向量库时只需要一个"嵌入函数对象"，要到 add/query 才真正调用。
       延迟加载把这段时间从启动路径挪到第一次检索，启动 14s → 3s 左右。

    2. **错误翻译**：把计费/鉴权/限流/网络/缓存损坏等错误翻成可操作提示。
       放在这里而不是每个调用点，是因为嵌入有两个调用点
       （建索引的 embed_documents、检索的 embed_query），
       而且都被埋在 Chroma 内部，外部很难逐个包裹。
    """

    def __init__(self, factory) -> None:
        self._factory = factory
        self._inner = None
        self._lock = threading.Lock()

    def _ensure(self) -> Embeddings:
        if self._inner is not None:
            return self._inner
        with self._lock:
            if self._inner is None:
                try:
                    self._inner = self._factory()
                except EmbeddingServiceError:
                    raise
                except Exception as e:
                    raise EmbeddingServiceError(
                        f"初始化嵌入后端失败（{EMBEDDING_BACKEND}）。\n"
                        f"  原始错误：{_raw(e)}"
                    ) from e
        return self._inner

    @property
    def inner(self) -> Embeddings:
        """底层嵌入对象。访问即触发加载（测试与排障用）。"""
        return self._ensure()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._ensure().embed_documents(texts)
        except EmbeddingServiceError:
            raise
        except Exception as e:
            raise EmbeddingServiceError(translate_error(e, phase="建索引")) from e

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._ensure().embed_query(text)
        except EmbeddingServiceError:
            raise
        except Exception as e:
            raise EmbeddingServiceError(translate_error(e, phase="检索")) from e


def build_embeddings() -> Embeddings:
    """按 EMBEDDING_BACKEND 选后端，返回延迟构造 + 错误翻译的包装对象。"""
    backend = (EMBEDDING_BACKEND or "api").strip().lower()
    if backend == "local":
        factory = _build_local_embeddings
    elif backend == "api":
        factory = _build_api_embeddings
    else:
        raise EmbeddingServiceError(
            f"未知的 EMBEDDING_BACKEND={EMBEDDING_BACKEND!r}，只支持 'api' 或 'local'。"
        )
    return GuardedEmbeddings(factory)
