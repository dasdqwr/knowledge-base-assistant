"""嵌入后端与错误翻译的单元测试（不联网、不加载真实模型）。"""
from __future__ import annotations

import pytest

from src import embeddings as E


class FakeStatusError(Exception):
    """模拟 openai.APIStatusError：带 status_code 属性。"""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(message)


# ---------------------------------------------------------------------------
# 错误分类
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "exc,expected",
    [
        (FakeStatusError(402, "Error code: 402"), "billing"),
        (FakeStatusError(400, "code': 30001, 'message': 'Sorry, your account balance is insufficient'"), "billing"),
        (Exception("insufficient balance"), "billing"),
        (FakeStatusError(401, "Unauthorized"), "auth"),
        (Exception("invalid api key"), "auth"),
        (FakeStatusError(429, "Rate limit reached"), "ratelimit"),
        (Exception("Too many requests"), "ratelimit"),
        (Exception("Connection timed out"), "network"),
        (FakeStatusError(504, "gateway timeout"), "network"),
        (ValueError("something else entirely"), None),
    ],
)
def test_classify_error(exc, expected):
    assert E.classify_error(exc) == expected


# ---------------------------------------------------------------------------
# 提示文案
# ---------------------------------------------------------------------------
def test_billing_message_is_actionable():
    msg = E.translate_error(FakeStatusError(402, "insufficient"), phase="建索引")
    assert "余额不足" in msg
    assert "建索引" in msg
    assert "EMBEDDING_BACKEND=local" in msg, "必须给出可操作的自救方案"
    assert "原始错误" in msg, "必须保留原始错误便于排查"


def test_ratelimit_message_mentions_local():
    msg = E.translate_error(FakeStatusError(429, "rate limit"), phase="检索")
    assert "限流" in msg
    assert "EMBEDDING_BACKEND=local" in msg


def test_unknown_error_keeps_raw_and_does_not_pretend():
    msg = E.translate_error(ValueError("boom"), phase="检索")
    assert "未识别" in msg
    assert "boom" in msg


def test_message_is_truncated():
    msg = E.translate_error(Exception("x" * 5000), phase="检索")
    assert len(msg) < 1000, "原始错误应被截断，避免刷屏"


# ---------------------------------------------------------------------------
# 包装器行为
# ---------------------------------------------------------------------------
class FakeInner:
    def __init__(self, exc=None):
        self.exc = exc
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(("docs", list(texts)))
        if self.exc:
            raise self.exc
        return [[0.1, 0.2] for _ in texts]

    def embed_query(self, text):
        self.calls.append(("query", text))
        if self.exc:
            raise self.exc
        return [0.3, 0.4]


def test_guarded_passes_through_on_success():
    inner = FakeInner()
    g = E.GuardedEmbeddings(lambda: inner)
    assert g.embed_documents(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]
    assert g.embed_query("q") == [0.3, 0.4]
    assert inner.calls == [("docs", ["a", "b"]), ("query", "q")]


def test_guarded_is_lazy():
    """构造时不应加载模型 —— 这是启动从 14s 降到 3s 的关键。"""
    built = []

    def factory():
        built.append(1)
        return FakeInner()

    g = E.GuardedEmbeddings(factory)
    assert built == [], "构造时不应调用 factory"
    g.embed_query("q")
    assert len(built) == 1, "首次使用时才加载"
    g.embed_query("q2")
    assert len(built) == 1, "只加载一次"


def test_guarded_translates_on_build_index_phase():
    g = E.GuardedEmbeddings(lambda: FakeInner(FakeStatusError(402, "insufficient")))
    with pytest.raises(E.EmbeddingServiceError) as ei:
        g.embed_documents(["a"])
    assert "建索引" in str(ei.value)
    assert "余额不足" in str(ei.value)


def test_guarded_translates_on_query_phase():
    g = E.GuardedEmbeddings(lambda: FakeInner(FakeStatusError(429, "rate limit")))
    with pytest.raises(E.EmbeddingServiceError) as ei:
        g.embed_query("q")
    assert "检索" in str(ei.value)
    assert "限流" in str(ei.value)


def test_guarded_keeps_exception_chain():
    """必须用 `from e` 保留原始异常，否则排查时看不到根因。"""
    g = E.GuardedEmbeddings(lambda: FakeInner(FakeStatusError(402, "insufficient")))
    with pytest.raises(E.EmbeddingServiceError) as ei:
        g.embed_documents(["a"])
    assert isinstance(ei.value.__cause__, FakeStatusError)


def test_guarded_wraps_factory_failure():
    """加载阶段就失败（例如下载失败）也要被翻译。"""
    def boom():
        raise OSError("load failed")

    g = E.GuardedEmbeddings(boom)
    with pytest.raises(E.EmbeddingServiceError) as ei:
        g.embed_query("q")
    assert "初始化嵌入后端失败" in str(ei.value)
    assert isinstance(ei.value.__cause__, OSError)


# ---------------------------------------------------------------------------
# 后端选择
# ---------------------------------------------------------------------------
def test_unknown_backend_rejected(monkeypatch):
    monkeypatch.setattr(E, "EMBEDDING_BACKEND", "banana")
    with pytest.raises(E.EmbeddingServiceError) as ei:
        E.build_embeddings()
    assert "banana" in str(ei.value)


def test_backend_local_selected(monkeypatch):
    sentinel = FakeInner()
    monkeypatch.setattr(E, "EMBEDDING_BACKEND", "local")
    monkeypatch.setattr(E, "_build_local_embeddings", lambda: sentinel)
    g = E.build_embeddings()
    assert isinstance(g, E.GuardedEmbeddings)
    assert g.inner is sentinel


def test_backend_api_selected(monkeypatch):
    sentinel = FakeInner()
    monkeypatch.setattr(E, "EMBEDDING_BACKEND", "API")   # 大小写不敏感
    monkeypatch.setattr(E, "_build_api_embeddings", lambda: sentinel)
    g = E.build_embeddings()
    assert g.inner is sentinel


def test_api_backend_requires_key(monkeypatch):
    monkeypatch.setattr(E, "EMBEDDING_BACKEND", "api")
    monkeypatch.setattr(E, "SILICONFLOW_API_KEY", "")
    g = E.build_embeddings()          # 懒加载：构造时不校验
    with pytest.raises(E.EmbeddingServiceError) as ei:
        g.embed_query("q")
    assert "SILICONFLOW_API_KEY" in str(ei.value)


# ---------------------------------------------------------------------------
# 本地模型加载失败的翻译
# ---------------------------------------------------------------------------
def test_local_incomplete_cache_message():
    """下载中断留下的残缺缓存会报一个和缓存无关的 Pooling 错误。"""
    exc = TypeError("Pooling.__init__() missing 1 required positional argument: 'embedding_dimension'")
    msg = E.local_error_message(exc)
    assert "缓存可能不完整" in msg
    assert "重新" in msg
    assert "Pooling" in msg  # 保留原始错误便于对照


def test_local_network_timeout_message():
    msg = E.local_error_message(
        OSError("[WinError 10060] 由于连接方在一段时间后没有正确答复")
    )
    assert "网络问题" in msg
    assert "HF_ENDPOINT" in msg


def test_local_offline_missing_cache_message():
    """离线模式下模型没缓存时，要告诉用户怎么恢复联网。"""
    msg = E.local_error_message(
        Exception("Cannot find the requested files in the local cache. "
                  "Set HF_HUB_OFFLINE=0 to retry.")
    )
    assert "离线模式" in msg
    assert "HF_HUB_OFFLINE" in msg


def test_local_unknown_error_keeps_raw():
    msg = E.local_error_message(RuntimeError("weird failure"))
    assert "weird failure" in msg


def test_local_build_wraps_exception(monkeypatch):
    """构造失败必须包成 EmbeddingServiceError，并保留异常链。

    注意：现在是延迟加载，所以异常在首次使用（embed_query）时才抛出，
    而不是 build_embeddings() 时。
    """
    import langchain_huggingface

    def boom(**kwargs):
        raise OSError("[WinError 10060] connection failed")

    monkeypatch.setattr(E, "EMBEDDING_BACKEND", "local")
    monkeypatch.setattr(langchain_huggingface, "HuggingFaceEmbeddings", boom)
    g = E.build_embeddings()          # 不抛（懒加载）
    with pytest.raises(E.EmbeddingServiceError) as ei:
        g.embed_query("q")
    assert isinstance(ei.value.__cause__, OSError)
    assert "网络问题" in str(ei.value)
