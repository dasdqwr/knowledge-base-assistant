from functools import lru_cache
from langchain.chat_models import init_chat_model
from src.config import (
    MODEL_NAME,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
)


def get_model():
    return init_chat_model(
        model=MODEL_NAME,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )

@lru_cache(maxsize=1)
def get_embeddings():
    """嵌入对象（按 EMBEDDING_BACKEND 选择 api / local，并套上错误翻译）。

    实现在 src/embeddings.py。加 @lru_cache 是因为本地模型加载很贵
    （bge-m3 约 2 GB），必须整个进程只加载一次。
    """
    from src.embeddings import build_embeddings

    return build_embeddings()

class LazyReranker:
    """延迟加载的 Reranker，第一次调用 predict 时才加载模型"""

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            print("🔄 首次加载 Rerank 模型（约 10 秒）...")
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder("BAAI/bge-reranker-v2-m3")
            print("✅ Rerank 模型加载完成")
        return self._model

    def predict(self, pairs):
        """对外接口和 CrossEncoder 一致"""
        return self._load().predict(pairs)


@lru_cache(maxsize=1)
def get_reranker():
    """返回一个延迟加载的 Reranker（不触发实际加载）"""
    return LazyReranker()