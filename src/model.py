from functools import lru_cache
from langchain.chat_models import init_chat_model
from src.config import MODEL_NAME, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, SILICONFLOW_BASE_URL, SILICONFLOW_API_KEY


def get_model():
    return init_chat_model(
        model=MODEL_NAME,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )

@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model="BAAI/bge-m3",
        api_key=SILICONFLOW_API_KEY,
        base_url=SILICONFLOW_BASE_URL,
    )

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