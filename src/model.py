import os

# 必须在 import transformers/sentence_transformers 之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from langchain.chat_models import init_chat_model
from src.config import MODEL_NAME, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, SILICONFLOW_BASE_URL, SILICONFLOW_API_KEY


def get_model():
    return init_chat_model(
        model=MODEL_NAME,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )

def get_embeddings():
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model="BAAI/bge-m3",
        api_key=SILICONFLOW_API_KEY,
        base_url=SILICONFLOW_BASE_URL,
    )