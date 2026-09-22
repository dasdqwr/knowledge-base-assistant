from pathlib import Path

from langchain_chroma import Chroma

from src.config import CHROMA_DIR
from src.model import get_embeddings


def build_vectorstore(documents):
    """构建并持久化向量库"""
    #向量存进 Chroma
    return Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        persist_directory="./chroma_db",  # 存到本地目录
    )

def load_vectorstore():
    """加载已有向量库"""
    if not Path(CHROMA_DIR).exists():
        raise FileNotFoundError(
            f"向量库目录不存在: {CHROMA_DIR}，请先运行 build_vectorstore()"
        )
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
    )