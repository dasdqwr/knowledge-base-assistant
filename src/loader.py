from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_documents():
    """加载 data/ 目录下所有文档"""
    docs = []
    for file in Path(DATA_DIR).iterdir():
        if file.is_dir():
            continue
        try:
            if file.suffix == '.pdf':
                # 转成字符串传给加载器
                docs.extend(PyPDFLoader(str(file)).load())
                print(f"✅ 加载 PDF: {file.name}")
            elif file.suffix in ('.txt', '.md'):
                docs.extend(TextLoader(str(file), encoding="utf-8").load())
                print(f"✅ 加载文本: {file.name}")
        except Exception as e:
            print(f"❌ 加载失败 {file.name}: {e}")
    return docs

def split_documents(docs):
    """切分文档"""
    if not docs:
        return []
    #用 len 这个函数来计算长度,函数本身可以像变量一样传递
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"📄 切分为 {len(chunks)} 个块")
    return chunks

