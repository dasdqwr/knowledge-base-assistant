from src.loader import load_documents, split_documents
from src.vectorstore import build_vectorstore, load_vectorstore
from src.rag_chain import build_rag_chain, build_hybrid_retriever
from src.config import TOP_K, CHROMA_DIR
from pathlib import Path

def index_documents(chunks=None):
    """构建向量库索引"""
    if chunks is None:
        docs = load_documents()
        chunks = split_documents(docs)
    print("构建向量库...")
    return build_vectorstore(chunks)

def main():
    # ① 加载 + 切分（BM25 和建库都需要）
    docs = load_documents()
    chunks = split_documents(docs)

    # ② 向量库：不存在则建，存在则加载
    if not Path(CHROMA_DIR).exists() or not any(Path(CHROMA_DIR).iterdir()):
        vectorstore = index_documents(chunks)
    else:
        print("加载已有向量库...")
        vectorstore = load_vectorstore()

    # ③ 构建混合检索器（向量 + BM25）
    retriever = build_hybrid_retriever(vectorstore, chunks)

    # ④ 构建 RAG 链
    chain = build_rag_chain(retriever)

    print("\n知识库助手已启动，输入 quit 退出\n")
    while True:
        question = input("你: ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue
        answer = chain.invoke(question)
        print(f"助手: {answer}\n")

if __name__ == "__main__":
    main()
