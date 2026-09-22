from src.loader import load_documents, split_documents
from src.vectorstore import build_vectorstore, load_vectorstore
from src.rag_chain import build_rag_chain
from src.config import TOP_K, CHROMA_DIR
from pathlib import Path

def index_documents():
    """构建索引"""
    print("加载文档...")
    docs = load_documents()
    if not docs:
        print("data/ 目录下没有文档")
        return
    print(f"加载了 {len(docs)} 个文档")
    chunks = split_documents(docs)
    print(f"切分为 {len(chunks)} 个块")
    build_vectorstore(chunks)
    print("索引构建完成")

def main():
    # 首次运行构建索引
    if not Path(CHROMA_DIR).exists() or not any(Path(CHROMA_DIR).iterdir()):
        index_documents()

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
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
