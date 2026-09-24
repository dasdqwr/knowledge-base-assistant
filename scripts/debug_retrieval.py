from src.loader import load_documents, split_documents
from src.vectorstore import load_vectorstore
from src.rag_chain import build_hybrid_retriever
from sentence_transformers import CrossEncoder

# 初始化
docs = load_documents()
chunks = split_documents(docs)
vectorstore = load_vectorstore()
retriever = build_hybrid_retriever(vectorstore, chunks)
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

question = "LangChain 的 init_chat_model 怎么用？"
KEYWORD = "init_chat_model"

# ========== 1. 混合检索 ==========
results = retriever.invoke(question)

print(f"【混合检索】检索到 {len(results)} 条：\n")
for i, doc in enumerate(results, 1):
    has = KEYWORD in doc.page_content
    mark = "✅" if has else "  "
    print(f"[{i}] {mark} | {doc.page_content[:60]}")

# ========== 2. 加 Rerank ==========
pairs = [[question, doc.page_content] for doc in results]
scores = reranker.predict(pairs)
ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)

print(f"\n【混合 + Rerank】取前 5 条：\n")
for i, (doc, score) in enumerate(ranked[:5], 1):
    has = KEYWORD in doc.page_content
    mark = "✅" if has else "  "
    print(f"[{i}] {mark} 分数: {score:6.3f} | {doc.page_content[:60]}")