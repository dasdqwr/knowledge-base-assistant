from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

from src.config import TOP_K
from src.model import get_model
from langchain_community.retrievers import BM25Retriever
import re
import jieba

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个知识库助手。请基于以下上下文回答问题。
如果上下文中没有相关信息，诚实地说"我不知道"。

上下文：
{context}"""),
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")


# 程序启动
#     ↓
# build_rag_chain(retriever)   ← retriever 传入，被闭包捕获
#     ↓ 返回 chain 对象
#     ↓
# （等待用户输入）
#     ↓
# chain.invoke("init_chat_model 怎么用？")   ← question 传入
#     ↓
# rerank_and_format("init_chat_model 怎么用？")   ← 两个参数汇合
#     ↓
# retriever.invoke("init_chat_model 怎么用？")   ← 检索
#     ↓
# 返回结果
def build_rag_chain(retriever):
    model = get_model()

    def rerank_and_format(question):
        # 1. 检索
        docs = retriever.invoke(question)
        # 2. 重排
        pairs = [(question, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in ranked[:TOP_K]]
        # 3. 格式化
        return format_docs(top_docs)

    return (
        {
            "context": rerank_and_format,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT # ← 接收字典，填充两个变量
        | model
        | StrOutputParser()
    )

def preprocess(text: str) -> list[str]:
    """英文标识符按整体提取，中文用 jieba 分词"""
    # 先提取英文标识符和数字
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|\d+', text)
    # 再对剩余中文部分用 jieba 分词
    chinese_text = re.sub(r'[a-zA-Z0-9_\s]+', ' ', text)
    tokens.extend(jieba.lcut(chinese_text))
    return tokens

def build_hybrid_retriever(vectorstore, all_chunks):
    # 向量检索：负责语义理解
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # BM25：负责精确关键词匹配
    bm25_retriever = BM25Retriever.from_documents(all_chunks, preprocess_func=preprocess)
    bm25_retriever.k = 5

    # 组合
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.8, 0.2],
    )