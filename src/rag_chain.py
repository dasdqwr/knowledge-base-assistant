from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.model import get_model

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个知识库助手。请基于以下上下文回答问题。
如果上下文中没有相关信息，诚实地说"我不知道"。

上下文：
{context}"""),
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(retriever):
    model = get_model()
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | model
        | StrOutputParser()
    )