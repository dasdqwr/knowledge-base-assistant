import os
from contextlib import asynccontextmanager

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from fastapi import FastAPI
from pydantic import BaseModel
from src.agent import build_agent



# --- 全局变量 ---
agent = None
pool = None

# --- 生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化
    #声明：我要改的是全局变量
    global agent, pool
    print("🚀 正在启动服务，初始化 Agent...")
    agent, pool = await build_agent()
    yield
    # 关闭时清理
    print("🛑 正在关闭服务，释放连接池...")
    await pool.close()

app = FastAPI(lifespan=lifespan)

# ========== 请求/响应模型 ==========
class ChatRequest(BaseModel):
    message: str
    thread_id: str


class ChatResponse(BaseModel):
    answer: str
    thread_id: str

# ========== 接口 ==========
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """聊天接口"""
    config = {"configurable": {"thread_id": req.thread_id}}

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": req.message}]},
        config=config,
    )

    # 取最后一条 AI 消息
    for msg in reversed(result["messages"]):
        if msg.type == "ai" and msg.content:
            return ChatResponse(answer=msg.content, thread_id=req.thread_id)

    return ChatResponse(answer="抱歉，我无法回答。", thread_id=req.thread_id)


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}

