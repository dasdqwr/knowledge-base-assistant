import os
from contextlib import asynccontextmanager

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from fastapi import FastAPI
from pydantic import BaseModel
from src.agent import build_agent
import json
from fastapi.responses import StreamingResponse


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

@app.post("/chat/stream", response_model=ChatResponse)
async def chat_stream(req: ChatRequest):
    """流式聊天接口"""
    config = {"configurable": {"thread_id": req.thread_id}}

    async def event_generator():
        """异步生成器，逐块产出SSE格式的数据"""
        try:
            async for chunk in agent.astream(
                {"messages": [{"role": "user", "content": req.message}]},
                config=config,
                stream_mode="messages",
            ):
                # chunk 是 (AIMessageChunk, metadata) 元组
                token, metadata = chunk
                if token.content:
                    # 包成SSE格式
                    data = json.dumps({
                        "type": "token",
                        "content": token.content,
                    }, ensure_ascii=False)
                    #每次 yield，就往外推一次数据，然后暂停，等下次迭代。
                    #FastAPI 的 StreamingResponse 会消费这个生成器，把每块数据推给前端
                    yield f"data: {data}\n\n"

            # 发送结束信号
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            error_data = json.dumps({"type": "error", "content": str(e)})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream", #SSE的标准Content-Type
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用Nginx缓冲
        },
    )

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}

