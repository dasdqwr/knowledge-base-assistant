import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks
from src.config import DATA_DIR
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from src.agent import build_agent, build_retriever, build_agent_with_memory
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
    # 首次初始化
    agent, pool = await build_agent_with_memory()
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

@app.post("/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """上传文档并重建索引"""

    # 1. 保存文件
    file_path = Path(DATA_DIR) / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)


    # 2. 后台异步重建
    background_tasks.add_task(rebuild_agent)

    return {"status": "ok", "message": "文件已保存，正在后台重建索引"}



async def rebuild_agent():
    """后台重建，不阻塞请求"""
    global agent
    print("🔄 后台重建索引...")
    retriever = build_retriever()
    agent = await build_agent(retriever, pool)
    print("✅ 重建完成")