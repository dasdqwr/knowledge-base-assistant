import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, HTTPException

from src.document import list_documents, delete_document, save_uploaded_file
from src.cache import close_redis, clear_pattern
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

# 串行化索引重建，避免并发 rmtree 同一个 chroma_db（见 rebuild_agent）
index_lock = asyncio.Lock()

# 最近一次索引重建的结果，供 /rebuild/status 查询
rebuild_state = {"state": "idle", "error": None, "at": None}

# --- 生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化
    #声明：我要改的是全局变量
    global agent, pool
    print("🚀 正在启动服务，初始化 Agent...")
    agent, pool = await build_agent_with_memory()
    yield
    # 关闭时清理
    print("🛑 正在关闭服务，释放连接池，关闭 Redis 连接...")
    await pool.close()
    await close_redis()

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
    # 文件名清洗与后缀校验都在 src.document 里做（含路径穿越防护）
    try:
        saved = save_uploaded_file(file.filename, file.file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 立即返回已保存的真实文件名，前端据此刷新列表
    background_tasks.add_task(rebuild_agent)

    return {"status": "ok", "filename": saved.name, "message": "文件已保存，正在后台重建索引"}



async def rebuild_agent():
    """后台重建 Agent（复用全局 pool）

    index_lock 的作用有两个：
    1. /upload 和 /delete 都走 background_tasks，用户快速连点会并发进入。
       两个 rebuild 同时 rmtree/写入同一个 chroma_db 会互相破坏。
       加锁后串行化。
    2. 顺带得到幂等性：排队中的第二个任务拿到锁时，索引已经被前一个建好了，
       哈希比对会跳过内容未变的重复重建。

    刻意**不传 force**：build_retriever 的新鲜度判断已经覆盖了
    "文件新增/删除/内容变化" 三种情况（用内容哈希，不依赖 mtime）。
    传 force=True 只会在"重复上传同一个文件"时白跑一遍 277 次嵌入。

    失败会写进 rebuild_state，前端可以经 /rebuild/status 查到。
    这对"删库成功但重建失败"尤其重要 —— 那种情况下服务活着但查不了，
    以前只往控制台打一行日志，前端毫无感知。
    """
    global agent, rebuild_state

    async with index_lock:
        print("🔄 后台重建索引...")
        try:
            retriever = build_retriever()
            new_agent = await build_agent(retriever, pool)
            # 成功之后才替换全局 agent，避免中途失败把可用的旧 agent 换掉
            agent = new_agent
            # 清空旧的检索缓存
            count = await clear_pattern("retrieval:*")
            rebuild_state = {
                "state": "ok",
                "error": None,
                "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            print(f"✅ 重建完成，清空 {count} 条缓存")
        except Exception as e:
            rebuild_state = {
                "state": "failed",
                "error": f"{type(e).__name__}: {e}",
                "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            print(f"❌ 重建失败: {e}")


@app.get("/rebuild/status")
async def rebuild_status():
    """查询最近一次索引重建的结果"""
    return rebuild_state

@app.get("/documents")
async def get_documents():
    """列出知识库中的所有文档"""
    return {"documents": list_documents()}


@app.post("/delete")
async def delete_file(
    filename: str,
    background_tasks: BackgroundTasks,
):
    """删除文档并后台重建索引"""
    if not delete_document(filename):
        raise HTTPException(status_code=404, detail="文件不存在或路径不合法")

    # 后台重建索引（和上传一样）
    background_tasks.add_task(rebuild_agent)

    return {"status": "ok", "message": f"已删除 {filename}，正在后台重建索引"}