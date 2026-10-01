# run.py
import asyncio
import uvicorn
from app import app

if __name__ == "__main__":
    config = uvicorn.Config(app, host="127.0.0.1", port=8000)
    server = uvicorn.Server(config)
    # 使用 loop_factory 指定 SelectorEventLoop
    asyncio.run(server.serve(), loop_factory=asyncio.SelectorEventLoop)