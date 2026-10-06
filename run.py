# run.py
import asyncio

# Windows 控制台默认 GBK，print emoji 会抛 UnicodeEncodeError，直接中断启动流程。
# 必须在 import app（它的导入链里就有 emoji 输出）之前把标准流切到 UTF-8。
from src.utf8 import enable_utf8_console

enable_utf8_console()

import uvicorn
from app import app

if __name__ == "__main__":
    config = uvicorn.Config(app, host="127.0.0.1", port=8000)
    server = uvicorn.Server(config)
    # 使用 loop_factory 指定 SelectorEventLoop
    asyncio.run(server.serve(), loop_factory=asyncio.SelectorEventLoop)
