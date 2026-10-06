import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import asyncio
import sys
import uuid

# Windows 控制台默认 GBK，print emoji 会抛 UnicodeEncodeError 打断对话流程。
from src.utf8 import enable_utf8_console

enable_utf8_console()

from src.agent import build_agent_with_memory


def run_async(coro):
    """
    在 Windows 上安全地跑异步函数。

    用 loop_factory 指定 SelectorEventLoop，
    解决 psycopg 异步模式与 Windows 默认 ProactorEventLoop 不兼容的问题。
    """
    if sys.platform == "win32":
        return asyncio.run(coro, loop_factory=asyncio.SelectorEventLoop)
    return asyncio.run(coro)


def print_answer(messages):
    """从消息列表里找最后一条 AI 回复并打印"""
    for msg in reversed(messages):
        if msg.type == "ai" and msg.content:
            print(f"助手: {msg.content}\n")
            return
    print("助手: （没有找到回答）\n")


async def _chat():
    # build_agent_with_memory 是异步的，且内部已经建好连接池
    agent, pool = await build_agent_with_memory()

    thread_id = input("会话 ID（回车自动生成）: ").strip() or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    print(f"会话 ID: {thread_id}")
    print("（下次输入相同 ID 可继续上次对话）\n")

    print("\n知识库助手已启动，输入 quit 退出\n")
    try:
        while True:
            try:
                question = input("你: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n再见！")
                break

            if question.lower() == "quit":
                print("再见！")
                break
            if not question:
                continue

            # 只传新消息，历史由 checkpointer 自动加载
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]},
                config=config,
            )
            print_answer(result["messages"])
    finally:
        # 异步连接池要用 aclose()
        await pool.close()


def main():
    run_async(_chat())


if __name__ == "__main__":
    main()
