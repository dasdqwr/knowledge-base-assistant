import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import uuid
from src.agent import build_agent

def main():
    agent, pool = build_agent()
    thread_id = input("会话 ID（回车自动生成）: ").strip() or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    print(f"会话 ID: {thread_id}")
    print("（下次输入相同 ID 可继续上次对话）\n")

    print("\n知识库助手已启动，输入 quit 退出\n")
    try:
        while True:
            question = input("你: ").strip()
            if question.lower() == "quit":
                break
            if not question:
                continue
            # 只传新消息，历史由 checkpointer 自动加载
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config=config,
            )
            for msg in reversed(result["messages"]):
                if msg.type == "ai" and msg.content:
                    print(f"助手: {msg.content}\n")
                    break
    finally:
        pool.close()


if __name__ == "__main__":
    main()
