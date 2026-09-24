import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from src.agent import build_agent

def main():
    agent = build_agent()
    messages = []

    print("\n知识库助手已启动，输入 quit 退出\n")
    while True:
        question = input("你: ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue
        messages.append({"role": "user", "content": question})
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        for msg in reversed(messages):
            if msg.type == "ai" and msg.content:
                print(f"助手: {msg.content}\n")
                break

if __name__ == "__main__":
    main()
