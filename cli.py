"""
知识库助手 CLI

用法：
    kb ask "init_chat_model 怎么用？"     # 单次提问
    kb chat                              # 交互式对话
    kb upload data/doc.pdf               # 上传文档
    kb delete doc.pdf                    # 删除文档
    kb rebuild                           # 重建索引
    kb serve                             # 启动后端
    kb web                               # 启动前端
    kb evaluate                          # 跑评估
    kb info                              # 查看配置

安装：
    pip install -e .
"""
import asyncio
import os
import shutil
import sys
import uuid
from pathlib import Path

# ========== 环境变量（必须在 import 其他库之前） ==========
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Windows 控制台默认 GBK，print emoji 会抛 UnicodeEncodeError 打断命令执行。
from src.utf8 import enable_utf8_console

enable_utf8_console()

import typer

from src.config import DATA_DIR, PROJECT_ROOT

app = typer.Typer(
    name="kb",
    help="📚 个人知识库助手命令行工具",
    add_completion=False,
    no_args_is_help=True,   # 不带参数时显示帮助
)


# ========== 工具函数 ==========

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
            typer.echo(f"\n🤖 {msg.content}\n")
            return
    typer.echo("\n🤖 （没有找到回答）\n")


# ========== 1. ask：单次提问 ==========

@app.command()
def ask(
    question: str = typer.Argument(..., help="要问的问题"),
    thread_id: str = typer.Option("cli_default", "--thread-id", "-t", help="会话 ID"),
):
    """向知识库提问一次"""
    from src.agent import build_agent_with_memory

    typer.echo(f"🙋 {question}")

    async def _run():
        agent, pool = await build_agent_with_memory()
        try:
            config = {"configurable": {"thread_id": thread_id}}
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]},
                config=config,
            )
            print_answer(result["messages"])
        finally:
            await pool.close()

    run_async(_run())


# ========== 2. chat：交互式对话 ==========

@app.command()
def chat(
    thread_id: str = typer.Option(None, "--thread-id", "-t", help="会话 ID（不填则自动生成）"),
):
    """进入交互式对话模式"""
    from src.agent import build_agent_with_memory

    if not thread_id:
        thread_id = str(uuid.uuid4())

    typer.echo(f"会话 ID: {thread_id}")
    typer.echo("输入 quit 退出\n")

    async def _run():
        agent, pool = await build_agent_with_memory()
        config = {"configurable": {"thread_id": thread_id}}

        try:
            while True:
                try:
                    question = typer.prompt("你", prompt_suffix=": ").strip()
                except (KeyboardInterrupt, EOFError):
                    typer.echo("\n再见！")
                    break

                if question.lower() == "quit":
                    typer.echo("再见！")
                    break
                if not question:
                    continue

                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": question}]},
                    config=config,
                )
                print_answer(result["messages"])
        finally:
            await pool.close()

    run_async(_run())


# ========== 3. upload：上传文档并重建索引 ==========

@app.command()
def upload(
    file: str = typer.Argument(..., help="要上传的文件路径"),
):
    """上传文档并重建索引"""
    src = Path(file)
    if not src.exists():
        typer.echo(f"❌ 文件不存在: {file}", err=True)
        raise typer.Exit(1)

    if src.suffix.lower() not in (".pdf", ".txt", ".md", ".docx"):
        typer.echo(f"❌ 不支持的类型: {src.suffix}", err=True)
        raise typer.Exit(1)

    dst = Path(DATA_DIR) / src.name
    shutil.copy(src, dst)
    typer.echo(f"✅ 已复制到: {dst}")

    rebuild()


# ========== 4. delete：删除文档并重建索引 ==========

@app.command()
def delete(
    filename: str = typer.Argument(..., help="要删除的文件名（不含路径）"),
    yes: bool = typer.Option(False, "--yes", "-y", help="跳过确认"),
):
    """删除 data/ 下的文档并重建索引"""
    path = Path(DATA_DIR) / filename

    if not path.exists():
        typer.echo(f"❌ 文件不存在: {filename}", err=True)
        raise typer.Exit(1)

    if not yes:
        confirm = typer.confirm(f"确认删除 {filename}？")
        if not confirm:
            typer.echo("已取消")
            raise typer.Exit(0)

    os.remove(path)
    typer.echo(f"✅ 已删除: {filename}")

    rebuild()


# ========== 5. rebuild：重建索引 ==========

@app.command()
def rebuild():
    """重建向量库和 Agent"""
    from src.agent import build_agent_with_memory
    from src.cache import clear_pattern

    async def _run():
        typer.echo("🔄 重建中...")
        agent, pool = await build_agent_with_memory()
        try:
            count = await clear_pattern("retrieval:*")
            typer.echo(f"✅ 重建完成，清空 {count} 条缓存")
        finally:
            await pool.close()

    run_async(_run())


# ========== 6. serve：启动后端 ==========

@app.command()
def serve():
    """启动 FastAPI 后端（调用 run.py）"""
    import subprocess

    typer.echo("🚀 启动后端: http://127.0.0.1:8000")
    subprocess.run([sys.executable, "run.py"])


# ========== 7. web：启动前端 ==========

@app.command()
def web():
    """启动 Streamlit 前端"""
    import subprocess

    typer.echo("🌐 启动前端: http://127.0.0.1:8501")
    subprocess.run(["streamlit", "run", "streamlit_app.py"])


# ========== 8. evaluate：跑检索评估 ==========

@app.command()
def evaluate():
    """运行检索层评估"""
    import subprocess

    typer.echo("📊 运行评估...")
    subprocess.run([sys.executable, "scripts/evaluate.py"])


# ========== 9. info：查看项目信息 ==========

@app.command()
def info():
    """显示项目配置信息"""
    from src.config import (
        CHROMA_DIR, DB_URL, REDIS_HOST, REDIS_PORT, TOP_K,
    )

    typer.echo("📋 项目配置")
    typer.echo("=" * 60)
    typer.echo(f"项目根目录:   {PROJECT_ROOT}")
    typer.echo(f"文档目录:     {DATA_DIR}")
    typer.echo(f"向量库目录:   {CHROMA_DIR}")

    # 脱敏显示数据库地址
    if DB_URL:
        masked = DB_URL.split("@")[-1] if "@" in DB_URL else DB_URL
        typer.echo(f"PostgreSQL:   ...@{masked}")
    else:
        typer.echo("PostgreSQL:   ❌ 未配置")

    typer.echo(f"Redis:        {REDIS_HOST}:{REDIS_PORT}")
    typer.echo(f"TOP_K:        {TOP_K}")
    typer.echo("=" * 60)

    # 列出文档
    data_path = Path(DATA_DIR)
    if data_path.exists():
        docs = [
            f for f in data_path.iterdir()
            if f.is_file() and f.suffix.lower() in (".pdf", ".txt", ".md", ".docx")
        ]
        typer.echo(f"\n📄 文档数量: {len(docs)}")
        for d in docs:
            size_kb = d.stat().st_size / 1024
            typer.echo(f"  - {d.name}  ({size_kb:.1f} KB)")
    else:
        typer.echo(f"\n⚠️ 文档目录不存在: {data_path}")


# ========== 入口 ==========

if __name__ == "__main__":
    app()