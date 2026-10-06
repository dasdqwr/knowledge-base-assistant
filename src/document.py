"""文档管理：列出、删除"""
from pathlib import Path

from src.config import DATA_DIR

# 支持的文档类型
SUPPORTED_SUFFIXES = (".pdf", ".txt", ".md", ".markdown", ".docx")


def list_documents() -> list[dict]:
    """列出 data/ 下的所有文档"""
    data_dir = Path(DATA_DIR)
    if not data_dir.exists():
        return []

    docs = []
    for f in sorted(data_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES:
            docs.append({
                "name": f.name,
                "size_kb": round(f.stat().st_size / 1024, 1),
            })
    return docs


def delete_document(filename: str) -> bool:
    """
    删除 data/ 下的指定文档。

    安全考虑：
        - 只允许删除 data/ 下的文件
        - 禁止路径穿越（如 ../../etc/passwd）
    """
    data_dir = Path(DATA_DIR).resolve()
    target = (data_dir / filename).resolve()

    # 安全检查：目标必须在 data/ 目录内
    if not str(target).startswith(str(data_dir)):
        return False

    if not target.exists() or not target.is_file():
        return False

    target.unlink()
    return True