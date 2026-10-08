"""文档管理：列出、删除、保存上传"""
from pathlib import Path

from src.config import DATA_DIR, SUPPORTED_SUFFIXES


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


def safe_target_path(filename: str) -> Path:
    """把客户端传来的文件名解析成 data/ 目录下的安全路径。

    两道检查，缺一不可：

    1. 必须是纯文件名，不能带目录成分。
       `../../x.md`、`..\\x.md`、`a/b.md` 一律拒绝。
       直接 `Path(DATA_DIR) / filename` 会顺从穿越，写到 data/ 外面去。

    2. 解析后的绝对路径必须真的在 data/ 里面。
       用 Path.relative_to 判断，而不是 str.startswith —— 后者会把
       `.../data_evil/x.md` 误判成合法（前缀碰巧相同）。

    返回值保证落在 data/ 内；非法输入抛 ValueError。
    """
    name = (filename or "").strip()
    if not name or name != Path(name).name:
        raise ValueError("文件名不合法：不允许包含路径分隔符")

    data_dir = Path(DATA_DIR).resolve()
    target = (data_dir / name).resolve()

    # relative_to 在不相容时抛 ValueError，正好用来判定"是否在 data/ 内"
    target.relative_to(data_dir)

    return target


def save_uploaded_file(filename: str, source) -> Path:
    """把上传的文件流保存到 data/ 下。

    source 需要是带 read() 的类文件对象（FastAPI 的 UploadFile.file 就是）。
    """
    target = safe_target_path(filename)

    if target.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的文件类型: {target.suffix}")

    with target.open("wb") as f:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return target


def delete_document(filename: str) -> bool:
    """
    删除 data/ 下的指定文档。

    安全考虑：
        - 只允许删除 data/ 下的文件
        - 禁止路径穿越（如 ../../etc/passwd）
    """
    try:
        target = safe_target_path(filename)
    except ValueError:
        return False

    if not target.exists() or not target.is_file():
        return False

    target.unlink()
    return True