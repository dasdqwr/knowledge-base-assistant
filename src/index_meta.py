"""
索引新鲜度判断（向量库能不能复用，而不是每次重建）。

要解决的是什么
--------------
原来的 build_retriever 无条件 rmtree + 全量重建，每次启动都重付一遍嵌入成本。
实测本项目 277 个块约 3.8 秒、277 次嵌入 API 调用。文档涨上去后这个成本线性增长。

核心设计：清单的写入时机
------------------------
index_manifest.json 的存在 = "索引完整，且与当前文件一致"。

    需要重建 → 先删清单 → 再 rmtree 旧库 → 嵌入构建 → 全部成功后写清单

为什么是"写前删除"而不是记一个 build_ok=false 标志：
    进程被 Ctrl+C 或被杀时，你根本没有机会去写那个 flag。
    "先删清单"不需要你还有机会 —— 半成品必然处于"没有清单"的状态，
    下次启动判定为不新鲜，自然重建。这是失败安全的。

签名记什么
----------
1. 文件名 + 大小 + mtime + 内容 SHA256
   mtime 会被"复制时保留时间戳"骗过（git checkout / robocopy / 解压），
   所以最终判据是内容哈希。本项目 data/ 只有 152 KB，哈希成本可忽略。
2. 切分参数（CHUNK_SIZE / CHUNK_OVERLAP）+ SCHEMA_VERSION
   改参数会得到不同的块，必须重建。
3. 嵌入模型名
   换模型所有向量作废。
"""

import hashlib
import json
import subprocess
import time
from pathlib import Path

# 清单格式版本。以后改了签名结构或切分逻辑，就 +1，
# 让所有旧清单自动失效并触发一次重建。
SCHEMA_VERSION = 2


# ========== 小工具 ==========

def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """流式计算文件内容哈希。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def _code_version() -> str:
    """取切分/索引相关代码的 git 版本号。

    用 `git hash-object` 算 blob 哈希 —— 它同时反映了已提交内容和工作区改动，
    所以未提交的改动也能被识别出来。
    拿不到 git（没装、不在仓库里、不在仓库根目录）时返回 "unknown"。
    """
    try:
        out = subprocess.run(
            ["git", "hash-object", "src/loader.py", "src/vectorstore.py"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().replace("\n", "-")
    except Exception:
        pass
    return "unknown"


def list_data_files() -> list[Path]:
    """列出 data/ 下参与索引的文件（排除 .gitkeep 等点文件）。"""
    from src.config import DATA_DIR

    data_dir = Path(DATA_DIR)
    if not data_dir.is_dir():
        return []
    return sorted(
        (p for p in data_dir.iterdir() if p.is_file() and not p.name.startswith(".")),
        key=lambda p: p.name,
    )


def _file_stats(path: Path) -> dict:
    """单个文件的签名项：名称、大小、mtime、内容哈希。"""
    st = path.stat()
    return {
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "sha256": _sha256(path),
    }


def file_stats(files=None) -> dict:
    """整个 data/ 目录的签名项，以文件名为 key。

    只取文件名而不是绝对路径：路径换机器就变，会让清单永远判定"变了"。
    """
    files = list_data_files() if files is None else files
    return {p.name: _file_stats(p) for p in files}


def current_params() -> dict:
    """所有决定索引内容的参数。任何一项变化都必须重建。"""
    from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL

    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "code_version": _code_version(),
        "schema_version": SCHEMA_VERSION,
    }


# ========== 清单读写 ==========

def load_manifest(path=None) -> dict | None:
    """读清单。文件不存在、是坏的、版本对不上，一律返回 None。"""
    from src.config import MANIFEST_PATH

    p = Path(MANIFEST_PATH if path is None else path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        return None
    return data


def write_manifest(params: dict, files: dict, chunk_count: int, path=None) -> Path:
    """写清单。**只在向量库构建成功之后调用。**

    这个文件的存在本身就是"索引完整且新鲜"的凭证。
    """
    from src.config import MANIFEST_PATH

    p = Path(MANIFEST_PATH if path is None else path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "params": params,
        "files": files,
        "chunk_count": chunk_count,
        "chroma_dir": str(Path(__file__).resolve().parent.parent / "chroma_db"),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(p)
    return p


def clear_manifest(path=None) -> bool:
    """删除清单 —— 标记"索引不可信"。返回是否真的删掉了东西。

    必须在 rmtree(chroma_db) **之前**调用。
    """
    from src.config import MANIFEST_PATH

    p = Path(MANIFEST_PATH if path is None else path)
    if p.exists():
        p.unlink()
        return True
    return False


# ========== 核心判定 ==========

def is_index_fresh(snapshot=None, params=None, path=None) -> tuple[bool, str, bool]:
    """判断现有向量库能否直接复用。

    返回 (是否新鲜, 原因, 是否需要刷新清单元数据)。

    第三个返回值处理这种情况：文件内容没变，但 mtime/size 变了
    （碰了文件、重新拷了一份、git checkout 回来）。此时索引完全有效，
    不必重建，但清单里的元数据该更新，否则每次启动都会重复报同一条提示。

    调用方顺序应当是：
        ok, reason, refresh = is_index_fresh()
        if ok:
            reuse()
            if refresh:
                write_manifest(...)     # 只刷新元数据，不重建索引
        else:
            clear_manifest()            # ← 先删凭证
            rmtree(chroma_db)           # ← 再删库
            build()
            write_manifest(...)         # ← 成功后才重新发凭证
    """
    manifest = load_manifest(path)
    if manifest is None:
        return False, "清单不存在或已失效（首次构建，或上次构建中断）", False

    params = current_params() if params is None else params
    if manifest.get("params") != params:
        # 找出具体是哪一项变了，方便排查
        old = manifest.get("params") or {}
        changed = [
            k for k in set(old) | set(params)
            if old.get(k) != params.get(k)
        ]
        return False, "索引参数已变化: " + ", ".join(sorted(changed)), False

    saved = manifest.get("files")
    if not isinstance(saved, dict):
        return False, "清单缺少文件信息", False

    snapshot = file_stats() if snapshot is None else snapshot

    if set(saved) != set(snapshot):
        added = sorted(set(snapshot) - set(saved))
        removed = sorted(set(saved) - set(snapshot))
        parts = []
        if added:
            parts.append("新增 " + ", ".join(added))
        if removed:
            parts.append("删除 " + ", ".join(removed))
        return False, "文档集合已变化: " + "；".join(parts), False

    needs_refresh = False
    for name, old in saved.items():
        new = snapshot[name]
        if old.get("sha256") != new["sha256"]:
            return False, f"文档内容已变化: {name}", False
        # 哈希相同但元数据不同 —— 内容没变，索引有效，只需刷新清单
        if old.get("size") != new["size"] or old.get("mtime_ns") != new["mtime_ns"]:
            needs_refresh = True

    if needs_refresh:
        return True, "索引可复用（内容未变，仅元数据变化）", True

    return True, f"索引新鲜，{manifest.get('chunk_count', '?')} 个块", False
