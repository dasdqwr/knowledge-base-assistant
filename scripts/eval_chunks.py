"""分块质量报告（纯静态分析，不调用任何 API）。

为什么单独做这一层
------------------
RAG 出问题时有三种病因：分块坏了、检索没召回、生成没用对。三者混在一起时
指标会互相污染 —— 检索 Recall 偏低，可能是检索器差，也可能是块本身残缺。
所以先单独量化"块的质量"，把它变成一个可以独立改善的变量。

指标分四组
----------
  1. 规模    ：文档数、块数、每篇分布
  2. 大小    ：min/中位/max、过小块、超长块
  3. 结构    ：代码围栏配对率、围栏坏块分类、标题元数据覆盖
  4. 冗余    ：完全重复块、近似重复块

用法
----
    python scripts/eval_chunks.py                     # 只打报告
    python scripts/eval_chunks.py --save baseline.json # 存基线快照
    python scripts/eval_chunks.py --compare baseline.json
                                                       # 与基线对比（参数不一致则拒绝）

设计约束
--------
块边界会随切分参数变化，所以快照里必须带 params；对比时先校验 params，
不一致就直接报错退出，而不是给出一个误导性的差值。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, SUPPORTED_SUFFIXES
from src.loader import load_documents, split_documents

# ---------------------------------------------------------------------------
# 围栏识别
# ---------------------------------------------------------------------------
# CommonMark 允许围栏前有 0-3 个空格；项目里还有 " - ```java " 这种写在列表项
# 里的围栏，所以额外容忍一个列表标记。不这样放宽会漏掉真实存在的代码块。
FENCE_LINE_RE = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?:(?:[-*+]|\d+[.)])\s+)?"          # 可选的列表标记
    r"(?P<fence>`{3,}|~{3,})"
    r"(?P<info>[^`~]*)$"
)

# 过小阈值：低于这个长度的块很难独立表达一个完整语义单元
TINY_CHARS = 50
# 近似重复判定的 Jaccard 阈值
NEAR_DUP_THRESHOLD = 0.85
# 近似重复用的字符 n-gram 长度
SHINGLE = 5


def fence_lines(text: str) -> list[tuple[int, str, str]]:
    """返回 [(行号, 围栏字符, info)]，只认真正的围栏行（不是在行内出现的反引号）。"""
    out = []
    for i, line in enumerate(text.splitlines()):
        m = FENCE_LINE_RE.match(line)
        if m:
            out.append((i, m.group("fence")[0], m.group("info").strip()))
    return out


def count_fences(text: str) -> int:
    return len(fence_lines(text))


def fence_balanced(text: str) -> bool:
    """围栏计数为偶数即视为配对。

    这是近似判定：严格来说 ``` 与 ~~~ 不能互相闭合，但一个块的围栏不配对
    通常就是切分边界造成的，偶数/奇数足以区分。
    """
    return count_fences(text) % 2 == 0


def code_char_ratio(text: str) -> tuple[int, int]:
    """返回 (围栏内字符数, 总字符数)。

    围栏内的内容算"代码/命令"，围栏外的算"正文"。用于看块是不是被代码占满。
    """
    inside = 0
    total = len(text)
    open_ch = None
    for line in text.splitlines():
        m = FENCE_LINE_RE.match(line)
        if m:
            ch = m.group("fence")[0]
            if open_ch is None:
                open_ch = ch
                continue                      # 围栏行本身不算代码内容
            elif ch == open_ch:
                open_ch = None
                continue
        if open_ch is not None:
            inside += len(line) + 1           # +1 补回换行
    return inside, total


# ---------------------------------------------------------------------------
# 冗余检测
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def shingles(text: str) -> set[str]:
    t = normalize(text)
    if len(t) < SHINGLE:
        return {t} if t else set()
    return {t[i:i + SHINGLE] for i in range(len(t) - SHINGLE + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def find_duplicates(chunks: list) -> tuple[list, list]:
    """返回 (完全重复组, 近似重复对)。"""
    # 完全重复
    by_hash = {}
    for idx, c in enumerate(chunks):
        h = hashlib.sha256(normalize(c.page_content).encode("utf-8")).hexdigest()
        by_hash.setdefault(h, []).append(idx)
    exact = [idxs for idxs in by_hash.values() if len(idxs) > 1]

    # 近似重复：只在长度足够的块之间比，避免短块被 n-gram 误判
    sigs = [(i, shingles(c.page_content))
            for i, c in enumerate(chunks)
            if len(normalize(c.page_content)) >= 100]
    near = []
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            ia, sa = sigs[a]
            ib, sb = sigs[b]
            j = jaccard(sa, sb)
            if j >= NEAR_DUP_THRESHOLD:
                near.append((ia, ib, round(j, 3)))
    return exact, near


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------
def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return int(s[lo] + (s[hi] - s[lo]) * (k - lo))


def doc_fingerprint(path: Path) -> dict:
    return {
        "name": path.name,
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
    }


def analyze() -> dict:
    docs = load_documents()
    chunks = split_documents(docs)

    # ---- 参数快照：对比时必须一致 ----
    # 只统计真正参与索引的文件：按支持后缀过滤，否则会把
    # langchain.md.bak-20261017-140109 这类备份文件也算成一篇文档。
    data_dir = Path(DATA_DIR)
    files = sorted(p for p in data_dir.iterdir()
                   if p.is_file()
                   and not p.name.startswith(".")
                   and p.suffix.lower() in SUPPORTED_SUFFIXES)
    params = {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "doc_count": len(files),
        "chunk_count": len(chunks),
        "files": [doc_fingerprint(p) for p in files],
    }

    sizes = [len(c.page_content) for c in chunks]

    # ---- 围栏统计 ----
    broken = []
    for i, c in enumerate(chunks):
        if not fence_balanced(c.page_content):
            first = next((l for l in c.page_content.splitlines() if l.strip()), "")
            last = next((l for l in reversed(c.page_content.splitlines()) if l.strip()), "")
            starts = bool(FENCE_LINE_RE.match(first))
            ends = bool(FENCE_LINE_RE.match(last))
            if starts and not ends:
                kind = "缺闭合围栏"
            elif ends and not starts:
                kind = "缺开启围栏"
            else:
                kind = "围栏在块中间"
            broken.append({
                "index": i,
                "source": Path(c.metadata.get("source", "?")).name,
                "kind": kind,
                "fence_count": count_fences(c.page_content),
                "length": len(c.page_content),
                "first_line": first[:80],
            })

    # ---- 代码占比 ----
    code_chars = 0
    total_chars = 0
    for c in chunks:
        ci, ti = code_char_ratio(c.page_content)
        code_chars += ci
        total_chars += ti

    # ---- 标题元数据 ----
    header_cov = {
        lvl: sum(1 for c in chunks if c.metadata.get(lvl))
        for lvl in ("h1", "h2", "h3")
    }

    # ---- 冗余 ----
    exact, near = find_duplicates(chunks)

    # ---- 每篇文档分布 ----
    per_doc = Counter(Path(c.metadata.get("source", "?")).name for c in chunks)

    # ---- 碎片块 ----
    tiny = [{"index": i, "length": sizes[i],
             "source": Path(chunks[i].metadata.get("source", "?")).name,
             "preview": normalize(chunks[i].page_content)[:60]}
            for i, s in enumerate(sizes) if s < TINY_CHARS]

    return {
        "params": params,
        "summary": {
            "doc_count": len(files),
            "chunk_count": len(chunks),
            "chunks_per_doc": dict(per_doc),
            "size_min": min(sizes) if sizes else 0,
            "size_median": percentile(sizes, 0.5),
            "size_p90": percentile(sizes, 0.9),
            "size_max": max(sizes) if sizes else 0,
            "size_mean": round(sum(sizes) / len(sizes), 1) if sizes else 0,
            "tiny_chunks": len(tiny),
            "oversize_chunks": sum(1 for s in sizes if s > CHUNK_SIZE),
            "broken_fence_chunks": len(broken),
            "broken_fence_ratio": round(len(broken) / len(chunks), 4) if chunks else 0,
            "code_char_ratio": round(code_chars / total_chars, 4) if total_chars else 0,
            "exact_dup_groups": len(exact),
            "exact_dup_chunks": sum(len(g) for g in exact),
            "near_dup_pairs": len(near),
            "header_coverage": header_cov,
        },
        "broken_fence_detail": broken,
        "tiny_detail": tiny,
        "exact_dup_groups": exact,
        "near_dup_pairs": near,
    }


# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------
def bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "█" * filled + "·" * (width - filled)


def print_report(r: dict) -> None:
    s = r["summary"]
    p = r["params"]

    print()
    print("=" * 74)
    print("分块质量报告")
    print("=" * 74)
    print(f"  文档数 {s['doc_count']}   块数 {s['chunk_count']}"
          f"   CHUNK_SIZE={p['chunk_size']}  CHUNK_OVERLAP={p['chunk_overlap']}")
    for name, n in s["chunks_per_doc"].items():
        print(f"    {name:32s} {n:4d} 块")

    print()
    print("-" * 74)
    print("① 块大小分布")
    print("-" * 74)
    print(f"  min {s['size_min']:5d}   中位 {s['size_median']:5d}   "
          f"P90 {s['size_p90']:5d}   max {s['size_max']:5d}   均值 {s['size_mean']}")
    tiny_pct = s["tiny_chunks"] / s["chunk_count"] if s["chunk_count"] else 0
    over_pct = s["oversize_chunks"] / s["chunk_count"] if s["chunk_count"] else 0
    print(f"  过小块 (<{TINY_CHARS} 字符): {s['tiny_chunks']:3d}  "
          f"[{bar(tiny_pct)}] {tiny_pct*100:5.1f}%")
    print(f"  超长块 (>CHUNK_SIZE)   : {s['oversize_chunks']:3d}  "
          f"[{bar(over_pct)}] {over_pct*100:5.1f}%")

    print()
    print("-" * 74)
    print("② 代码围栏完整性")
    print("-" * 74)
    ratio = s["broken_fence_ratio"]
    ok = 1 - ratio
    print(f"  配对正常: {s['chunk_count'] - s['broken_fence_chunks']:3d} / {s['chunk_count']}  "
          f"[{bar(ok)}] {ok*100:5.1f}%")
    print(f"  坏块    : {s['broken_fence_chunks']:3d} / {s['chunk_count']}  "
          f"[{bar(ratio)}] {ratio*100:5.1f}%")
    if s["broken_fence_chunks"]:
        kinds = Counter(b["kind"] for b in r["broken_fence_detail"])
        for k, v in kinds.most_common():
            print(f"      · {k:12s} {v:3d}")
    print(f"  代码字符占比: {s['code_char_ratio']*100:.1f}%  "
          f"（围栏内内容 / 全部内容）")

    print()
    print("-" * 74)
    print("③ 标题元数据覆盖")
    print("-" * 74)
    n = s["chunk_count"] or 1
    for lvl in ("h1", "h2", "h3"):
        v = s["header_coverage"][lvl]
        print(f"  {lvl}: {v:3d} / {n}  [{bar(v/n)}] {v/n*100:5.1f}%")

    print()
    print("-" * 74)
    print("④ 冗余")
    print("-" * 74)
    print(f"  完全重复组: {s['exact_dup_groups']}  （涉及 {s['exact_dup_chunks']} 个块）")
    print(f"  近似重复对: {s['near_dup_pairs']}  （Jaccard ≥ {NEAR_DUP_THRESHOLD}）")

    # ---- 问题清单 ----
    issues = []
    if s["broken_fence_chunks"]:
        issues.append(
            f"代码围栏被切碎：{s['broken_fence_chunks']} 个块（{ratio*100:.1f}%）含未配对围栏。"
            f"根因是 RecursiveCharacterTextSplitter 不认识围栏，在代码块中间切分。")
    if s["tiny_chunks"]:
        issues.append(
            f"过小块 {s['tiny_chunks']} 个（<{TINY_CHARS} 字符）：语义不完整，"
            f"嵌入向量信息量低，检索时容易污染 Top-K。")
    if s["exact_dup_chunks"]:
        issues.append(
            f"完全重复块 {s['exact_dup_chunks']} 个：白占嵌入成本，检索时还会重复占位。")
    if s["near_dup_pairs"]:
        issues.append(
            f"近似重复对 {s['near_dup_pairs']} 对：通常是 chunk_overlap={p['chunk_overlap']} "
            f"造成的相邻块重叠，属正常但会稀释排序信号。")
    if s["header_coverage"]["h2"] < n:
        issues.append(
            f"有 {n - s['header_coverage']['h2']} 个块缺 h2 元数据：这些块来自无二级标题的段落，"
            f"无法靠元数据过滤定位。")

    print()
    print("-" * 74)
    print("⑤ 问题清单（按影响排序）")
    print("-" * 74)
    if issues:
        for i, t in enumerate(issues, 1):
            print(f"  {i}. {t}")
    else:
        print("  未发现明显问题")

    # 坏块样例
    if r["broken_fence_detail"]:
        print()
        print("-" * 74)
        print("⑥ 坏块样例（前 5 个）")
        print("-" * 74)
        for b in r["broken_fence_detail"][:5]:
            print(f"  [{b['index']:3d}] {b['source']:24s} {b['kind']:10s} "
                  f"围栏数={b['fence_count']} 长度={b['length']}")
            print(f"        首行: {b['first_line']!r}")
    print()


# ---------------------------------------------------------------------------
# 快照对比
# ---------------------------------------------------------------------------
def compare(current: dict, baseline: dict) -> int:
    """对比两份快照。params 不一致则拒绝比较 —— 块边界变了，差值没有意义。"""
    bp = baseline.get("params", {})
    cp = current["params"]
    if bp.get("chunk_size") != cp["chunk_size"] or bp.get("chunk_overlap") != cp["chunk_overlap"]:
        print("=" * 74)
        print("拒绝比较：切分参数不一致")
        print("=" * 74)
        print(f"  基线  chunk_size={bp.get('chunk_size')} overlap={bp.get('chunk_overlap')}")
        print(f"  当前  chunk_size={cp['chunk_size']} overlap={cp['chunk_overlap']}")
        print("  块边界已改变，指标差值不可比较。请在新参数下重建基线。")
        return 2

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    def row(name, b, c, better_low=False, suffix=""):
        d = c - b
        if abs(d) < 1e-9:
            arrow = "  ="
        elif d < 0:
            arrow = " ↓ "
        else:
            arrow = " ↑ "
        good = ""
        if abs(d) > 1e-9:
            improved = (d < 0) if better_low else (d > 0)
            good = "✅" if improved else "⚠️"
        print(f"  {name:26s} {fmt(b):>10} → {fmt(c):>10}   {d:+9.2f}{suffix} {arrow} {good}")

    bs, cs = baseline["summary"], current["summary"]
    print()
    print("=" * 74)
    print("与基线对比")
    print("=" * 74)
    print(f"  基线块数 {bs['chunk_count']} → 当前 {cs['chunk_count']}")

    # 数据文件是否变过 —— 变了的话，差值里混着"换了文档"这个因素，
    # 不能直接归因到切分改动上。
    old_files = {f["name"]: f["sha256"] for f in bp.get("files", [])}
    new_files = {f["name"]: f["sha256"] for f in cp["files"]}
    changed = sorted(n for n in old_files if n in new_files and old_files[n] != new_files[n])
    added = sorted(set(new_files) - set(old_files))
    removed = sorted(set(old_files) - set(new_files))
    if changed or added or removed:
        print()
        print("  ⚠️  数据文件本身也变了，差值不能只归因于切分逻辑：")
        if added:
            print(f"      新增: {', '.join(added)}")
        if removed:
            print(f"      删除: {', '.join(removed)}")
        if changed:
            print(f"      内容变化: {', '.join(changed)}")
        print("      想只衡量切分改动，请用同一份 data/ 重跑两次。")

    print()
    row("坏围栏块数", bs["broken_fence_chunks"], cs["broken_fence_chunks"], better_low=True)
    row("坏围栏占比(%)", bs["broken_fence_ratio"]*100, cs["broken_fence_ratio"]*100, better_low=True)
    row("过小块数", bs["tiny_chunks"], cs["tiny_chunks"], better_low=True)
    row("超长块数", bs["oversize_chunks"], cs["oversize_chunks"], better_low=True)
    row("完全重复块数", bs["exact_dup_chunks"], cs["exact_dup_chunks"], better_low=True)
    row("近似重复对数", bs["near_dup_pairs"], cs["near_dup_pairs"], better_low=True)
    row("块大小中位数", bs["size_median"], cs["size_median"])
    row("代码字符占比(%)", bs["code_char_ratio"]*100, cs["code_char_ratio"]*100)
    row("h1 覆盖块数", bs["header_coverage"]["h1"], cs["header_coverage"]["h1"])
    row("h2 覆盖块数", bs["header_coverage"]["h2"], cs["header_coverage"]["h2"])
    print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="分块质量报告（纯静态，不需要 API）")
    ap.add_argument("--save", metavar="PATH", help="把本次快照写入 JSON 文件")
    ap.add_argument("--compare", metavar="PATH", help="与已有快照对比")
    ap.add_argument("--json", action="store_true", help="额外打印原始 JSON")
    args = ap.parse_args()

    result = analyze()
    print_report(result)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"快照已写入: {path}")

    if args.compare:
        path = Path(args.compare)
        if not path.is_file():
            print(f"基线文件不存在: {path}")
            return 1
        return compare(result, json.loads(path.read_text(encoding="utf-8")))

    return 0


if __name__ == "__main__":
    sys.exit(main())
