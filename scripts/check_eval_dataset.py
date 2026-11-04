"""校验检索评估数据集的质量。

为什么需要它
------------
自动生成的数据集"形式上都对"，但语义质量只有两条能靠程序判：
  1. must_contain 是否真的出现在某些块里（否则这题永远不可能命中）
  2. must_contain 是否太常见 / 太长 / 直接出现在题干里（否则这题退化成字符查找）

剩下的（问题是否自然、答案是否真在该块）必须人工看。这个脚本负责把需要人看的
条目挑出来，而不是替你判断。

用法
----
    python scripts/check_eval_dataset.py
    python scripts/check_eval_dataset.py --strict   # 有 ERROR 时退出码 1（可接 CI）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATASET = Path(__file__).resolve().parent.parent / "tests" / "eval_retrieval_dataset.json"

# 阈值（按经验定的，可按需调整）
MAX_MUST_CHARS = 40        # must_contain 单串超过这个长度 → 变成"找整行"
MAX_GOLD_HITS = 3          # 全中该串的块数，超过则区分度不足
MIN_QUESTION_CHARS = 12    # 太短的问题通常太泛
LEAK_RATIO_WARN = 1.0      # 题干包含全部 must_contain → 退化成精确查找
DUP_SIMILARITY = 0.85


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def main() -> int:
    ap = argparse.ArgumentParser(description="校验检索评估数据集")
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--strict", action="store_true", help="有 ERROR 时返回非零退出码")
    args = ap.parse_args()

    path = Path(args.dataset)
    if not path.is_file():
        print(f"数据集不存在: {path}")
        return 1
    ds = json.loads(path.read_text(encoding="utf-8"))

    from src.loader import load_documents, split_documents

    chunks = split_documents(load_documents())
    print(f"数据集 {len(ds)} 条，当前块 {len(chunks)} 个\n")

    errors: list[str] = []
    warns: list[str] = []

    def err(item_id: str, msg: str) -> None:
        errors.append(f"{item_id}: {msg}")

    def warn(item_id: str, msg: str) -> None:
        warns.append(f"{item_id}: {msg}")

    # ---- 逐条检查 ----
    for it in ds:
        iid = it.get("id", "?")
        q = it.get("question", "")
        must = it.get("must_contain") or []

        if not q or not isinstance(q, str):
            err(iid, "缺少 question")
            continue
        if not must or not isinstance(must, list):
            err(iid, "缺少 must_contain")
            continue

        # 1. 每个串都必须真的出现在某些块里，否则这题不可能命中
        for s in must:
            hits = [c for c in chunks if s in c.page_content]
            if not hits:
                err(iid, f"字符串 {s!r} 不在任何块里（这题永远无法命中）")
            elif len(hits) > MAX_GOLD_HITS:
                warn(iid, f"字符串 {s!r} 出现在 {len(hits)} 个块里（区分度低）")

        # 2. 单串过长 → 退化成"找整行代码"
        for s in must:
            if len(s) > MAX_MUST_CHARS:
                warn(iid, f"字符串过长({len(s)}字符)：{s[:40]!r}… 建议改成短标识符")

        # 3. 题干包含全部 must_contain → BM25 直接精确命中，题目太简单
        leaked = [s for s in must if s and s in q]
        if must and len(leaked) == len(must):
            warn(iid, f"题干包含全部金标准字符串 {leaked} → 退化为精确查找")
        elif leaked:
            warn(iid, f"题干包含部分金标准字符串 {leaked}")

        # 4. 问题过短/过泛
        if len(q) < MIN_QUESTION_CHARS:
            warn(iid, f"问题过短({len(q)}字符)，可能太泛")

        # 5. must_contain 之间不应互相包含（冗余）
        for i, a in enumerate(must):
            for b in must[i + 1:]:
                if a in b or b in a:
                    warn(iid, f"must_contain 冗余：{a!r} 与 {b!r}")

    # ---- 跨条重复 ----
    for i in range(len(ds)):
        for j in range(i + 1, len(ds)):
            a, b = ds[i].get("question", ""), ds[j].get("question", "")
            if not a or not b:
                continue
            sim = SequenceMatcher(None, norm(a), norm(b)).ratio()
            if sim >= DUP_SIMILARITY:
                warn(ds[i].get("id", "?"), f"与 {ds[j].get('id','?')} 高度相似({sim:.2f})")

    # ---- 输出 ----
    print("=" * 78)
    print("ERROR（必须修，否则该题无效）")
    print("=" * 78)
    if errors:
        for e in errors:
            print("  ❌", e)
    else:
        print("  无")

    print()
    print("=" * 78)
    print("WARN（建议看，可能让指标失真）")
    print("=" * 78)
    if warns:
        for w in warns:
            print("  ⚠️ ", w)
    else:
        print("  无")

    # 统计
    leak_all = 0
    for it in ds:
        must = it.get("must_contain") or []
        q = it.get("question", "")
        if must and all(s in q for s in must):
            leak_all += 1

    print()
    print("=" * 78)
    print("汇总")
    print("=" * 78)
    print(f"  条目数                     {len(ds)}")
    print(f"  ERROR                      {len(errors)}")
    print(f"  WARN                       {len(warns)}")
    print(f"  题干含全部金标准（偏简单）  {leak_all} 条 ({leak_all/len(ds)*100:.0f}%)")
    print()
    print("  建议配比：约 1/3 精确标识符题（含 API 名，测关键词召回）")
    print("           约 2/3 改述题（不含金标准字符串，测语义召回）")
    print("  目前偏简单或偏难都能接受，但**不要清一色都是简单题**，")
    print("  否则混合检索的优势会被高估、微调权重时看不到真实差异。")

    if args.strict and errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
