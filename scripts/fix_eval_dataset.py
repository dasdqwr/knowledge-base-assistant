"""为评估集里的条目提议更好的 must_contain（金标准定位字符串）。

要解决什么问题
--------------
自动生成数据集时，模型常把"整行代码"或"文档里到处都有的词"当成金标准：

    'BaseModel'          → 出现在 22 个块里，定位不到具体答案
    "name: str"          → 出现在 21 个块里
    "{'name': <class 'str'>, ...}" → 48 字符，退化成"查找这一整行"

好的金标准应该：**短、稀、且不是题干里已经给出的词**。

这个工具做什么
--------------
对每条记录：
  1. 找出当前金标准锁定的块（gold chunk）
  2. 从该块里抽取候选标识符，按「稀有权重 × 长度权重」排序
  3. 排除：题干里已出现的词（避免退化成精确查找）、常见编程词（停用表）
  4. 提议 1~2 个新字符串，仍然能唯一定位到同一个块

只提议，不改文件；确认后加 --apply 才写入。

用法
----
    python scripts/fix_eval_dataset.py                 # 只提议
    python scripts/fix_eval_dataset.py --apply         # 写入
    python scripts/fix_eval_dataset.py --ids r017,r031 # 只处理指定条目
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATASET = Path(__file__).resolve().parent.parent / "tests" / "eval_retrieval_dataset.json"

MAX_HITS_OK = 2        # 出现块数 ≤ 这个值才算"稀有"
MAX_LEN_OK = 40        # 单串长度 ≤ 这个值才算"短"
MIN_TOKEN_LEN = 3

# 太常见、区分度低的词（出现在多数技术文档里）
STOPWORDS = {
    "python", "langchain", "langgraph", "import", "from", "class", "def", "self",
    "return", "true", "false", "none", "print", "str", "int", "float", "bool",
    "list", "dict", "tuple", "set", "optional", "model", "value", "values",
    "default", "field", "fields", "data", "type", "types", "name", "age",
    "true", "false", "async", "await", "get", "post", "init", "main",
    "python3", "pip", "install",
}

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{2,}")
# 中文里较长的专有短语（「结构化类型」这类），也作为候选
CN_PHRASE_RE = re.compile(r"[\u4e00-\u9fa5]{4,12}")


def rarity(chunks: list, token: str) -> int:
    return sum(1 for c in chunks if token in c.page_content)


def candidates_for(chunk_text: str, question: str, chunks: list) -> list[tuple[float, str]]:
    """从金块里挑候选定位串。

    只挑 ASCII 标识符，不挑中文片段 —— 实测中文短语会被截成
    「查看每次调用的」这种无意义片段，且容易指向块里与问题无关的另一段。

    另外用「问题关键词在块中的位置」当锚点：候选越靠近锚点越优先。
    否则会挑到同块里另一个话题的稀有词（比如问 `__annotations__` 的输出，
    却挑到 `from __future__ import annotations` 那段的词）。
    """
    lines = chunk_text.splitlines()

    # 锚点：问题里出现在块中的最长标识符
    q_tokens = [t for t in TOKEN_RE.findall(question)]
    anchors = [(len(t), t) for t in q_tokens if t in chunk_text]
    anchors.sort(reverse=True)
    anchor_line = None
    if anchors:
        anchor = anchors[0][1]
        for i, line in enumerate(lines):
            if anchor in line:
                anchor_line = i
                break

    scored: dict[str, float] = {}
    for t in set(TOKEN_RE.findall(chunk_text)):
        base = t.split(".")[-1].split("(")[0]
        if base.lower() in STOPWORDS or len(base) < MIN_TOKEN_LEN:
            continue
        if t in question or base in question:
            continue                      # 题干已给出 → 不能再当金标准
        # 只要"有结构"的标识符：带下划线、点、驼峰，或足够长的专名
        structured = ("_" in t) or ("." in t) or bool(re.search(r"[a-z][A-Z]", t)) or len(t) >= 8
        if not structured:
            continue
        n = rarity(chunks, t)
        if n == 0 or n > MAX_HITS_OK:
            continue

        # 与锚点的行距 → 越近越优先
        prox = 1.0
        if anchor_line is not None:
            dists = [abs(i - anchor_line) for i, line in enumerate(lines) if t in line]
            if dists:
                d = min(dists)
                prox = 1.0 / (1.0 + d / 5.0)     # 5 行内几乎不衰减，远处迅速降低
            else:
                prox = 0.2

        scored[t] = prox * (1.0 / n) + len(t) * 0.005

    return sorted(((s, t) for t, s in scored.items()), reverse=True)


def needs_fix(must: list[str], chunks: list, question: str) -> bool:
    for s in must:
        if len(s) > MAX_LEN_OK:
            return True
        if rarity(chunks, s) > MAX_HITS_OK:
            return True
    if must and all(s in question for s in must):
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="为数据集提议更好的 must_contain")
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--apply", action="store_true", help="写入文件（默认只提议）")
    ap.add_argument("--ids", default="", help="只处理这些 id，逗号分隔")
    ap.add_argument("--limit", type=int, default=0, help="最多提议几条")
    ap.add_argument("--show", default="", help="显示指定 id 的金块内容后退出（辅助手改）")
    args = ap.parse_args()

    path = Path(args.dataset)
    ds = json.loads(path.read_text(encoding="utf-8"))

    from src.loader import load_documents, split_documents

    chunks = split_documents(load_documents())

    # ---- --show：把金块原文打出来，方便手改 ----
    if args.show:
        want = {s.strip() for s in args.show.split(",") if s.strip()}
        for it in ds:
            if it["id"] not in want:
                continue
            must = it.get("must_contain") or []
            gold = [c for c in chunks if all(s in c.page_content for s in must)]
            print("=" * 78)
            print(f"{it['id']}  {it['question']}")
            print(f"当前 must_contain: {must}")
            for s in must:
                print(f"  {s!r} → 出现在 {rarity(chunks, s)} 个块")
            if not gold:
                print("❌ 金标准无法定位到任何块")
                continue
            print(f"金块（{len(gold)} 个，显示第一个，共 {len(gold[0].page_content)} 字符）:")
            for i, line in enumerate(gold[0].page_content.splitlines()[:40], 1):
                print(f"  {i:3d}| {line[:90]}")
            if len(gold) > 1:
                print(f"  （另有 {len(gold)-1} 个块也匹配，说明金标准区分度不足）")
            print()
        return 0
    only = {s.strip() for s in args.ids.split(",") if s.strip()}

    changed = 0
    print(f"{'id':<6}{'旧 must_contain':<50}→ 新提议")
    print("-" * 110)
    for it in ds:
        iid = it["id"]
        if only and iid not in only:
            continue
        must = it.get("must_contain") or []
        q = it.get("question", "")

        if not needs_fix(must, chunks, q):
            continue

        gold = [c for c in chunks if all(s in c.page_content for s in must)]
        if not gold:
            print(f"{iid:<6}{'（当前金标准已无法定位，跳过）'}")
            continue
        gold_text = gold[0].page_content

        cands = candidates_for(gold_text, q, chunks)
        # 取 2 个：能同时满足"稀有"且都落在同一个 gold 块
        picked: list[str] = []
        for _, tok in cands:
            if all(tok in c.page_content for c in [gold[0]]):
                picked.append(tok)
            if len(picked) >= 2:
                break

        if not picked:
            print(f"{iid:<6}{str(must)[:46]:<50}（找不到更稀有的候选，建议手改或删除该题）")
            continue

        print(f"{iid:<6}{str(must)[:46]:<50}→ {picked}")
        if args.apply:
            it["must_contain"] = picked
            it["gold_hits"] = rarity(chunks, picked[0])
        changed += 1
        if args.limit and changed >= args.limit:
            break

    print()
    print(f"共 {changed} 条{'已修改' if args.apply else '被提议'}（数据集 {len(ds)} 条）")
    if not args.apply and changed:
        print("确认无误后加 --apply 写入；写入后请重新跑一遍：")
        print("  python scripts/check_eval_dataset.py")
        print("  python scripts/eval_retrieval.py --save tests/eval_retrieval_baseline.json")

    if args.apply and changed:
        path.write_text(json.dumps(ds, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入 {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
