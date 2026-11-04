"""从现有文档块反生成检索评估数据集。

思路
----
手工写 40 条评估问题要几小时，而且容易只挑自己熟悉的部分（有偏）。
反过来做更省力也更公平：**从块生成问题**，出处（gold chunk）天然已知。

   采样块 → LLM 读块生成问题 + 必须出现的字符串 → 自动校验 → 人工抽查

关键设计：金标准用「问题 + must_contain 字符串」而不是 chunk_id。
块边界会随切分参数变化，chunk_id 一改就全废；而"某个字符串必须出现在
正确块里"跨切分依然成立。

用法
----
    python scripts/make_eval_dataset.py                 # 生成 40 条
    python scripts/make_eval_dataset.py --count 60
    python scripts/make_eval_dataset.py --dry-run       # 只采样不调 LLM

生成结果需要人工抽查：删掉"问题太泛""字符串太常见""其实答不全"的条目。
校验只能保证形式正确，判断不了语义质量。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path(__file__).resolve().parent.parent / "tests" / "eval_retrieval_dataset.json"

MIN_CHUNK_CHARS = 150      # 太短的块问不出有信息量的问题
MAX_CHUNK_FOR_PROMPT = 1500  # 喂给 LLM 的块内容上限，避免超长
MAX_GOLD_HITS = 2          # must_contain 命中块数超过这个值视为"不够有区分度"

PROMPT = """你在为一个技术文档知识库构建检索评估集。

下面是一个文档块（可能含标题、正文、代码）。请生成 **1 个** 该块能够回答的问题，
并给出 2~3 个 **必须出现在该块里** 的字符串作为答案定位标记。

要求：
1. 问题要具体，像真实用户会问的（例如"init_chat_model 怎么指定 model_provider"），
   不要用"这段代码讲什么"这种空泛问题。
2. 字符串必须在块中**逐字出现**（区分大小写），优先选有区分度的：
   函数名、参数名、类名、独特名词短语。不要选"python""函数""使用"这类通用词。
3. 如果这个块信息太碎、问不出有意义的问题，返回 {{"answerable": false}}。
4. 只输出 JSON，不要解释、不要 markdown 代码围栏。

JSON 格式：
{{"answerable": true, "question": "……", "must_contain": ["……", "……"]}}

文档块（来源：{source}{header}）：
---
{content}
---
"""


def load_chunks() -> list:
    from src.loader import load_documents, split_documents

    docs = load_documents()
    return split_documents(docs)


def header_path(meta: dict) -> str:
    parts = [meta.get(k) for k in ("h1", "h2", "h3")]
    parts = [p for p in parts if p]
    return "｜" + " > ".join(parts) if parts else ""


def is_mostly_code(text: str, meta: dict) -> bool:
    """粗判：代码块占了大半时也保留（API 用法类问题很重要），这里只标记。"""
    return bool(meta.get("Code"))


def sample_chunks(chunks: list, count: int, seed: int = 42) -> list:
    """分层采样：按来源分组，组内均匀取样，尽量避免全是同一章。"""
    usable = [c for c in chunks
              if len(c.page_content) >= MIN_CHUNK_CHARS
              and len(c.page_content.strip()) > 0]
    by_source: dict[str, list] = {}
    for c in usable:
        src = Path(c.metadata.get("source", "?")).name
        by_source.setdefault(src, []).append(c)

    rng = random.Random(seed)
    picked: list = []
    sources = sorted(by_source)
    # 先按来源轮转，保证两篇文档都有代表
    idx = 0
    while len(picked) < count:
        progressed = False
        for src in sources:
            pool = by_source[src]
            if idx < len(pool):
                picked.append(pool[idx])
                progressed = True
                if len(picked) >= count:
                    break
        if not progressed:
            break
        idx += 1

    # 层内再加一点随机性，避免总取同样的序列
    head = picked[: len(sources)]
    tail = picked[len(sources):]
    rng.shuffle(tail)
    return head + tail


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    m = JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def count_gold_hits(chunks: list, must: list[str]) -> int:
    """must_contain 全中的块有多少个 —— 用来判断这个金标准够不够独特。"""
    return sum(1 for c in chunks if all(s in c.page_content for s in must))


def main() -> int:
    ap = argparse.ArgumentParser(description="从文档块反生成检索评估数据集")
    ap.add_argument("--count", type=int, default=40, help="生成条数（默认 40）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dry-run", action="store_true", help="只采样，不调用 LLM")
    args = ap.parse_args()

    chunks = load_chunks()
    print(f"总块数 {len(chunks)}，采样 {args.count} 个用于生成问题")
    picked = sample_chunks(chunks, args.count, args.seed)

    if args.dry_run:
        for i, c in enumerate(picked[:10], 1):
            print(f"[{i}] {Path(c.metadata.get('source','?')).name} "
                  f"len={len(c.page_content)} meta={ {k: v for k, v in c.metadata.items() if k.startswith('h')} }")
        print(f"（dry-run 只显示前 10 个，共 {len(picked)} 个）")
        return 0

    from src.model import get_model

    model = get_model().bind(temperature=0)

    items: list[dict] = []
    skipped: list[str] = []
    for i, c in enumerate(picked, 1):
        src = Path(c.metadata.get("source", "?")).name
        content = c.page_content[:MAX_CHUNK_FOR_PROMPT]
        prompt = PROMPT.format(
            source=src,
            header=header_path(c.metadata),
            content=content,
        )
        try:
            resp = model.invoke(prompt)
            data = parse_json(getattr(resp, "content", str(resp)))
        except Exception as e:
            skipped.append(f"#{i} 调用失败: {type(e).__name__}: {e}")
            continue

        if not data or not data.get("answerable", True):
            skipped.append(f"#{i} 模型判定不可提问")
            continue

        q = (data.get("question") or "").strip()
        must = [s for s in (data.get("must_contain") or []) if isinstance(s, str) and s.strip()]

        # 校验 1：必须逐字出现在该块里
        not_in_chunk = [s for s in must if s not in c.page_content]
        if not q or len(must) < 1:
            skipped.append(f"#{i} 字段缺失")
            continue
        if not_in_chunk:
            skipped.append(f"#{i} 字符串不在块内: {not_in_chunk}")
            continue

        # 校验 2：不能太常见（否则召回率虚高）
        hits = count_gold_hits(chunks, must)
        if hits > MAX_GOLD_HITS:
            skipped.append(f"#{i} 区分度不足（{hits} 个块都含 {must}）")
            continue

        items.append({
            "id": f"r{len(items)+1:03d}",
            "question": q,
            "must_contain": must,
            "source": src,
            "header": header_path(c.metadata).lstrip("｜"),
            "gold_hits": hits,
            "chunk_chars": len(c.page_content),
        })
        print(f"  [{len(items):02d}] {q}  → {must}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"生成 {len(items)} 条，跳过 {len(skipped)} 条")
    if skipped:
        for s in skipped[:10]:
            print("  跳过:", s)
    print(f"已写入 {out}")
    print("⚠️ 请人工抽查：删掉问题太泛、字符串太常见、实际答不全的条目")
    return 0


if __name__ == "__main__":
    sys.exit(main())
