"""生成层评估：Faithfulness（忠实度）+ Answer Relevance（切题度）。

评什么
------
固定检索（混合 + Rerank，与生产一致），固定 prompt，固定 temperature=0，
然后让 LLM 裁判打分。**不经过 agent** —— agent 可能不调工具、会带记忆，
那测的是"模型会不会用工具"，不是"拿到正确上下文后答得好不好"。

两个指标
--------
Faithfulness（忠实度）
    答案里的论断能不能在给定上下文里找到依据。**RAG 最该盯的指标**，
    直接反映"有没有瞎编"。低分 → 收紧 prompt 或换模型。
    实现：让裁判把答案拆成论断，逐条判断是否被上下文支持，输出计数。

Answer Relevance（切题度）
    答案是否在回答所问。低分 → 答非所问，通常是 prompt 或模型问题。
    实现：裁判给 1~5 分（附评分标准），并说明理由。

不做 Correctness
    正确性需要参考答案（人写或强模型生成），否则测不了。
    且要注意：**忠实度高 ≠ 正确** —— 上下文本身错了，模型忠实转述也会高分。

裁判的偏差控制
--------------
    temperature=0        —— 减少评分漂移
    先理由后分数          —— 一致性明显更好
    强制 JSON 输出        —— 可自动聚合
    附原始上下文与答案     —— 便于人工复核低分样本

用法
----
    python scripts/eval_generation.py --limit 5     # 先试跑几条看裁判质量
    python scripts/eval_generation.py               # 全量
    python scripts/eval_generation.py --save tests/eval_generation_baseline.json
    python scripts/eval_generation.py --compare tests/eval_generation_baseline.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CHUNK_SIZE,
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    RERANK_CANDIDATES,
    TOP_K,
)

DATASET = Path(__file__).resolve().parent.parent / "tests" / "eval_retrieval_dataset.json"
MAX_CONTEXT_CHARS = 3000      # 裁判看到的上下文上限，控制成本
MAX_ANSWER_CHARS = 2000

FAITHFULNESS_PROMPT = """你是严格的评估员。判断【答案】中的论断是否都能在【上下文】中找到依据。

规则：
1. 把答案拆成若干条独立论断（通常 1~5 条）。
2. 逐条判断是否被上下文支持。"上下文没提到"算不支持，不要用你自己的知识补充。
3. 只输出 JSON，不要解释、不要 markdown 围栏。

JSON 格式：
{{"total": 论断总数, "supported": 被支持的条数, "unsupported": ["不支持的论断1", "…"]}}

【上下文】
{context}

【答案】
{answer}
"""

RELEVANCE_PROMPT = """你是严格的评估员。判断【答案】是否在回答【问题】。

评分标准：
5 = 直接、完整地回答了问题
4 = 基本回答，缺少少量细节
3 = 部分相关，但答非所问的成分明显
2 = 只有零散相关词，没有真正回答
1 = 完全无关，或明确拒绝回答

注意：答案长短不代表质量，简洁但切题应得高分；冗长跑题应低分。
先写理由，再给分数。只输出 JSON，不要 markdown 围栏。

JSON 格式：
{{"reason": "……", "score": 1-5 的整数}}

【问题】
{question}

【答案】
{answer}
"""

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    m = JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 生产一致的检索（混合 + Rerank，截断到 TOP_K）
# ---------------------------------------------------------------------------
def build_production_retriever():
    from src.loader import load_documents, split_documents
    from src.model import get_reranker
    from src.rag_chain import build_hybrid_retriever
    from src.vectorstore import load_vectorstore

    # 与 eval_retrieval 同一套一致性检查，避免块集合错位
    from eval_retrieval import check_index_consistency  # type: ignore

    check_index_consistency()

    vectorstore = load_vectorstore()
    chunks = split_documents(load_documents())
    hybrid = build_hybrid_retriever(vectorstore, chunks)
    reranker = get_reranker()

    def retrieve(question: str):
        cands = hybrid.invoke(question)[:RERANK_CANDIDATES]
        if not cands:
            return []
        scores = reranker.predict([[question, d.page_content] for d in cands])
        ranked = sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:TOP_K]]

    return retrieve


def build_prompt():
    """复用 src/rag_chain.py 的 RAG_PROMPT（生产 agent 没用它，这里作为固定模板）。"""
    from src.rag_chain import RAG_PROMPT

    return RAG_PROMPT


def answer_once(model, prompt_tpl, question: str, context: str) -> str:
    msgs = prompt_tpl.format_messages(context=context, question=question)
    resp = model.invoke(msgs)
    return getattr(resp, "content", str(resp)).strip()


def judge_faithfulness(model, context: str, answer: str) -> dict:
    prompt = FAITHFULNESS_PROMPT.format(
        context=context[:MAX_CONTEXT_CHARS],
        answer=answer[:MAX_ANSWER_CHARS],
    )
    data = parse_json(getattr(model.invoke(prompt), "content", ""))
    if not data:
        return {"total": 0, "supported": 0, "unsupported": [], "parse_error": True}
    try:
        total = int(data.get("total", 0))
        supported = int(data.get("supported", 0))
    except Exception:
        return {"total": 0, "supported": 0, "unsupported": [], "parse_error": True}
    return {
        "total": total,
        "supported": min(supported, total) if total else 0,
        "unsupported": data.get("unsupported") or [],
        "parse_error": False,
    }


def judge_relevance(model, question: str, answer: str) -> dict:
    prompt = RELEVANCE_PROMPT.format(
        question=question,
        answer=answer[:MAX_ANSWER_CHARS],
    )
    data = parse_json(getattr(model.invoke(prompt), "content", ""))
    if not data:
        return {"score": 0, "reason": "(裁判输出无法解析)", "parse_error": True}
    try:
        score = int(data.get("score", 0))
    except Exception:
        score = 0
    return {
        "score": score,
        "reason": (data.get("reason") or "")[:300],
        "parse_error": False,
    }


REFUSAL_MARKERS = ("我不知道", "没有找到", "无法回答", "没有相关信息", "无法确定")


def is_refusal(answer: str) -> bool:
    return any(m in answer for m in REFUSAL_MARKERS)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成层评估（忠实度 + 切题度）")
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（试跑用）")
    ap.add_argument("--save", metavar="PATH")
    ap.add_argument("--compare", metavar="PATH")
    ap.add_argument("--verbose", action="store_true", help="打印每条答案与裁判理由")
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.is_file():
        print(f"数据集不存在: {ds_path}（先跑 make_eval_dataset.py）")
        return 1
    dataset = json.loads(ds_path.read_text(encoding="utf-8"))
    if args.limit:
        dataset = dataset[: args.limit]
    print(f"数据集 {len(dataset)} 条")

    from src.model import get_model

    model = get_model().bind(temperature=0)
    prompt_tpl = build_prompt()
    retrieve = build_production_retriever()

    rows = []
    t_start = time.perf_counter()
    for i, item in enumerate(dataset, 1):
        q = item["question"]
        docs = retrieve(q)
        context = "\n\n".join(
            f"[来源: {Path(d.metadata.get('source', '未知')).name}]\n{d.page_content}"
            for d in docs
        )
        t0 = time.perf_counter()
        answer = answer_once(model, prompt_tpl, q, context)
        dt = time.perf_counter() - t0

        faith = judge_faithfulness(model, context, answer)
        rel = judge_relevance(model, q, answer)

        row = {
            "id": item["id"],
            "question": q,
            "answer_chars": len(answer),
            "answer_latency_s": round(dt, 2),
            "refusal": is_refusal(answer),
            "faith_total": faith["total"],
            "faith_supported": faith["supported"],
            # 保存"无依据的论断"原文，便于人工复核到底是模型瞎编，
            # 还是上下文里只有引导语、具体示例被切分切到了别的块
            "faith_unsupported": faith["unsupported"][:5],
            "faith_parse_error": faith["parse_error"],
            "relevance": rel["score"],
            "relevance_reason": rel["reason"],
            "n_context_docs": len(docs),
        }
        rows.append(row)

        fs = (faith["supported"] / faith["total"]) if faith["total"] else 0.0
        print(f"  [{i:02d}/{len(dataset)}] {item['id']}  faithful={fs:.2f} "
              f"({faith['supported']}/{faith['total']})  relevance={rel['score']}/5  "
              f"refusal={row['refusal']}  {dt:.1f}s")
        if args.verbose:
            print(f"      答案: {answer[:160]}…")
            print(f"      裁判: {rel['reason']}")

    elapsed = time.perf_counter() - t_start

    # 聚合：只在"成功解析出论断"的样本上算忠实度均值
    valid = [r for r in rows if r["faith_total"] > 0]
    faith_scores = [r["faith_supported"] / r["faith_total"] for r in valid]
    rel_scores = [r["relevance"] for r in rows if r["relevance"] > 0]

    summary = {
        "n": len(rows),
        "faithfulness_mean": round(sum(faith_scores) / len(faith_scores), 4) if faith_scores else 0.0,
        "faithfulness_n": len(faith_scores),
        "faith_parse_errors": sum(1 for r in rows if r["faith_parse_error"]),
        "relevance_mean": round(sum(rel_scores) / len(rel_scores), 4) if rel_scores else 0.0,
        "relevance_n": len(rel_scores),
        "relevance_parse_errors": sum(1 for r in rows if r["relevance"] == 0),
        "refusal_rate": round(sum(1 for r in rows if r["refusal"]) / len(rows), 4),
        "avg_answer_chars": round(sum(r["answer_chars"] for r in rows) / len(rows), 1),
        "elapsed_s": round(elapsed, 1),
    }

    print()
    print("=" * 74)
    print("生成层评估结果")
    print("=" * 74)
    print(f"  样本数              {summary['n']}")
    print(f"  Faithfulness 均值   {summary['faithfulness_mean']:.3f}  "
          f"（基于 {summary['faithfulness_n']} 条可解析样本）")
    print(f"  Relevance 均值      {summary['relevance_mean']:.2f} / 5  "
          f"（基于 {summary['relevance_n']} 条）")
    print(f"  拒答率              {summary['refusal_rate']*100:.1f}%")
    print(f"  平均答案长度        {summary['avg_answer_chars']:.0f} 字符")
    print(f"  裁判解析失败        faithfulness {summary['faith_parse_errors']} 条 / "
          f"relevance {summary['relevance_parse_errors']} 条")
    print(f"  总耗时              {summary['elapsed_s']:.0f}s")

    # 低分样本，方便人工复核（自动指标只是筛子，不是判决）
    print()
    print("低分样本（供人工复核）")
    low = sorted(valid, key=lambda r: r["faith_supported"] / r["faith_total"])[:5]
    shown = 0
    for r in low:
        fs = r["faith_supported"] / r["faith_total"]
        if fs >= 0.999 and r["relevance"] >= 4:
            continue
        shown += 1
        print(f"  {r['id']}  faithful={fs:.2f} relevance={r['relevance']}/5  {r['question'][:44]}")
        print(f"      裁判理由: {r['relevance_reason'][:90]}")
    if shown == 0:
        print("  无（全部高分）")

    payload = {
        "params": {
            "chunk_size": CHUNK_SIZE,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_backend": (EMBEDDING_BACKEND or "api").strip().lower(),
            "top_k": TOP_K,
            "rerank_candidates": RERANK_CANDIDATES,
            "n_dataset": len(dataset),
        },
        "summary": summary,
        "rows": rows,
    }

    if args.save:
        p = Path(args.save)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n基线已写入 {p}")

    if args.compare:
        p = Path(args.compare)
        if not p.is_file():
            print(f"基线不存在: {p}")
            return 1
        return compare(summary, payload["params"], json.loads(p.read_text(encoding="utf-8")))

    return 0


def compare(cur: dict, cur_params: dict, baseline: dict) -> int:
    bp = baseline.get("params", {})
    if (bp.get("chunk_size") != cur_params["chunk_size"]
            or bp.get("embedding_model") != cur_params["embedding_model"]
            or bp.get("n_dataset") != cur_params["n_dataset"]):
        print("=" * 74)
        print("拒绝比较：切分/嵌入/数据集规模不一致")
        print("=" * 74)
        print(f"  基线 {bp}")
        print(f"  当前 {cur_params}")
        print("  请在同一配置与同一数据集上重取基线。")
        return 2

    bs = baseline["summary"]
    print()
    print("=" * 74)
    print("与基线对比")
    print("=" * 74)

    def row(name, b, c, unit="", higher_better=True):
        d = c - b
        if abs(d) < 1e-9:
            mark = "  ="
        else:
            improved = (d > 0) if higher_better else (d < 0)
            mark = "✅" if improved else "⚠️"
        print(f"  {name:<22} {b:>8.3f} → {c:>8.3f}   {d:+.3f}{unit}  {mark}")

    row("Faithfulness", bs["faithfulness_mean"], cur["faithfulness_mean"])
    row("Relevance(1-5)", bs["relevance_mean"], cur["relevance_mean"])
    row("拒答率", bs["refusal_rate"] * 100, cur["refusal_rate"] * 100, "%", higher_better=False)
    row("平均答案长度", bs["avg_answer_chars"], cur["avg_answer_chars"], higher_better=False)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
