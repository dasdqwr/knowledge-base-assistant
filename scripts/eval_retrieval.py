"""检索层评估：三种配置的 Recall@k / MRR / Hit@1 对比。

评什么
------
只评"检索器返回了哪些块"，不评生成。金标准用「must_contain 字符串」而不是
chunk_id —— 块边界随切分参数变化，字符串判据跨切分依然有效。

三种配置（同一份问题集、同一批块）：
    A 纯向量        vectorstore.as_retriever(k=RETRIEVAL_K)
    B 混合           BM25 + 向量，EnsembleRetriever 融合
    C 混合 + Rerank  B 取 RERANK_CANDIDATES 个候选 → CrossEncoder 精排 → TOP_K

单看一个绝对分数没有意义（取决于数据集难度），有价值的是**相对比较**：
C 比 B 高多少，就是 Rerank 的收益；B 比 A 高多少，就是混合检索的收益。

用法
----
    python scripts/eval_retrieval.py                        # 打表
    python scripts/eval_retrieval.py --save tests/x.json    # 存基线
    python scripts/eval_retrieval.py --compare tests/x.json # 回归对比
    python scripts/eval_retrieval.py --only B,C             # 只跑部分配置
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CHUNK_SIZE,
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    MANIFEST_PATH,
    RERANK_CANDIDATES,
    RETRIEVAL_K,
    TOP_K,
)

DATASET = Path(__file__).resolve().parent.parent / "tests" / "eval_retrieval_dataset.json"
K_VALUES = (5, 10)


# ---------------------------------------------------------------------------
# 命中判定
# ---------------------------------------------------------------------------
def hit(doc, must: list[str]) -> bool:
    return all(s in doc.page_content for s in must)


def first_hit_rank(docs, must: list[str]) -> int | None:
    for i, d in enumerate(docs, 1):
        if hit(d, must):
            return i
    return None


def evaluate_config(name: str, retrieve, dataset: list[dict]) -> dict:
    """对一个检索函数跑完整数据集，返回指标。"""
    ranks: list[int | None] = []
    cand_counts: list[int] = []
    latencies: list[float] = []

    for item in dataset:
        t0 = time.perf_counter()
        docs = retrieve(item["question"])
        latencies.append(time.perf_counter() - t0)
        ranks.append(first_hit_rank(docs, item["must_contain"]))
        cand_counts.append(len(docs))

    n = len(dataset)
    result = {"config": name, "n": n}
    for k in K_VALUES:
        got = sum(1 for r in ranks if r is not None and r <= k)
        result[f"recall@{k}"] = round(got / n, 4) if n else 0.0
    hit1 = sum(1 for r in ranks if r == 1)
    result["hit@1"] = round(hit1 / n, 4) if n else 0.0
    mrr_num = sum(1 / r for r in ranks if r is not None)
    result["mrr"] = round(mrr_num / n, 4) if n else 0.0
    result["avg_candidates"] = round(sum(cand_counts) / n, 2) if n else 0.0
    result["avg_latency_s"] = round(sum(latencies) / n, 4) if n else 0.0
    result["missed_ids"] = [item["id"] for item, r in zip(dataset, ranks) if r is None]
    return result


# ---------------------------------------------------------------------------
# 构建三种检索器
# ---------------------------------------------------------------------------
def check_index_consistency() -> None:
    """确保持久化向量库与当前切分参数一致。

    否则 BM25 用"现场切出来的块"，向量检索用"库里的旧块"，
    两者块集合不同，配置对比会失真。
    """
    p = Path(MANIFEST_PATH)
    if not p.is_file():
        raise SystemExit(
            "index_manifest.json 不存在：索引状态未知。\n"
            "  请先运行 python run.py（或 kb rebuild）建立索引。"
        )
    m = json.loads(p.read_text(encoding="utf-8"))
    params = m.get("params", {})
    problems = []
    if params.get("chunk_size") != CHUNK_SIZE:
        problems.append(f"chunk_size: 索引 {params.get('chunk_size')} ≠ 当前 {CHUNK_SIZE}")
    if params.get("embedding_model") != EMBEDDING_MODEL:
        problems.append(f"embedding_model: 索引 {params.get('embedding_model')} ≠ 当前 {EMBEDDING_MODEL}")
    if (params.get("embedding_backend") or "api") != (EMBEDDING_BACKEND or "api"):
        problems.append(
            f"embedding_backend: 索引 {params.get('embedding_backend')} ≠ 当前 {EMBEDDING_BACKEND}"
        )
    if problems:
        raise SystemExit(
            "索引与当前配置不一致，评估结果会失真：\n  - " + "\n  - ".join(problems) +
            "\n  请先重建索引：python run.py（或 kb rebuild --force）"
        )


def build_retrievers(which: set[str]) -> dict[str, callable]:
    from src.loader import load_documents, split_documents
    from src.model import get_reranker
    from src.rag_chain import build_hybrid_retriever
    from src.vectorstore import load_vectorstore

    check_index_consistency()

    vectorstore = load_vectorstore()
    docs = load_documents()
    chunks = split_documents(docs)
    print(f"块数 {len(chunks)}（与索引一致由 check_index_consistency 保证）")

    retrievers: dict[str, callable] = {}

    if "A" in which:
        vr = vectorstore.as_retriever(search_kwargs={"k": max(K_VALUES)})
        retrievers["A 纯向量"] = lambda q, _r=vr: _r.invoke(q)

    if "B" in which or "C" in which:
        hybrid = build_hybrid_retriever(vectorstore, chunks)

    if "B" in which:
        retrievers["B 混合"] = lambda q, _r=hybrid: _r.invoke(q)

    if "C" in which:
        reranker = get_reranker()

        def _hybrid_rerank(q: str):
            cands = hybrid.invoke(q)[:RERANK_CANDIDATES]
            if not cands:
                return []
            scores = reranker.predict([[q, d.page_content] for d in cands])
            ranked = sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)
            # 返回全部候选而不是只取 TOP_K：评估要衡量"排序质量"，
            # 生产链路再截断到 TOP_K=8 是"喂给模型多少上下文"的取舍，
            # 截断会让 Recall@10 天然吃亏、配置之间不可比。
            return [d for d, _ in ranked]

        retrievers["C 混合+Rerank"] = _hybrid_rerank

    return retrievers


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def print_table(results: list[dict]) -> None:
    print()
    print("=" * 78)
    print("检索层评估结果")
    print("=" * 78)
    header = f"{'配置':<16}{'Recall@5':>10}{'Recall@10':>11}{'MRR':>8}{'Hit@1':>8}{'耗时':>9}"
    print(header)
    print("-" * 78)
    for r in results:
        print(
            f"{r['config']:<16}"
            f"{r['recall@5']:>10.3f}"
            f"{r['recall@10']:>11.3f}"
            f"{r['mrr']:>8.3f}"
            f"{r['hit@1']:>8.3f}"
            f"{r['avg_latency_s']:>8.3f}s"
        )
    print("-" * 78)

    # 相对收益：这才是评估的重点，绝对分数受数据集难度影响
    if len(results) >= 2:
        print()
        print("相对收益（同一数据集）")
        base = results[0]
        for r in results[1:]:
            d5 = (r["recall@5"] - base["recall@5"]) * 100
            dm = (r["mrr"] - base["mrr"]) * 100
            print(f"  {r['config']} vs {base['config']}: "
                  f"Recall@5 {d5:+.1f}pp，MRR {dm:+.1f}pp")

    for r in results:
        if r["missed_ids"]:
            print(f"\n{r['config']} 未命中的题目: {', '.join(r['missed_ids'][:12])}"
                  f"{' …' if len(r['missed_ids']) > 12 else ''}")


def compare(current: list[dict], baseline: dict) -> int:
    from src.config import CHUNK_SIZE as CS, EMBEDDING_MODEL as EM

    bp = baseline.get("params", {})
    if bp.get("chunk_size") != CS or bp.get("embedding_model") != EM:
        print("=" * 78)
        print("拒绝比较：切分参数或嵌入模型不一致")
        print("=" * 78)
        print(f"  基线  chunk_size={bp.get('chunk_size')}  embedding={bp.get('embedding_model')}")
        print(f"  当前  chunk_size={CS}  embedding={EM}")
        print("  块边界/向量空间已改变，指标差值不可比较。请重新取基线。")
        return 2

    base_by_cfg = {r["config"]: r for r in baseline.get("results", [])}
    print()
    print("=" * 78)
    print("与基线对比")
    print("=" * 78)
    for r in current:
        b = base_by_cfg.get(r["config"])
        if not b:
            print(f"  {r['config']}: 基线里没有该配置，跳过")
            continue
        parts = []
        for metric in ("recall@5", "recall@10", "mrr", "hit@1"):
            d = (r[metric] - b[metric]) * 100
            mark = "✅" if d > 0.01 else ("⚠️" if d < -0.01 else "  ")
            parts.append(f"{metric} {d:+.1f}pp{mark}")
        print(f"  {r['config']:<16} " + "  ".join(parts))
    print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="检索层评估（Recall@k / MRR / Hit@1）")
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--save", metavar="PATH", help="保存本次结果作为基线")
    ap.add_argument("--compare", metavar="PATH", help="与基线对比")
    ap.add_argument("--only", default="A,B,C", help="只跑部分配置，如 B,C")
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.is_file():
        print(f"数据集不存在: {ds_path}")
        print("  先运行: python scripts/make_eval_dataset.py")
        return 1
    dataset = json.loads(ds_path.read_text(encoding="utf-8"))
    if not dataset:
        print("数据集为空")
        return 1
    print(f"数据集: {len(dataset)} 条（{ds_path}）")

    which = {s.strip().upper() for s in args.only.split(",") if s.strip()}
    retrievers = build_retrievers(which)

    results = []
    for name, fn in retrievers.items():
        print(f"\n运行 {name} …")
        results.append(evaluate_config(name, fn, dataset))

    print_table(results)

    payload = {
        "params": {
            "chunk_size": CHUNK_SIZE,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_backend": (EMBEDDING_BACKEND or "api").strip().lower(),
            "retrieval_k": RETRIEVAL_K,
            "rerank_candidates": RERANK_CANDIDATES,
            "top_k": TOP_K,
            "n_dataset": len(dataset),
        },
        "results": results,
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
        return compare(results, json.loads(p.read_text(encoding="utf-8")))

    return 0


if __name__ == "__main__":
    sys.exit(main())
