"""离线重放：精排分数融合参数扫描（不消耗任何 API 额度）。

流程：
1. 复现 v8 评测的候选捕获——monkeypatch 精排入口只捕获不打分，
   捕获的即特征重排后的 Top10（与 --rerank 评测的精排输入完全一致）；
2. 精排分数直接读 data/rerank_cache.json（v8 评测已把全部 1600 对打分落盘），
   发现缓存缺失直接报错，绝不发起 API 请求；
3. 离线扫描融合策略：加权融合 α / 翻序置信阈值 δ，对比纯特征序与纯精排序。

用法：python tests/tune_rerank_fusion.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

sys.path.insert(0, ".")

import config
from tests.eval_retrieval_gold import GOLD_PATH, load_json, resolve_relevant_parent_ids

HOLDOUT_PATH = "tests/fixtures/retrieval_holdout.json"

REPLAY_REPORT = "data/rerank_fusion_tuning.json"

ALPHAS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
DELTAS = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3)


def capture_candidates() -> dict[str, list[dict]]:
    """跑一遍检索管线，捕获每题特征序 Top10（精排入口被替换为只捕获）。"""
    captured: dict[str, list[dict]] = {}
    import retrieval.reranker as reranker_module

    original = reranker_module.rerank
    original_flag = config.ENABLE_RERANKING
    config.ENABLE_RERANKING = True

    def capturing_rerank(query, candidates, top_k):
        captured[query] = list(candidates)
        return candidates[:top_k]

    reranker_module.rerank = capturing_rerank
    try:
        from retrieval.hybrid import HybridRetriever

        retriever = HybridRetriever.get()
        for path in (GOLD_PATH, HOLDOUT_PATH):
            for case in load_json(path):
                retriever.retrieve(case["question"], top_k=10)
    finally:
        reranker_module.rerank = original
        config.ENABLE_RERANKING = original_flag
    return captured


def build_scores_map(captured: dict[str, list[dict]]) -> dict[str, list[float]]:
    from llms import DashScopeReranker

    probe = DashScopeReranker(api_key="unused", base_url="https://example.invalid",
                              model=config.DASHSCOPE_RERANK_MODEL)
    with open(config.RERANK_CACHE_PATH, encoding="utf-8") as fp:
        cache = json.load(fp)
    scores_map: dict[str, list[float]] = {}
    for query, candidates in captured.items():
        texts = [c.get("embedding_text") or c.get("child_text") or c["text"]
                 for c in candidates[:config.RERANK_TOP_K]]
        missing = [t[:40] for t in texts if probe._cache_key(query, t) not in cache]
        if missing:
            raise RuntimeError(f"缓存缺失 {len(missing)} 对，拒绝发起 API 请求: {query[:30]} {missing}")
        scores_map[query] = [cache[probe._cache_key(query, t)] for t in texts]
    return scores_map


def order_feature(candidates: list[dict]) -> list[int]:
    return list(range(len(candidates)))


def order_rerank(scores: list[float], n: int) -> list[int]:
    return sorted(range(n), key=lambda i: (-scores[i], i))


def order_alpha(scores: list[float], n: int, alpha: float) -> list[int]:
    """加权融合：α·精排分(集合内 min-max 归一) + (1-α)·特征名次(线性归一)。"""
    r_min, r_max = min(scores), max(scores)
    span = (r_max - r_min) or 1.0
    fused = []
    for i, s in enumerate(scores):
        r_norm = (s - r_min) / span
        f_norm = (n - i) / max(n - 1, 1)
        fused.append(-(alpha * r_norm + (1 - alpha) * f_norm))
    return sorted(range(n), key=lambda i: (fused[i], i))


def order_margin(scores: list[float], n: int, delta: float) -> list[int]:
    """精排序为基底，分差小于 δ 的同桶内保持特征序（小分差不许翻序）。"""
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(scores):
        buckets[round(s / delta)].append(i)
    ordered: list[int] = []
    for b in sorted(buckets, reverse=True):
        ordered.extend(sorted(buckets[b]))
    return ordered


def evaluate(order_builder, cases, scores_map, relevant_map) -> dict:
    r1, r5, rr = [], [], []
    for case in cases:
        question = case["question"]
        n = config.RERANK_TOP_K
        rel = relevant_map[question]
        order = order_builder(scores_map[question], n)
        rank = next((i + 1 for i, cid in enumerate(order)
                     if _captured_candidates(question)[cid]
                     .get("parent_id", _captured_candidates(question)[cid]["id"]) in rel), None)
        r1.append(1.0 if rank == 1 else 0.0)
        r5.append(1.0 if rank and rank <= 5 else 0.0)
        rr.append(1.0 / rank if rank else 0.0)
    return {"recall_at_1": round(sum(r1) / len(r1), 4),
            "recall_at_5": round(sum(r5) / len(r5), 4),
            "mrr_at_10": round(sum(rr) / len(rr), 4)}


def main() -> None:
    gold = load_json(GOLD_PATH)
    holdout = load_json(HOLDOUT_PATH)
    chunks = load_json(config.KB_CHUNKS_PATH)

    print("捕获候选（检索向量全部走缓存，零 API 消耗）...")
    global _CAPTURED
    _CAPTURED = capture_candidates()
    print(f"捕获 {len(_CAPTURED)} 题")
    scores_map = build_scores_map(_CAPTURED)
    print("精排分数全部来自磁盘缓存 ✓")

    relevant_map = {
        case["question"]: resolve_relevant_parent_ids(chunks, case)
        for case in gold + holdout
    }

    results: dict = {
        "captured_queries": len(_CAPTURED),
        "rerank_top_k": config.RERANK_TOP_K,
        "holdout": {},
        "gold": {},
    }
    for dataset, cases in (("holdout", holdout), ("gold", gold)):
        rows = results[dataset]
        rows["feature_only"] = evaluate(lambda s, n: list(range(n)), cases, scores_map, relevant_map)
        rows["rerank_only"] = evaluate(order_rerank, cases, scores_map, relevant_map)
        for alpha in ALPHAS:
            rows[f"alpha_{alpha}"] = evaluate(
                lambda s, n, a=alpha: order_alpha(s, n, a), cases, scores_map, relevant_map)
        for delta in DELTAS:
            rows[f"margin_{delta}"] = evaluate(
                lambda s, n, d=delta: order_margin(s, n, d), cases, scores_map, relevant_map)

    with open(REPLAY_REPORT, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)

    header = f"{'策略':<16} {'holdout R1':>10} {'holdout R5':>10} {'holdout MRR':>11} {'gold R1':>9} {'gold R5':>9}"
    print(header)
    print("-" * len(header))
    strategies = (["feature_only", "rerank_only"]
                  + [f"alpha_{a}" for a in ALPHAS]
                  + [f"margin_{d}" for d in DELTAS])
    for name in strategies:
        h, g = results["holdout"][name], results["gold"][name]
        print(f"{name:<16} {h['recall_at_1']:>10.4f} {h['recall_at_5']:>10.4f} "
              f"{h['mrr_at_10']:>11.4f} {g['recall_at_1']:>9.4f} {g['recall_at_5']:>9.4f}")
    print(f"\n报告: {REPLAY_REPORT}")


_CAPTURED: dict[str, list[dict]] = {}


def _captured_candidates(question: str) -> list[dict]:
    return _CAPTURED[question][:config.RERANK_TOP_K]


if __name__ == "__main__":
    main()
