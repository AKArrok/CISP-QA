"""知识点级检索评测：以课件来源、页码和证据词作为 gold。"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from statistics import mean

sys.path.insert(0, ".")

import config
from domains import DOMAINS

GOLD_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "retrieval_gold.json")
QUERY_TYPES = {"fact", "definition", "acronym", "comparison", "process", "scenario", "mechanism"}


def load_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def embedding_label() -> str:
    if config.EMBEDDING_BACKEND == "dashscope":
        return f"dashscope:{config.DASHSCOPE_EMBEDDING_MODEL}:{config.DASHSCOPE_EMBEDDING_DIMENSIONS}"
    if config.EMBEDDING_BACKEND == "ark":
        return f"ark:{config.ARK_EMBEDDING_MODEL}:{config.ARK_EMBEDDING_DIMENSIONS}"
    return f"local:{config.LOCAL_EMBEDDING_MODEL}"


def matches_evidence(chunk: dict, evidence: dict, require_terms: bool = True) -> bool:
    if chunk.get("source") != evidence["source"] or chunk.get("page") != evidence["page"]:
        return False
    if not require_terms:
        return True
    text = re.sub(r"\s+", "", chunk.get("text", "")).lower()
    return all(re.sub(r"\s+", "", term).lower() in text for term in evidence["must_contain"])


def resolve_relevant_parent_ids(chunks: list[dict], case: dict) -> set[str]:
    return {
        chunk.get("parent_id", chunk["id"])
        for chunk in chunks
        for evidence in case["evidence"]
        if matches_evidence(chunk, evidence)
    }


def validate_gold(gold: list[dict], chunks: list[dict]) -> dict:
    errors: list[str] = []
    ids = [case.get("id") for case in gold]
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"重复 case id: {duplicates}")

    domain_counts = Counter()
    type_counts = Counter()
    resolved_counts = []
    for case in gold:
        case_id = case.get("id", "<missing>")
        if not case.get("question", "").strip():
            errors.append(f"{case_id}: question 为空")
        domain = case.get("primary_domain")
        if domain not in DOMAINS:
            errors.append(f"{case_id}: 非法知识域 {domain!r}")
        else:
            domain_counts[domain] += 1
        query_type = case.get("query_type")
        if query_type not in QUERY_TYPES:
            errors.append(f"{case_id}: 非法 query_type {query_type!r}")
        else:
            type_counts[query_type] += 1
        if not case.get("evidence"):
            errors.append(f"{case_id}: evidence 为空")
            continue
        relevant = resolve_relevant_parent_ids(chunks, case)
        resolved_counts.append(len(relevant))
        if not relevant:
            evidence_desc = [
                f"{item['source']} p{item['page']} terms={item['must_contain']}"
                for item in case["evidence"]
            ]
            errors.append(f"{case_id}: gold 证据未匹配知识库: {evidence_desc}")

    missing_domains = sorted(set(DOMAINS) - set(domain_counts))
    if missing_domains:
        errors.append(f"未覆盖知识域: {missing_domains}")
    if errors:
        raise AssertionError("Gold 数据集校验失败:\n  " + "\n  ".join(errors))
    return {
        "cases": len(gold),
        "domains": dict(sorted(domain_counts.items())),
        "query_types": dict(sorted(type_counts.items())),
        "resolved_parent_avg": round(mean(resolved_counts), 2),
    }


def _dcg(relevance: list[int]) -> float:
    return sum(value / math.log2(rank + 2) for rank, value in enumerate(relevance))


def _keyword_coverage(results: list[dict], case: dict) -> float:
    best = 0.0
    for result in results:
        text = re.sub(r"\s+", "", result.get("text", "")).lower()
        for evidence in case["evidence"]:
            terms = evidence["must_contain"]
            best = max(best, sum(
                re.sub(r"\s+", "", term).lower() in text for term in terms
            ) / len(terms))
    return best


def evaluate_case(case: dict, results: list[dict], relevant_ids: set[str], elapsed_ms: float) -> dict:
    result_parent_ids = [item.get("parent_id", item["id"]) for item in results]
    relevance = [int(parent_id in relevant_ids) for parent_id in result_parent_ids]
    first_rank = next((index + 1 for index, value in enumerate(relevance) if value), 0)
    ideal = [1] * min(len(relevant_ids), len(results))
    idcg = _dcg(ideal)
    return {
        "id": case["id"],
        "question": case["question"],
        "primary_domain": case["primary_domain"],
        "query_type": case["query_type"],
        "first_relevant_rank": first_rank,
        "recall_at_1": int(any(relevance[:1])),
        "recall_at_3": int(any(relevance[:3])),
        "recall_at_5": int(any(relevance[:5])),
        "recall_at_10": int(any(relevance[:10])),
        "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
        "ndcg_at_10": _dcg(relevance[:10]) / idcg if idcg else 0.0,
        "keyword_coverage_at_5": _keyword_coverage(results[:5], case),
        "latency_ms": elapsed_ms,
        "retrieved": [
            {
                "rank": index + 1,
                "relevant": bool(relevance[index]),
                "domain": item["domain"],
                "source": item["source"],
                "page": item["page"],
                "title": item.get("title", ""),
            }
            for index, item in enumerate(results)
        ],
    }


def aggregate(items: list[dict]) -> dict:
    return {
        "count": len(items),
        "recall_at_1": round(mean(item["recall_at_1"] for item in items), 4),
        "recall_at_3": round(mean(item["recall_at_3"] for item in items), 4),
        "recall_at_5": round(mean(item["recall_at_5"] for item in items), 4),
        "recall_at_10": round(mean(item["recall_at_10"] for item in items), 4),
        "mrr_at_10": round(mean(item["reciprocal_rank"] for item in items), 4),
        "ndcg_at_10": round(mean(item["ndcg_at_10"] for item in items), 4),
        "keyword_coverage_at_5": round(mean(item["keyword_coverage_at_5"] for item in items), 4),
        "latency_ms_avg": round(mean(item["latency_ms"] for item in items), 1),
    }


def run_evaluation(gold: list[dict], chunks: list[dict], top_k: int) -> dict:
    from retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever.get()
    details = []
    for index, case in enumerate(gold, start=1):
        relevant_ids = resolve_relevant_parent_ids(chunks, case)
        started = time.perf_counter()
        results = retriever.retrieve(case["question"], top_k=top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000
        detail = evaluate_case(case, results, relevant_ids, elapsed_ms)
        details.append(detail)
        mark = "PASS" if detail["recall_at_5"] else "MISS"
        print(f"[{index:02d}/{len(gold)}] {mark} R={detail['first_relevant_rank'] or '-'} {case['id']}")

    groups: dict[str, dict[str, list[dict]]] = {
        "by_domain": defaultdict(list),
        "by_query_type": defaultdict(list),
    }
    for detail in details:
        groups["by_domain"][detail["primary_domain"]].append(detail)
        groups["by_query_type"][detail["query_type"]].append(detail)
    return {
        "embedding_model": embedding_label(),
        "dense_degraded": bool(getattr(retriever, "_dense_degraded", False)),
        "query_rewrite": config.ENABLE_QUERY_REWRITE,
        "feature_reranking": config.ENABLE_FEATURE_RERANKING,
        "reranker": config.ENABLE_RERANKING,
        "rerank_backend": config.RERANK_BACKEND if config.ENABLE_RERANKING else "off",
        "overall": aggregate(details),
        "by_domain": {
            key: aggregate(value) for key, value in sorted(groups["by_domain"].items())
        },
        "by_query_type": {
            key: aggregate(value) for key, value in sorted(groups["by_query_type"].items())
        },
        "failures_at_5": [item for item in details if not item["recall_at_5"]],
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=GOLD_PATH)
    parser.add_argument("--chunks", default=config.KB_CHUNKS_PATH)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", default="data/retrieval_gold_report.json")
    args = parser.parse_args()

    gold = load_json(args.gold)
    chunks = load_json(args.chunks)
    validation = validate_gold(gold, chunks)
    print("Gold 校验通过:", json.dumps(validation, ensure_ascii=False))
    if args.validate_only:
        return
    if args.rerank and args.no_rerank:
        raise SystemExit("--rerank 与 --no-rerank 不能同时使用")
    if args.rerank:
        config.ENABLE_RERANKING = True
    elif args.no_rerank:
        config.ENABLE_RERANKING = False
    report = {"validation": validation, **run_evaluation(gold, chunks, args.top_k)}
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    print("总指标:", json.dumps(report["overall"], ensure_ascii=False))
    print(f"失败数: {len(report['failures_at_5'])}; 报告: {args.output}")


if __name__ == "__main__":
    main()
