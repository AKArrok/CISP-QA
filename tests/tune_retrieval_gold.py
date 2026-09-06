"""用知识点级 gold 集对 Dense/BM25/RRF 参数做小规模网格评测。"""
from __future__ import annotations

import argparse
import json
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "tests")

import config
from eval_retrieval_gold import (
    GOLD_PATH,
    aggregate,
    embedding_label,
    evaluate_case,
    load_json,
    resolve_relevant_parent_ids,
    validate_gold,
)
from retrieval.bm25 import BM25Index
from retrieval.hybrid import _rrf_fuse
from retrieval.rank_features import fold_near_duplicate_pages, rerank_with_features
from retrieval.query_rewrite import rewrite_query
from retrieval.vector_store import VectorStore


class TunableRetriever:
    def __init__(self, chunks: list[dict], dense_k: int, sparse_k: int, rrf_k: int):
        self.chunks = {chunk["id"]: chunk for chunk in chunks}
        self.vector = VectorStore.load()
        self.bm25 = BM25Index.build(chunks)
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.rrf_k = rrf_k

    def retrieve(self, query: str, query_vec: list[float], top_k: int) -> list[dict]:
        dense = self.vector.search(query_vec, self.dense_k)
        sparse = self.bm25.search(query, self.sparse_k)
        fused = _rrf_fuse(dense, sparse, k=self.rrf_k)
        full = [
            {**self.chunks[item["id"]], "score": item["rrf_score"]}
            for item in fused
        ]
        unique = []
        seen = set()
        for chunk in full:
            parent_id = chunk.get("parent_id", chunk["id"])
            if parent_id not in seen:
                seen.add(parent_id)
                unique.append(chunk)
        if config.ENABLE_FEATURE_RERANKING:
            unique = fold_near_duplicate_pages(rerank_with_features(query, unique))
        return unique[:top_k]


def evaluate_config(
    gold: list[dict],
    chunks: list[dict],
    query_vectors: dict[str, list[float]],
    dense_k: int,
    sparse_k: int,
    rrf_k: int,
    top_k: int,
) -> dict:
    retriever = TunableRetriever(chunks, dense_k, sparse_k, rrf_k)
    details = []
    for case in gold:
        retrieval_query = rewrite_query(case["question"]) if config.ENABLE_QUERY_REWRITE else case["question"]
        started = time.perf_counter()
        results = retriever.retrieve(retrieval_query, query_vectors[case["id"]], top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000
        details.append(evaluate_case(
            case,
            results,
            resolve_relevant_parent_ids(chunks, case),
            elapsed_ms,
        ))
    return {
        "dense_k": dense_k,
        "sparse_k": sparse_k,
        "rrf_k": rrf_k,
        **aggregate(details),
        "failures_at_5": [item["id"] for item in details if not item["recall_at_5"]],
    }


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=GOLD_PATH)
    parser.add_argument("--chunks", default=config.KB_CHUNKS_PATH)
    parser.add_argument("--dense-ks", default="10,20,40")
    parser.add_argument("--sparse-ks", default="10,20,40")
    parser.add_argument("--rrf-ks", default="20,60")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--report", default="data/retrieval_gold_tuning_report.json")
    args = parser.parse_args()

    gold = load_json(args.gold)
    chunks = load_json(args.chunks)
    validation = validate_gold(gold, chunks)

    from llms import get_embeddings

    questions = [
        rewrite_query(case["question"]) if config.ENABLE_QUERY_REWRITE else case["question"]
        for case in gold
    ]
    vectors = get_embeddings().embed_documents(questions)
    query_vectors = {case["id"]: vector for case, vector in zip(gold, vectors, strict=True)}

    results = []
    for dense_k in parse_ints(args.dense_ks):
        for sparse_k in parse_ints(args.sparse_ks):
            for rrf_k in parse_ints(args.rrf_ks):
                result = evaluate_config(
                    gold, chunks, query_vectors, dense_k, sparse_k, rrf_k, args.top_k
                )
                results.append(result)
                print(json.dumps(result, ensure_ascii=False))

    results.sort(key=lambda item: (
        -item["recall_at_5"],
        -item["recall_at_1"],
        -item["mrr_at_10"],
        item["latency_ms_avg"],
    ))
    report = {
        "validation": validation,
        "embedding_model": embedding_label(),
        "query_rewrite": config.ENABLE_QUERY_REWRITE,
        "feature_reranking": config.ENABLE_FEATURE_RERANKING,
        "top_k": args.top_k,
        "best": results[0],
        "results": results,
    }
    with open(args.report, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    print("BEST:", json.dumps(results[0], ensure_ascii=False))
    print(f"报告已保存: {args.report}")


if __name__ == "__main__":
    main()
