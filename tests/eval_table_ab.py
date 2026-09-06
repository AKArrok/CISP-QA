"""表格感知切片 A/B 评测：同一 gold 集、同一查询向量，仅切换 chunk 集。

用法:
    python tests/eval_table_ab.py
A 侧 = 生产 kb_chunks.json + vectors.npz；B 侧 = kb_chunks_tableaware.json，
向量按 Chunk ID 复用 A 未变化部分，新增块真实嵌入并缓存到 ab_vectors_tableaware.npz。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

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


class FixedVectorRetriever:
    """与 TunableRetriever 同链路，但向量库由外部注入（支持 A/B 两套 chunk 集）。"""

    def __init__(self, chunks: list[dict], vectors: np.ndarray):
        self.chunks = {chunk["id"]: chunk for chunk in chunks}
        self.vector = VectorStore(vectors, [{"id": c["id"]} for c in chunks])
        self.bm25 = BM25Index.build(chunks)

    def retrieve(self, query: str, query_vec: list[float], top_k: int) -> list[dict]:
        dense = self.vector.search(query_vec, config.DENSE_K)
        sparse = self.bm25.search(query, config.SPARSE_K)
        full = [
            {**self.chunks[item["id"]], "score": item["rrf_score"]}
            for item in _rrf_fuse(dense, sparse, k=config.RRF_K)
        ]
        unique, seen = [], set()
        for chunk in full:
            parent_id = chunk.get("parent_id", chunk["id"])
            if parent_id not in seen:
                seen.add(parent_id)
                unique.append(chunk)
        if config.ENABLE_FEATURE_RERANKING:
            unique = fold_near_duplicate_pages(rerank_with_features(query, unique))
        return unique[:top_k]


def align_vectors(chunks: list[dict], cache_path: str) -> np.ndarray:
    """按 Chunk ID 对齐向量：复用生产索引，缺失块真实嵌入后整体缓存。"""
    ids = [c["id"] for c in chunks]
    by_id = {}
    production = np.load(config.VECTORS_PATH)
    prod_meta = json.load(open(config.VECTORS_PATH + ".meta.json", encoding="utf-8"))
    for i, meta in enumerate(prod_meta):
        by_id[meta["id"]] = production["vectors"][i]
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached["ids"].tolist() == ids:
            print(f"复用 B 侧向量缓存: {cache_path}")
            return cached["vectors"]

    missing_ids = [cid for cid in ids if cid not in by_id]
    print(f"B 侧 {len(ids)} 块: 复用生产向量 {len(ids) - len(missing_ids)}，新嵌入 {len(missing_ids)}")
    if missing_ids:
        from llms import get_embeddings
        chunk_by_id = {c["id"]: c for c in chunks}
        vectors = get_embeddings().embed_documents([
            c.get("embedding_text", c["text"]) for c in (chunk_by_id[cid] for cid in missing_ids)
        ])
        for cid, vector in zip(missing_ids, vectors, strict=True):
            by_id[cid] = np.asarray(vector, dtype=np.float32)
    matrix = np.asarray([by_id[cid] for cid in ids], dtype=np.float32)
    np.savez_compressed(cache_path, vectors=matrix, ids=np.asarray(ids))
    return matrix


def evaluate_side(
    label: str,
    gold: list[dict],
    retriever: FixedVectorRetriever,
    query_vectors: dict[str, list[float]],
    top_k: int,
) -> dict:
    details = []
    for case in gold:
        retrieval_query = (
            rewrite_query(case["question"]) if config.ENABLE_QUERY_REWRITE else case["question"]
        )
        started = time.perf_counter()
        results = retriever.retrieve(retrieval_query, query_vectors[case["id"]], top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000
        details.append(evaluate_case(
            case, results, resolve_relevant_parent_ids(retriever.chunks.values(), case), elapsed_ms,
        ))
    by_rank = {d["id"]: d for d in details}
    print(f"[{label}] ", json.dumps(aggregate(details), ensure_ascii=False))
    return by_rank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-b", default="data/kb_chunks_tableaware.json")
    parser.add_argument("--vector-cache", default="data/ab_vectors_tableaware.npz")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--report", default="data/table_ab_report.json")
    args = parser.parse_args()

    gold = load_json(GOLD_PATH)
    chunks_a = load_json(config.KB_CHUNKS_PATH)
    chunks_b = load_json(args.chunks_b)
    validate_gold(gold, chunks_a)
    validate_gold(gold, chunks_b)
    print("Gold 校验：A/B 双侧通过")

    vectors_a = align_vectors(chunks_a, "data/ab_vectors_baseline_aligned.npz")
    vectors_b = align_vectors(chunks_b, args.vector_cache)

    from llms import get_embeddings
    questions = [
        rewrite_query(case["question"]) if config.ENABLE_QUERY_REWRITE else case["question"]
        for case in gold
    ]
    vectors = get_embeddings().embed_documents(questions)
    query_vectors = {c["id"]: v for c, v in zip(gold, vectors, strict=True)}

    rank_a = evaluate_side("A 生产切片", gold, FixedVectorRetriever(chunks_a, vectors_a), query_vectors, args.top_k)
    rank_b = evaluate_side("B 表格感知", gold, FixedVectorRetriever(chunks_b, vectors_b), query_vectors, args.top_k)

    # 表格相关子集：B 侧证据页含表格 Chunk 的用例
    table_pages = {
        (c["source"], c["page"]) for c in chunks_b if c.get("element") == "table"
    }
    table_case_ids = [
        case["id"] for case in gold
        if any((ev["source"], ev["page"]) in table_pages for ev in case["evidence"])
    ]
    subset_a = aggregate([rank_a[cid] for cid in table_case_ids]) if table_case_ids else {}
    subset_b = aggregate([rank_b[cid] for cid in table_case_ids]) if table_case_ids else {}

    diffs = [
        {
            "id": case_id,
            "question": next(c["question"] for c in gold if c["id"] == case_id),
            "rank_a": rank_a[case_id]["first_relevant_rank"],
            "rank_b": rank_b[case_id]["first_relevant_rank"],
        }
        for case_id in rank_a
        if rank_a[case_id]["first_relevant_rank"] != rank_b[case_id]["first_relevant_rank"]
    ]
    report = {
        "embedding_model": embedding_label(),
        "overall": {"A_baseline": aggregate(list(rank_a.values())), "B_tableaware": aggregate(list(rank_b.values()))},
        "table_cases": {"ids": table_case_ids, "A_baseline": subset_a, "B_tableaware": subset_b},
        "rank_changes": diffs,
    }
    with open(args.report, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=1)
    print("表格相关用例:", json.dumps(table_case_ids, ensure_ascii=False))
    print("表格子集 A:", json.dumps(subset_a, ensure_ascii=False))
    print("表格子集 B:", json.dumps(subset_b, ensure_ascii=False))
    print("位次变化:", json.dumps(diffs, ensure_ascii=False, indent=1))
    print(f"报告已保存: {args.report}")


if __name__ == "__main__":
    main()
