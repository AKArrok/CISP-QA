"""三方案切片对比：legacy（按页）/ fixed（定长+字符重叠）/ structured（结构化父子）。

用法（推荐，全量真实 embedding）:
    python tests/compare_chunking.py --full-embeddings
默认 proxy 模式复用同页 legacy 向量，只隔离切片结构差异，不调用 embedding API。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from statistics import mean, median

import numpy as np

sys.path.insert(0, ".")

import config
from retrieval.bm25 import BM25Index
from retrieval.hybrid import _rrf_fuse
from retrieval.reranker import rerank
from retrieval.vector_store import VectorStore
from eval_retrieval import EVAL_SET


def load_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def percentile(values: list[int], ratio: float) -> int:
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * ratio)]


def structure_metrics(chunks: list[dict]) -> dict:
    child_lengths = [len(c.get("child_text", c["text"])) for c in chunks]
    parent_lengths = [len(c["text"]) for c in chunks]
    parent_ids = {c.get("parent_id", c["id"]) for c in chunks}
    normalized = [re.sub(r"\W+", "", c.get("child_text", c["text"]).lower()) for c in chunks]
    duplicate_rate = 1 - len(set(normalized)) / max(len(normalized), 1)
    return {
        "chunks": len(chunks),
        "parents": len(parent_ids),
        "children_per_parent": round(len(chunks) / max(len(parent_ids), 1), 2),
        "child_avg": round(mean(child_lengths), 1),
        "child_p50": median(child_lengths),
        "child_p90": percentile(child_lengths, 0.9),
        "child_max": max(child_lengths),
        "parent_avg": round(mean(parent_lengths), 1),
        "title_coverage": round(
            sum(bool(c.get("title")) for c in chunks) / len(chunks) * 100, 1
        ),
        "exact_duplicate_rate": round(duplicate_rate * 100, 2),
    }


def load_or_embed(chunks: list[dict], cache_path: str, force_embed: bool) -> np.ndarray:
    ids = [c["id"] for c in chunks]
    if not force_embed and os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached["ids"].tolist() == ids:
            print(f"复用向量缓存: {cache_path}")
            return cached["vectors"]
    from llms import get_embeddings
    print(f"生成向量: {len(chunks)} 块")
    vectors = np.asarray(get_embeddings().embed_documents([
        c.get("embedding_text", c["text"]) for c in chunks
    ]), dtype=np.float32)
    np.savez_compressed(cache_path, vectors=vectors, ids=np.asarray(ids))
    return vectors


def proxy_vectors(
    source_chunks: list[dict], source_vectors: np.ndarray, target_chunks: list[dict]
) -> np.ndarray:
    """复用同源同页向量，隔离比较切片结构对 BM25/去重/排序的收益。"""
    by_page: dict[tuple, np.ndarray] = {}
    for chunk, vector in zip(source_chunks, source_vectors, strict=True):
        by_page.setdefault((chunk["domain"], chunk["source"], chunk["page"]), vector)
    missing = [
        chunk for chunk in target_chunks
        if (chunk["domain"], chunk["source"], chunk["page"]) not in by_page
    ]
    if missing:
        raise RuntimeError(f"有 {len(missing)} 个块找不到同页向量（proxy 模式）")
    return np.asarray([
        by_page[(chunk["domain"], chunk["source"], chunk["page"])]
        for chunk in target_chunks
    ], dtype=np.float32)


class InMemoryRetriever:
    def __init__(self, chunks: list[dict], vectors: np.ndarray, use_reranker: bool):
        self.chunks = {c["id"]: c for c in chunks}
        self.vector = VectorStore(vectors, [{"id": c["id"]} for c in chunks])
        self.bm25 = BM25Index.build(chunks)
        self.use_reranker = use_reranker

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        from llms import get_embeddings
        dense = self.vector.search(get_embeddings().embed_query(query), config.DENSE_K)
        sparse = self.bm25.search(query, config.SPARSE_K)
        full = [
            {**self.chunks[item["id"]], "score": item["rrf_score"]}
            for item in _rrf_fuse(dense, sparse)
        ]
        unique = []
        seen = set()
        for chunk in full:
            parent_id = chunk.get("parent_id", chunk["id"])
            if parent_id not in seen:
                seen.add(parent_id)
                unique.append(chunk)
        if self.use_reranker:
            unique = rerank(query, unique[:config.RERANK_TOP_K], top_k)
        return unique[:top_k]


def evaluate(name: str, retriever: InMemoryRetriever) -> dict:
    """领域级 Recall@5 / MRR@10 + 上下文重复率 + 输入长度与延迟（检索侧）。"""
    top1 = top5 = 0
    mrr_sum = 0.0
    context_chars: list[int] = []
    dup_rates: list[float] = []
    latencies: list[float] = []
    misses = []
    for question, expected in EVAL_SET:
        t0 = time.perf_counter()
        results = retriever.retrieve(question, top_k=10)
        latencies.append((time.perf_counter() - t0) * 1000)
        domains = [r["domain"] for r in results]
        top1 += bool(domains) and domains[0] == expected
        top5 += expected in domains[:5]
        rank = domains[:10].index(expected) + 1 if expected in domains[:10] else 0
        mrr_sum += 1.0 / rank if rank else 0.0
        context_chars.append(sum(len(r["text"]) for r in results[:5]))
        norm = [
            re.sub(r"\W+", "", r.get("child_text", r["text"]).lower())
            for r in results[:5]
        ]
        dup_rates.append((len(norm) - len(set(norm))) / max(len(norm), 1))
        if expected not in domains[:5]:
            misses.append((question, expected, domains[:5]))

    n = len(EVAL_SET)
    result = {
        "total": n,
        "recall_top1": round(top1 / n * 100, 1),
        "recall_top5": round(top5 / n * 100, 1),
        "mrr10": round(mrr_sum / n, 4),
        "top5_dup_rate_pct": round(mean(dup_rates) * 100, 2),
        "avg_context_chars": round(mean(context_chars), 1),
        "avg_latency_ms": round(mean(latencies), 1),
    }
    print(f"{name}: {json.dumps(result, ensure_ascii=False)}")
    for question, expected, domains in misses:
        print(f"  MISS [{expected}] {question} → {domains}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", default="data/kb_chunks_legacy.json")
    parser.add_argument("--fixed", default="data/kb_chunks_fixed.json")
    parser.add_argument("--structured", default="data/kb_chunks_structured.json")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument(
        "--full-embeddings", action="store_true",
        help="三方案全部用当前 embedding 模型真实嵌入（默认 proxy 复用同页向量）",
    )
    parser.add_argument("--report", default="data/chunking_abc_report.json")
    args = parser.parse_args()

    scheme_files = {
        "legacy": args.legacy,
        "fixed": args.fixed,
        "structured": args.structured,
    }
    print("结构指标")
    loaded: dict[str, list[dict]] = {}
    for name, path in scheme_files.items():
        loaded[name] = load_json(path)
        print(f"[{name}]", json.dumps(structure_metrics(loaded[name]), ensure_ascii=False))

    legacy_vectors = load_or_embed(
        loaded["legacy"], "data/ab_vectors_legacy.npz", args.full_embeddings
    )
    print(f"\n检索指标（reranker={'on' if args.rerank else 'off'}）")
    results: dict[str, dict] = {}
    for name, chunks in loaded.items():
        if args.full_embeddings or name == "legacy":
            vectors = load_or_embed(
                chunks, f"data/ab_vectors_{name}.npz", args.full_embeddings
            )
            label = name if name == "legacy" else f"{name}(真实 embedding)"
        else:
            vectors = proxy_vectors(loaded["legacy"], legacy_vectors, chunks)
            label = f"{name}(同页向量代理)"
        results[name] = {
            "structure": structure_metrics(chunks),
            "retrieval": evaluate(label, InMemoryRetriever(chunks, vectors, args.rerank)),
        }

    report = {
        "embeddings": "full_dashscope" if args.full_embeddings else "same_page_legacy_proxy",
        "reranker": args.rerank,
        "eval_set": len(EVAL_SET),
        **results,
    }
    with open(args.report, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    print(f"报告已保存: {args.report}")


if __name__ == "__main__":
    main()
