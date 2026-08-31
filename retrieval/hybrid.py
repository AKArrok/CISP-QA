"""混合检索：dense（向量）+ sparse（BM25）→ RRF 融合 → top-k。

进程内单例：首次调用时加载向量库与 BM25 索引。
"""
from __future__ import annotations

import json
import logging
import os
import threading

import config


def _rrf_fuse(dense: list[dict], sparse: list[dict], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion，按 chunk id 去重合并。"""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for rank, item in enumerate(dense):
        cid = item["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        items[cid] = item
    for rank, item in enumerate(sparse):
        cid = item["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        items.setdefault(cid, item)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [{**items[cid], "rrf_score": s} for cid, s in ranked]


class HybridRetriever:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        from data_ingest.manifest import validate_index
        from retrieval.vector_store import VectorStore
        from retrieval.bm25 import BM25Index

        validate_index()  # 启动自检：数据资产齐全且与清单一致
        if not os.path.exists(config.VECTORS_PATH):
            raise FileNotFoundError(
                f"向量索引不存在: {config.VECTORS_PATH}，请先运行 python data_ingest/build_index.py"
            )
        self._vector = VectorStore.load()
        # JSON 是向量索引的规范数据源，必须保留父子切片字段并与向量 ID 严格对齐。
        with open(config.KB_CHUNKS_PATH, encoding="utf-8") as fp:
            chunks = json.load(fp)
        self._bm25 = BM25Index.build(chunks)
        self._chunks_by_id = {c["id"]: c for c in chunks}
        logging.info("  检索层就绪: %d 向量块 / %d BM25 文档",
                     self._vector._matrix.shape[0], len(self._chunks_by_id))

    @classmethod
    def get(cls) -> "HybridRetriever":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def retrieve(self, query: str, top_k: int | None = None,
                 domain: str | None = None) -> list[dict]:
        """返回 top-k 知识块（含 id/domain/source/page/text/score）。

        domain 指定时在该知识域内检索（刷题"关联课件"用）。
        """
        from llms import get_embeddings

        top_k = top_k or config.RETRIEVER_K
        # 检索结果 Redis 缓存（同一查询/域过滤组合在 TTL 内直接复用）
        import hashlib
        cache_key = (f"ret:{hashlib.md5(f'{query}|{top_k}|{domain}'.encode()).hexdigest()}")
        from storage import cache
        cached = cache.get_json(cache_key)
        if cached is not None:
            return cached
        query_vec = get_embeddings().embed_query(query)
        dense = self._vector.search(query_vec, config.DENSE_K)
        sparse = self._bm25.search(query, config.SPARSE_K)
        fused = _rrf_fuse(dense, sparse)
        # 统一补全 chunk 元数据（向量库 metadata 只含 id）
        full = []
        for c in fused:
            chunk = self._chunks_by_id[c["id"]]
            full.append({**chunk, "score": c.get("rrf_score", c.get("score", 0.0))})
        if domain:
            full = [c for c in full if c["domain"] == domain]
            # 域内过滤后不足时，用 BM25 域内结果补
            if len(full) < top_k:
                seen = {c["id"] for c in full}
                extra = [c for c in self._bm25.search(query, config.SPARSE_K * 2)
                         if c["domain"] == domain and c["id"] not in seen]
                for c in extra[:top_k - len(full)]:
                    chunk = self._chunks_by_id[c["id"]]
                    full.append({**chunk, "score": c.get("score", 0.0)})
        # 同一 Parent 的多个 Child 只保留召回排名最高者，避免重复上下文挤占 top-k。
        unique_parents = []
        seen_parents = set()
        for chunk in full:
            parent_id = chunk.get("parent_id", chunk["id"])
            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                unique_parents.append(chunk)
        full = unique_parents

        # 精排：用 Child 检索文本评分，最终返回完整 Parent 文本。
        if config.ENABLE_RERANKING:
            from retrieval.reranker import rerank
            full = rerank(query, full[:config.RERANK_TOP_K], top_k)
        results = full[:top_k]
        cache.set_json(cache_key, results, ttl=config.RETRIEVAL_CACHE_TTL)
        return results


if __name__ == "__main__":
    import sys
    retriever = HybridRetriever.get()
    for q in sys.argv[1:] or ["什么是深度防御"]:
        print(f"\n== {q}")
        for r in retriever.retrieve(q, top_k=3):
            print(f"  [{r['domain']}|{r['source']} p{r['page']}] {r['text'][:80]}...")
