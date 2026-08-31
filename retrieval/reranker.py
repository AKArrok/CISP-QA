"""精排：bge-reranker-v2-m3 交叉编码器，对混合检索的候选重排序。

CPU 上对 10-20 个候选约需数秒，调用方应放在线程池（hybrid 检索已被 to_thread 包裹）。
模型下载失败/加载失败时优雅降级：跳过精排，保持 RRF 顺序。
"""
from __future__ import annotations

import logging
import threading

import config

_lock = threading.Lock()
_predict_gate = threading.Semaphore(1)  # CPU 重排序串行化，避免并发互相拖垮
_cross_encoder = None
_failed = False


def _get_encoder():
    global _cross_encoder, _failed
    if _failed:
        return None
    if _cross_encoder is None:
        with _lock:
            if _cross_encoder is None and not _failed:
                try:
                    from sentence_transformers import CrossEncoder
                    logging.info("加载精排模型 %s ...", config.LOCAL_RERANKER_MODEL)
                    _cross_encoder = CrossEncoder(config.LOCAL_RERANKER_MODEL, max_length=512)
                except Exception:
                    logging.exception("精排模型加载失败，已降级为 RRF 顺序（不影响可用性）")
                    _failed = True
                    return None
    return _cross_encoder


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """对候选拿 query 相关性重排序，返回 top-k（失败时原序截断）。"""
    if not candidates:
        return []
    encoder = _get_encoder()
    if encoder is None or len(candidates) == 1:
        return candidates[:top_k]
    try:
        pairs = [[
            query,
            c.get("embedding_text") or c.get("child_text") or c["text"],
        ] for c in candidates]
        with _predict_gate:
            scores = encoder.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: -x[1])
        return [{**c, "rerank_score": float(s)} for c, s in ranked[:top_k]]
    except Exception:
        logging.exception("精排失败，降级为 RRF 顺序")
        return candidates[:top_k]
