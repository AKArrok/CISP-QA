"""可观测性：请求级指标采集（metrics 表）与聚合查询。"""
from __future__ import annotations

import time

from sqlalchemy import select

from storage import db
from storage.models import Metric


def record(thread_id: str, question: str, intent: str | None,
           route_ms: float | None, retrieval_ms: float | None,
           first_token_ms: float | None, total_ms: float | None,
           prompt_tokens: int | None, completion_tokens: int | None) -> None:
    with db.session() as s:
        s.add(Metric(thread_id=thread_id, question=question[:300], intent=intent,
                     route_ms=route_ms, retrieval_ms=retrieval_ms,
                     first_token_ms=first_token_ms, total_ms=total_ms,
                     prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                     created_at=time.time()))
        s.commit()


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(int(len(values) * p), len(values) - 1)
    return round(values[idx], 1)


def summary(limit: int = 500) -> dict:
    with db.session() as s:
        rows = s.execute(
            select(Metric.intent, Metric.route_ms, Metric.retrieval_ms,
                   Metric.first_token_ms, Metric.total_ms,
                   Metric.prompt_tokens, Metric.completion_tokens)
            .order_by(Metric.id.desc()).limit(limit)
        ).all()
    n = len(rows)
    if n == 0:
        return {"total_requests": 0}
    keys = ["intent", "route_ms", "retrieval_ms", "first_token_ms", "total_ms",
            "prompt_tokens", "completion_tokens"]
    cols = {k: [r[idx] for r in rows if r[idx] is not None]
            for idx, k in enumerate(keys)}
    by_intent: dict[str, int] = {}
    for intent in cols["intent"]:
        by_intent[intent or "unknown"] = by_intent.get(intent or "unknown", 0) + 1
    return {
        "total_requests": n,
        "by_intent": by_intent,
        "avg_total_ms": round(sum(cols["total_ms"]) / max(len(cols["total_ms"]), 1), 1),
        "p95_first_token_ms": _percentile(cols["first_token_ms"], 0.95),
        "p95_total_ms": _percentile(cols["total_ms"], 0.95),
        "avg_route_ms": round(sum(cols["route_ms"]) / max(len(cols["route_ms"]), 1), 1),
        "avg_retrieval_ms": round(sum(cols["retrieval_ms"]) / max(len(cols["retrieval_ms"]), 1), 1),
        "total_prompt_tokens": sum(cols["prompt_tokens"]),
        "total_completion_tokens": sum(cols["completion_tokens"]),
    }
