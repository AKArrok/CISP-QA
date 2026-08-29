"""可观测性：请求级指标采集（SQLite metrics 表）与聚合查询。

每次 SSE 问答由 server 写入一行：意图、各阶段耗时、Token 用量。
/api/metrics 提供聚合（均值/分位数/按意图分布），供面板与调试。
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time

import config

_lock = threading.Lock()
_initialized = False

_SCHEMA = """
CREATE TABLE IF NOT EXISTS metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id       TEXT,
    question        TEXT,
    intent          TEXT,
    route_ms        REAL,
    retrieval_ms    REAL,
    first_token_ms  REAL,
    total_ms        REAL,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    created_at      REAL
);
"""


def _conn() -> sqlite3.Connection:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    global _initialized
    if _initialized:
        return
    with _lock, _conn() as conn:
        conn.executescript(_SCHEMA)
    _initialized = True


def record(thread_id: str, question: str, intent: str | None,
           route_ms: float | None, retrieval_ms: float | None,
           first_token_ms: float | None, total_ms: float | None,
           prompt_tokens: int | None, completion_tokens: int | None) -> None:
    init_db()
    with _lock, _conn() as conn:
        conn.execute(
            "INSERT INTO metrics (thread_id, question, intent, route_ms, retrieval_ms, "
            "first_token_ms, total_ms, prompt_tokens, completion_tokens, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (thread_id, question[:300], intent, route_ms, retrieval_ms,
             first_token_ms, total_ms, prompt_tokens, completion_tokens, time.time()),
        )


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(int(len(values) * p), len(values) - 1)
    return round(values[idx], 1)


def summary(limit: int = 500) -> dict:
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT intent, route_ms, retrieval_ms, first_token_ms, total_ms, "
            "prompt_tokens, completion_tokens FROM metrics "
            "ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    n = len(rows)
    if n == 0:
        return {"total_requests": 0}
    col = lambda name: [r[name] for r in rows if r[name] is not None]  # noqa: E731
    by_intent: dict[str, int] = {}
    for r in rows:
        by_intent[r["intent"] or "unknown"] = by_intent.get(r["intent"] or "unknown", 0) + 1
    return {
        "total_requests": n,
        "by_intent": by_intent,
        "avg_total_ms": round(sum(col("total_ms")) / max(len(col("total_ms")), 1), 1),
        "p95_first_token_ms": _percentile(col("first_token_ms"), 0.95),
        "p95_total_ms": _percentile(col("total_ms"), 0.95),
        "avg_route_ms": round(sum(col("route_ms")) / max(len(col("route_ms")), 1), 1),
        "avg_retrieval_ms": round(sum(col("retrieval_ms")) / max(len(col("retrieval_ms")), 1), 1),
        "total_prompt_tokens": sum(col("prompt_tokens")),
        "total_completion_tokens": sum(col("completion_tokens")),
    }
