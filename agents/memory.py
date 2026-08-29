"""长期记忆：对话轮次落盘（SQLite conversations 表）。

每轮问答由 graph 的 persist 节点写入；重启后仍可查询历史。
表结构与 quiz 共用 data/cisp_qa.db。
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
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id  TEXT NOT NULL,
    role       TEXT NOT NULL,           -- 'user' | 'assistant'
    content    TEXT NOT NULL,
    intent     TEXT,                    -- 该轮意图（assistant 行记录）
    created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_conv_thread ON conversations(thread_id, id);
"""


def _conn() -> sqlite3.Connection:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_init() -> None:
    global _initialized
    if _initialized:
        return
    with _lock, _conn() as conn:
        conn.executescript(_SCHEMA)
    _initialized = True


def save_round(thread_id: str, question: str, answer: str, intent: str) -> None:
    """一轮对话（用户问题 + 助手回答）两行落盘。"""
    _ensure_init()
    now = time.time()
    with _lock, _conn() as conn:
        conn.executemany(
            "INSERT INTO conversations (thread_id, role, content, intent, created_at) VALUES (?,?,?,?,?)",
            [
                (thread_id, "user", question, None, now),
                (thread_id, "assistant", answer, intent, now),
            ],
        )


def load_history(thread_id: str, limit: int = 50) -> list[dict]:
    """按时间序返回某会话的历史消息（旧的在前）。"""
    _ensure_init()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content, intent, created_at FROM conversations "
            "WHERE thread_id = ? ORDER BY id DESC LIMIT ?",
            (thread_id, limit),
        ).fetchall()
    return [
        {"role": r["role"], "content": r["content"],
         "intent": r["intent"], "created_at": r["created_at"]}
        for r in reversed(rows)
    ]


def list_threads(limit: int = 20) -> list[dict]:
    """历史会话列表（按最近消息时间倒序），供会话恢复/续聊入口用。"""
    _ensure_init()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT thread_id, MAX(created_at) AS last_at, COUNT(*) AS rounds "
            "FROM conversations GROUP BY thread_id ORDER BY last_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {"thread_id": r["thread_id"], "last_at": r["last_at"], "rounds": r["rounds"] // 2}
        for r in rows
    ]
