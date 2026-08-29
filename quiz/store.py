"""SQLite 存储：AI 生成题 + 答题记录。真题库在 questions.json（只读），AI 题落库。"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time

import config

_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ai_questions (
    id          TEXT PRIMARY KEY,
    stem        TEXT NOT NULL,
    options     TEXT NOT NULL,   -- JSON {A,B,C,D}
    answer      TEXT NOT NULL,   -- 'A'/'AB'
    analysis    TEXT DEFAULT '',
    domain      TEXT,
    created_at  REAL
);
CREATE TABLE IF NOT EXISTS attempts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id    TEXT NOT NULL,
    question_source TEXT NOT NULL,
    domain         TEXT,
    stem           TEXT,
    choice         TEXT NOT NULL,
    correct        INTEGER NOT NULL,
    answered_at    REAL
);
CREATE INDEX IF NOT EXISTS idx_attempts_domain ON attempts(domain);
"""


def _conn() -> sqlite3.Connection:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _lock, _conn() as conn:
        conn.executescript(_SCHEMA)


# ── AI 生成题 ───────────────────────────────────────────────────────────

def insert_ai_question(stem: str, options: dict, answer: str,
                       analysis: str, domain: str) -> str:
    qid = f"ai_{int(time.time() * 1000)}"
    with _lock, _conn() as conn:
        conn.execute(
            "INSERT INTO ai_questions (id, stem, options, answer, analysis, domain, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (qid, stem, json.dumps(options, ensure_ascii=False), answer, analysis, domain, time.time()),
        )
    return qid


def list_ai_questions() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM ai_questions").fetchall()
    return [
        {
            "id": r["id"], "source": "AI生成", "num": 0, "stem": r["stem"],
            "options": json.loads(r["options"]), "answer": r["answer"],
            "analysis": r["analysis"], "domain": r["domain"],
            "needs_review": False,
        }
        for r in rows
    ]


def existing_ai_stems() -> set[str]:
    with _conn() as conn:
        rows = conn.execute("SELECT stem FROM ai_questions").fetchall()
    return {r["stem"] for r in rows}


# ── 答题记录 ────────────────────────────────────────────────────────────

def record_attempt(question_id: str, question_source: str, domain: str | None,
                   stem: str, choice: str, correct: bool) -> None:
    with _lock, _conn() as conn:
        conn.execute(
            "INSERT INTO attempts (question_id, question_source, domain, stem, choice, correct, answered_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (question_id, question_source, domain, stem[:500], choice, int(correct), time.time()),
        )


def attempted_ids(correct_only: bool = False) -> set[str]:
    sql = "SELECT DISTINCT question_id FROM attempts" + (" WHERE correct = 1" if correct_only else "")
    with _conn() as conn:
        return {r["question_id"] for r in conn.execute(sql)}
