"""仓库层：题库/知识块的读写与计数（ORM 唯一出口，供 selector/manifest/import 使用）。"""
from __future__ import annotations

import json
import logging
import time

from sqlalchemy import func, select

from storage import db
from storage.models import Attempt, KBChunk, Question, QuestionSignal

logger = logging.getLogger(__name__)


# ── 题库 ────────────────────────────────────────────────────────────────

def count_questions(exclude_ai: bool = False) -> int:
    with db.session() as s:
        stmt = select(func.count()).select_from(Question)
        if exclude_ai:
            stmt = stmt.where(Question.source != "AI生成")
        return s.scalar(stmt) or 0


def upsert_questions(items: list[dict], batch_size: int = 200) -> int:
    """按 id upsert 批量写入题库，返回写入条数。"""
    n = 0
    seen: set[str] = set()
    items = [q for q in items if not (q["id"] in seen or seen.add(q["id"]))]
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        with db.session() as s:
            ids = [q["id"] for q in batch]
            existing = set(s.scalars(
                select(Question.id).where(Question.id.in_(ids))))
            for q in batch:
                if q["id"] in existing:
                    continue
                s.add(Question(
                    id=q["id"], source=q["source"], num=q.get("num", 0),
                    stem=q["stem"], options=json.dumps(q["options"], ensure_ascii=False),
                    answer=q.get("answer", ""), analysis=q.get("analysis", ""),
                    domain=q.get("domain"), needs_review=bool(q.get("needs_review")),
                    dup_of=q.get("dup_of"),
                ))
            s.commit()
        n += len(batch)
    return n


def load_questions() -> list[dict]:
    """题库全量加载（QuestionBank 用）。表为空返回 []，调用方决定是否回退 JSON。"""
    out = []
    with db.session() as s:
        rows = s.scalars(select(Question)).all()
    for r in rows:
        out.append({
            "id": r.id, "source": r.source, "num": r.num, "stem": r.stem,
            "options": json.loads(r.options), "answer": r.answer,
            "analysis": r.analysis, "domain": r.domain,
            "needs_review": bool(r.needs_review),
            **({"dup_of": r.dup_of} if r.dup_of else {}),
        })
    return out


# ── 知识块 ──────────────────────────────────────────────────────────────

def count_chunks() -> int:
    with db.session() as s:
        return s.scalar(select(func.count()).select_from(KBChunk)) or 0


def upsert_chunks(items: list[dict], batch_size: int = 200) -> int:
    n = 0
    seen: set[str] = set()
    items = [q for q in items if not (q["id"] in seen or seen.add(q["id"]))]
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        with db.session() as s:
            ids = [c["id"] for c in batch]
            existing = set(s.scalars(select(KBChunk.id).where(KBChunk.id.in_(ids))))
            for c in batch:
                if c["id"] in existing:
                    continue
                s.add(KBChunk(id=c["id"], domain=c["domain"], source=c["source"],
                              page=c["page"], kind=c["kind"], text=c["text"]))
            s.commit()
        n += len(batch)
    return n


def load_chunks() -> list[dict]:
    out = []
    with db.session() as s:
        rows = s.scalars(select(KBChunk)).all()
    for r in rows:
        out.append({"id": r.id, "domain": r.domain, "source": r.source,
                    "page": r.page, "kind": r.kind, "text": r.text})
    return out


# ── 答题记录（画像统计用）────────────────────────────────────────────────

def domain_accuracy_rows() -> list[tuple[str, int, int]]:
    """[(知识域, 答题数, 正确数)]。"""
    from sqlalchemy import case
    with db.session() as s:
        rows = s.execute(
            select(Attempt.domain,
                   func.count().label("n"),
                   func.sum(case((Attempt.correct.is_(True), 1), else_=0)).label("c"))
            .where(Attempt.domain.is_not(None))
            .group_by(Attempt.domain)
        ).all()
    return [(d, n, int(c or 0)) for d, n, c in rows]


def attempt_rows() -> list[tuple[str, int, float]]:
    """[(知识域, 是否正确, 答题时间戳)]，画像聚合用（Elo 需按时间 replay）。"""
    with db.session() as s:
        rows = s.execute(
            select(Attempt.domain, Attempt.correct, Attempt.answered_at)
            .where(Attempt.domain.is_not(None))
        ).all()
    return [(d, bool(c), float(t)) for d, c, t in rows]


def last_wrong_choice(question_id: str) -> str | None:
    """某题最近一次答错时用户选的选项（定向出题的干扰项依据）。"""
    with db.session() as s:
        stmt = (select(Attempt.choice)
                .where(Attempt.question_id == question_id, Attempt.correct.is_(False))
                .order_by(Attempt.answered_at.desc())
                .limit(1))
        return s.scalar(stmt)


def recent_wrong_attempts(domain: str, limit: int = 5) -> list[dict]:
    """某域最近答错的题（去重取每题最后一次），定向出题的锚点。"""
    with db.session() as s:
        rows = s.execute(
            select(Attempt.question_id, Attempt.stem, Attempt.choice, Attempt.answered_at)
            .where(Attempt.domain == domain, Attempt.correct.is_(False))
            .order_by(Attempt.answered_at.desc())
            .limit(limit * 3)
        ).all()
    seen: set[str] = set()
    out = []
    for qid, stem, choice, at in rows:
        if qid in seen:
            continue
        seen.add(qid)
        out.append({"question_id": qid, "stem": stem or "", "choice": choice, "answered_at": at})
        if len(out) >= limit:
            break
    return out


# ── 提问信号（问答侧薄弱信号）────────────────────────────────────────────

def record_question_signal(thread_id: str, question: str,
                           domain: str | None, intent: str | None) -> None:
    with db.session() as s:
        s.add(QuestionSignal(thread_id=thread_id, question=question[:2000],
                             domain=domain, intent=intent, created_at=time.time()))
        s.commit()


def signal_rows() -> list[tuple[str, int, float]]:
    """[(知识域, 提问次数, 最近提问时间戳)]，画像融合用（仅已有域标签的）。"""
    with db.session() as s:
        rows = s.execute(
            select(QuestionSignal.domain,
                   func.count().label("n"),
                   func.max(QuestionSignal.created_at).label("last"))
            .where(QuestionSignal.domain.is_not(None))
            .group_by(QuestionSignal.domain)
        ).all()
    return [(d, int(n), float(last)) for d, n, last in rows]
