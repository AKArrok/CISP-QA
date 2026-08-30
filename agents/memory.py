"""长期记忆：对话轮次落盘（conversations 表，MySQL 主 / SQLite 降级）。"""
from __future__ import annotations

import time

from sqlalchemy import func, select

from storage import db
from storage.models import Conversation


def save_round(thread_id: str, question: str, answer: str, intent: str) -> None:
    """一轮对话（用户问题 + 助手回答）两行落盘。"""
    now = time.time()
    with db.session() as s:
        s.add(Conversation(thread_id=thread_id, role="user", content=question, created_at=now))
        s.add(Conversation(thread_id=thread_id, role="assistant", content=answer,
                           intent=intent, created_at=now))
        s.commit()


def load_history(thread_id: str, limit: int = 50) -> list[dict]:
    """按时间序返回某会话的历史消息（旧的在前）。"""
    with db.session() as s:
        rows = s.scalars(
            select(Conversation)
            .where(Conversation.thread_id == thread_id)
            .order_by(Conversation.id.desc()).limit(limit)
        ).all()
    return [
        {"role": r.role, "content": r.content, "intent": r.intent,
         "created_at": r.created_at}
        for r in reversed(rows)
    ]


def list_threads(limit: int = 20) -> list[dict]:
    """历史会话列表（按最近消息时间倒序），供会话恢复/续聊入口用。"""
    with db.session() as s:
        rows = s.execute(
            select(Conversation.thread_id,
                   func.max(Conversation.created_at).label("last_at"),
                   func.count().label("rounds"))
            .group_by(Conversation.thread_id)
            .order_by(func.max(Conversation.created_at).desc())
            .limit(limit)
        ).all()
    return [
        {"thread_id": t, "last_at": last, "rounds": n // 2}
        for t, last, n in rows
    ]
