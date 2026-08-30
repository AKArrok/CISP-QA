"""ORM 模型 — 仿 Dify 的做法：全项目单一建模，方言由 DATABASE_URL 决定。

MySQL（utf8mb4）为主后端，SQLite 为零配置降级。向量索引不进关系库
（与 Dify/RAGFlow 一致：向量放在独立引擎，这里用本地 npz）。
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Question(Base):
    """结构化真题 + AI 生成题统一入库（AI 题复用此表，source='AI生成'）。"""
    __tablename__ = "questions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source: Mapped[str] = mapped_column(String(64), index=True)
    num: Mapped[int] = mapped_column(Integer, default=0)
    stem: Mapped[str] = mapped_column(Text)
    options: Mapped[str] = mapped_column(Text)      # JSON {"A":...}
    answer: Mapped[str] = mapped_column(String(8), default="")
    analysis: Mapped[str] = mapped_column(Text, default="")
    domain: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    dup_of: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class KBChunk(Base):
    """课件知识块（检索层启动时加载；BM25 与向量索引仍由本地文件构建）。"""
    __tablename__ = "kb_chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    domain: Mapped[str] = mapped_column(String(32), index=True)
    source: Mapped[str] = mapped_column(String(128))
    page: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(Text)


class Attempt(Base):
    __tablename__ = "attempts"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"),
                                     primary_key=True, autoincrement=True)
    question_id: Mapped[str] = mapped_column(String(64), index=True)
    question_source: Mapped[str] = mapped_column(String(64))
    domain: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    stem: Mapped[str | None] = mapped_column(Text, nullable=True)
    choice: Mapped[str] = mapped_column(String(8))
    correct: Mapped[bool] = mapped_column(Boolean)
    answered_at: Mapped[float] = mapped_column(Float)


class Conversation(Base):
    """长期记忆：对话轮次落盘。"""
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"),
                                     primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))          # user | assistant
    content: Mapped[str] = mapped_column(Text)
    intent: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[float] = mapped_column(Float)


class Metric(Base):
    """请求级可观测性指标。"""
    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"),
                                     primary_key=True, autoincrement=True)
    thread_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    question: Mapped[str | None] = mapped_column(Text, nullable=True)
    intent: Mapped[str | None] = mapped_column(String(16), nullable=True)
    route_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    retrieval_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    first_token_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[float] = mapped_column(Float)
