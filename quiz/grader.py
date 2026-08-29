"""判分与解析组装（确定性比对，不走 LLM）。"""
from __future__ import annotations

from quiz import store


def normalize_choice(choice: str) -> str:
    return "".join(sorted(c.upper() for c in choice if c.upper() in "ABCD"))


def grade(question: dict, choice: str) -> dict:
    """判分 → 落库 → 返回对错/正确答案/解析/关联课件。"""
    correct = normalize_choice(choice) == normalize_choice(question["answer"])
    store.record_attempt(
        question_id=question["id"],
        question_source=question.get("source", "unknown"),
        domain=question.get("domain"),
        stem=question["stem"],
        choice=normalize_choice(choice),
        correct=correct,
    )
    return {
        "correct": correct,
        "answer": question["answer"],
        "analysis": question.get("analysis") or "题库未附解析。",
        "related_chunks": related_chunks(question),
    }


def related_chunks(question: dict, top_k: int = 2) -> list[dict]:
    """用题干在该题知识域内检索关联课件片段（答错时帮助回看原文）。"""
    try:
        from retrieval.hybrid import HybridRetriever

        return HybridRetriever.get().retrieve(
            question["stem"], top_k=top_k, domain=question.get("domain"))
    except Exception:
        return []
