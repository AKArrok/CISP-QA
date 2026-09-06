"""quiz 门面：出题 → 作答 → 统计的对外接口（server.py 调用）。"""
from __future__ import annotations

import logging

from quiz import store
from quiz.selector import next_question, QuestionBank
from quiz.grader import grade
from quiz.stats import summary
import config

logger = logging.getLogger(__name__)

store.init_db()


def get_next_question(mode: str = "random", domain: str | None = None,
                      anchor_question_id: str | None = None) -> dict:
    """抽题；weak 模式可带锚点题定向生成同类题（脱敏：不含答案）。"""
    q = None
    # "再练一道同类题"：以指定题（通常是刚答错的）为锚点，围绕同一考点定向生成
    if anchor_question_id:
        anchor_q = QuestionBank.get().get_by_id(anchor_question_id)
        if anchor_q is not None:
            wrong_choice = _last_wrong_choice(anchor_question_id)
            try:
                from quiz.generator import _anchor_from_question, generate_question
                q = generate_question(anchor_q["domain"],
                                      anchor=_anchor_from_question(anchor_q, wrong_choice))
            except Exception:
                logger.exception("锚点定向出题失败，回退常规抽题")
    if q is None:
        q = next_question(mode=mode, domain=domain)

    # AI 补充出题条件：指定域无题可抽，或域内可做题极少
    if q is None and domain is not None:
        bank = QuestionBank.get()
        if len(bank.available(domain)) < config.QUIZ_AI_MIN_BANK:
            try:
                from quiz.generator import generate_question
                generated = generate_question(domain)
            except Exception:
                generated = None
            if generated:
                q = generated

    if q is None:
        raise ValueError(f"该知识域没有可用题目: {domain or '全部'}")
    # 脱敏返回
    return {
        "id": q["id"],
        "source": q["source"],
        "domain": q["domain"],
        "stem": q["stem"],
        "options": q["options"],
    }


def _last_wrong_choice(question_id: str) -> str | None:
    """该题最近一次答错的选项（定向出题的干扰项依据）。"""
    try:
        from storage import repos
        return repos.last_wrong_choice(question_id)
    except Exception:
        return None


def submit_answer(question_id: str, choice: str) -> dict:
    """作答判分。真题从内存题库取，AI 题从 SQLite 取。"""
    question = QuestionBank.get().get_by_id(question_id)
    if question is None:
        raise KeyError(f"题目不存在: {question_id}")
    result = grade(question, choice)
    return {
        "correct": result["correct"],
        "answer": result["answer"],
        "analysis": result["analysis"],
        "related_chunks": [
            {
                "source": c["source"], "page": c["page"],
                "domain": c["domain"], "text": c["text"][:300],
            }
            for c in result["related_chunks"]
        ],
    }


def get_stats() -> dict:
    return summary()
