"""quiz 门面：出题 → 作答 → 统计的对外接口（server.py 调用）。"""
from __future__ import annotations

from quiz import store
from quiz.selector import next_question, QuestionBank
from quiz.grader import grade
from quiz.stats import summary
import config

store.init_db()


def get_next_question(mode: str = "random", domain: str | None = None) -> dict:
    """抽题；真题不足时按需 AI 生成补充（脱敏：不含答案）。"""
    q = next_question(mode=mode, domain=domain)

    # AI 补充出题条件：weak 模式且该域可做题极少，或指定域无真题
    if (q is None or mode == "weak") and domain is not None:
        bank = QuestionBank.get()
        if q is None or len(bank.available(domain)) < config.QUIZ_AI_MIN_BANK:
            from quiz.generator import generate_question
            try:
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


def submit_answer(question_id: str, choice: str) -> dict:
    """作答判分。真题从内存题库取，AI 题从 SQLite 取。"""
    bank = QuestionBank.get()
    question = next((q for q in bank._real + bank._ai if q["id"] == question_id), None)
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
