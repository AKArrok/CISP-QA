"""AI 补充出题：检索该域知识块 → LLM 生成单选题（落库去重）。"""
from __future__ import annotations

import logging

from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage

from domains import normalize_for_match
from llms import get_answer_llm, invoke_structured
from quiz import store
from quiz.selector import QuestionBank

logger = logging.getLogger(__name__)


class GeneratedQuestion(BaseModel):
    stem: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    answer: str      # 'A'-'D'
    analysis: str


GEN_SYSTEM = """你是 CISP 考试出题人。依据给定的课件知识块出 1 道单选题，要求:
1. 考点必须完全来自资料内容，不得编造
2. 风格贴近 CISP 真题（"以下哪一项…""关于…说法正确的是"）
3. 四个选项干扰项要有合理性，答案唯一且在 A-D 中
4. analysis 给出答案依据（引用资料要点）
只输出 JSON。"""


def _looks_duplicate(stem: str) -> bool:
    key = normalize_for_match(stem)[:40]
    bank = QuestionBank.get()
    known = {normalize_for_match(q["stem"])[:40] for q in bank._questions}
    return key in known or stem in store.existing_ai_stems()


def generate_question(domain: str) -> dict | None:
    """为指定知识域生成一道新题；与现有题库查重，失败返回 None。"""
    from retrieval.hybrid import HybridRetriever

    chunks = HybridRetriever.get().retrieve(
        f"{domain} 核心考点", top_k=3, domain=domain)
    if not chunks:
        return None
    context_text = "\n\n".join(c["text"][:600] for c in chunks)
    llm = get_answer_llm(temperature=0.5)
    try:
        out = invoke_structured(llm, GeneratedQuestion, [
            SystemMessage(content=GEN_SYSTEM),
            HumanMessage(content=f"知识域: {domain}\n\n课件资料:\n{context_text}"),
        ])
    except Exception:
        logger.exception("AI 出题失败 (%s)", domain)
        return None

    answer = out.answer.strip().upper()[:1]
    if answer not in "ABCD" or not out.stem.strip():
        return None
    stem = out.stem.strip()
    if _looks_duplicate(stem):
        return None
    options = {"A": out.option_a, "B": out.option_b, "C": out.option_c, "D": out.option_d}
    qid = store.insert_ai_question(stem, options, answer, out.analysis.strip(), domain)
    QuestionBank.get().reload()
    return {
        "id": qid, "source": "AI生成", "num": 0, "stem": stem,
        "options": options, "answer": answer, "analysis": out.analysis.strip(),
        "domain": domain, "needs_review": False,
    }
