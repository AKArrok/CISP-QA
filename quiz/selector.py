"""抽题：random（均匀）/ weak（薄弱域 1-正确率 加权）。"""
from __future__ import annotations

import json
import random

import config
from quiz import store


class QuestionBank:
    """真题库（questions.json）+ AI 生成题（SQLite）的统一视图。"""

    _instance = None

    def __init__(self):
        with open(config.QUESTIONS_PATH, encoding="utf-8") as fp:
            self._real = json.load(fp)
        self.reload_ai()

    @classmethod
    def get(cls) -> "QuestionBank":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reload_ai(self) -> None:
        self._ai = store.list_ai_questions()

    def available(self, domain: str | None = None) -> list[dict]:
        """可出题目：有效（题干/选项/答案齐全）、非重复题；默认排除已答对的题。"""
        done_correct = store.attempted_ids(correct_only=True)
        pool = [
            q for q in self._real + self._ai
            if q["answer"] and not q.get("needs_review") and not q.get("dup_of")
        ]
        if domain:
            pool = [q for q in pool if q["domain"] == domain]
        fresh = [q for q in pool if q["id"] not in done_correct]
        return fresh or pool  # 全部做过时允许复做

    def domains(self) -> list[str]:
        return sorted({q["domain"] for q in self._real + self._ai if q["domain"]})


def domain_weights(bank: QuestionBank, domain: str | None) -> dict[str, float]:
    """weak 模式: 各域权重 = 1 - 正确率（无答题记录的域给中等权重 0.5）。"""
    from quiz.stats import domain_accuracy

    accs = domain_accuracy()
    domains = [d for d in bank.domains() if (not domain or d == domain)]
    weights = {}
    for d in domains:
        attempts, acc = accs.get(d, (0, None))
        weights[d] = (1.0 - acc) if acc is not None else 0.5
    total = sum(weights.values()) or 1.0
    return {d: w / total for d, w in weights.items()}


def next_question(mode: str = "random", domain: str | None = None) -> dict | None:
    """抽一道题。weak 模式先按薄弱加权抽域，再在域内均匀抽题。"""
    bank = QuestionBank.get()
    if mode == "weak":
        weights = domain_weights(bank, domain)
        if not weights:
            return None
        domains, probs = list(weights.keys()), list(weights.values())
        picked_domain = random.choices(domains, weights=probs, k=1)[0]
        pool = bank.available(picked_domain)
    else:
        pool = bank.available(domain)
    return random.choice(pool) if pool else None
