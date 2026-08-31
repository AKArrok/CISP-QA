"""薄弱点画像：Beta 掌握度 + Elo 评级 + 遗忘曲线（结果经 Redis 缓存，答题时失效）。

参考成熟做法（轻量、无需训练）：
- Beta 后验置信区间：样本不足时不确定能力高低，用 P(能力<阈值) 判薄弱，避免小样本误判。
- Elo 评级（Pelánek 2016）：按时间顺序 replay attempts，答对/答错按期望概率增量更新。
- 遗忘曲线（Duolingo HLR / Ebbinghaus）：记忆强度随间隔指数衰减，到期进入复习队列。
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from quiz.store import STATS_CACHE_KEY
from storage import cache, repos


def _beta_mean(a: float, b: float) -> float:
    """Beta 分布均值（后验掌握度）。"""
    return a / (a + b) if a + b else 0.5


def _beta_p_below(a: float, b: float, threshold: float) -> float:
    """P(X < threshold)，X ~ Beta(a, b)。用不完整 Beta 函数（正则化）精确计算。"""
    from scipy.special import betainc
    return betainc(a, b, threshold)


def _confidence(a: float, b: float) -> float:
    """置信度 = 1 - 90% 区间宽度（样本越多越窄 → 越接近 1）。"""
    from scipy.stats import beta
    lo, hi = beta.ppf([0.05, 0.95], a, b)
    return max(0.0, min(1.0, 1.0 - (hi - lo)))


def _memory_strength(days: float) -> float:
    """遗忘曲线：2^(-天数/半衰期)。"""
    if days <= 0:
        return 1.0
    return 2.0 ** (-days / config.MASTERY_HALF_LIFE_DAYS)


def _elo_expected(rating: float, opponent: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(rating - opponent) / 400.0))


def domain_profiles(now: float | None = None) -> dict[str, dict]:
    """{域: 画像}，由 attempts 实时聚合。画像含 Beta 后验参数、Elo、遗忘曲线指标。"""
    now = now if now is not None else time.time()
    # rows: (domain, correct, answered_at) 按时间排序
    rows = sorted(
        repos.attempt_rows(), key=lambda r: r[2]  # answered_at
    )
    by_domain: dict[str, dict] = {}

    for domain, correct, answered_at in rows:
        p = by_domain.setdefault(domain, {
            "attempts": 0, "correct": 0, "wrong": 0,
            "last_answered_at": 0.0, "streak": 0, "max_streak": 0,
            "rating": config.ELO_INIT, "rating_history": [],
        })
        p["attempts"] += 1
        p["correct"] += 1 if correct else 0
        p["wrong"] += 0 if correct else 1
        p["last_answered_at"] = max(p["last_answered_at"], answered_at)
        p["streak"] = p["streak"] + 1 if correct else 0
        p["max_streak"] = max(p["max_streak"], p["streak"])

        # Elo 增量更新：对手难度 = 该域历史正确率
        opp_guess = (p["correct"] / p["attempts"]) if p["attempts"] else 0.5
        opponent = 1500 + (opp_guess - 0.5) * 800
        expected = _elo_expected(p["rating"], opponent)
        outcome = 1.0 if correct else 0.0
        p["rating"] += config.ELO_K * (outcome - expected)
        p["rating_history"].append(p["rating"])

    profiles: dict[str, dict] = {}
    for domain, p in by_domain.items():
        a = p["correct"] + config.BETA_PRIOR
        b = p["wrong"] + config.BETA_PRIOR
        mastery = _beta_mean(a, b)
        days_since = (now - p["last_answered_at"]) / 86400.0 if p["last_answered_at"] else None
        strength = _memory_strength(days_since) if days_since is not None else 1.0
        p_below = float(_beta_p_below(a, b, config.MASTERY_WEAK_THRESHOLD))
        profiles[domain] = {
            **p,
            "mastery": round(mastery, 4),
            "confidence": round(float(_confidence(a, b)), 4),
            "p_below_threshold": round(p_below, 4),
            "beta_a": round(a, 2), "beta_b": round(b, 2),
            "last_seen_days": round(days_since, 1) if days_since is not None else None,
            "memory_strength": round(float(strength), 4),
            "due_for_review": bool(strength <= config.REVIEW_DUE_STRENGTH),
            # 薄弱 = 后验概率超过阈值（样本不足时概率自然不高，不会误判）
            "weak": bool(p["attempts"] >= 2 and p_below > config.MASTERY_WEAK_PROB),
            "insufficient": bool(p["attempts"] < 2),
        }
    return profiles


def weak_domains() -> list[str]:
    return sorted(
        d for d, p in domain_profiles().items() if p["weak"]
    )


def review_due_domains() -> list[str]:
    return sorted(
        d for d, p in domain_profiles().items()
        if not p["insufficient"] and p["due_for_review"]
    )


def summary() -> dict:
    cached = cache.get_json(STATS_CACHE_KEY)
    if cached is not None:
        return cached
    profiles = domain_profiles()
    domains_detail = [
        {
            "domain": d,
            "attempts": p["attempts"],
            "correct": p["correct"],
            "accuracy": round(p["correct"] / p["attempts"], 4) if p["attempts"] else None,
            "mastery": p["mastery"],
            "confidence": p["confidence"],
            "weak": p["weak"],
            "insufficient": p["insufficient"],
            "due_for_review": p["due_for_review"],
            "last_seen_days": p["last_seen_days"],
            "rating": round(p["rating"], 1),
            "streak": p["streak"],
        }
        for d, p in sorted(profiles.items())
    ]
    total_attempts = sum(p["attempts"] for p in profiles.values())
    total_correct = sum(p["correct"] for p in profiles.values())
    weak = weak_domains()
    result = {
        "total_attempts": total_attempts,
        "total_accuracy": round(total_correct / total_attempts, 4) if total_attempts else None,
        "domains": domains_detail,
        "weak_domains": weak,
        "review_due_domains": review_due_domains(),
        "model": {
            "type": "beta_elo_forgetting",
            "half_life_days": config.MASTERY_HALF_LIFE_DAYS,
            "weak_prob": config.MASTERY_WEAK_PROB,
        },
    }
    cache.set_json(STATS_CACHE_KEY, result, ttl=config.STATS_CACHE_TTL)
    return result


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), ensure_ascii=False, indent=1))
