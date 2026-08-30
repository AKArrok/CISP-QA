"""薄弱点画像：按知识域统计正确率，判定薄弱域（结果经 Redis 缓存，答题时失效）。"""
from __future__ import annotations

import config
from quiz.store import STATS_CACHE_KEY
from storage import cache, repos


def domain_accuracy() -> dict[str, tuple[int, float | None]]:
    """{知识域: (答题数, 正确率|None)}。"""
    return {
        domain: (n, (c / n) if n else None)
        for domain, n, c in repos.domain_accuracy_rows()
    }


def weak_domains(min_attempts: int = config.WEAK_DOMAIN_MIN_ATTEMPTS,
                 threshold: float = config.WEAK_DOMAIN_ACCURACY) -> list[str]:
    return [
        domain for domain, (n, acc) in domain_accuracy().items()
        if n >= min_attempts and acc < threshold
    ]


def summary() -> dict:
    cached = cache.get_json(STATS_CACHE_KEY)
    if cached is not None:
        return cached
    accs = domain_accuracy()
    domains_detail = [
        {
            "domain": d,
            "attempts": n,
            "accuracy": round(acc, 4) if acc is not None else None,
            "weak": n >= config.WEAK_DOMAIN_MIN_ATTEMPTS
                    and acc is not None and acc < config.WEAK_DOMAIN_ACCURACY,
        }
        for d, (n, acc) in sorted(accs.items())
    ]
    total_attempts = sum(n for n, _ in accs.values())
    total_correct = sum(round(n * (acc or 0)) for n, acc in accs.values())
    result = {
        "total_attempts": total_attempts,
        "total_accuracy": round(total_correct / total_attempts, 4) if total_attempts else None,
        "domains": domains_detail,
        "weak_domains": weak_domains(),
    }
    cache.set_json(STATS_CACHE_KEY, result, ttl=config.STATS_CACHE_TTL)
    return result


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), ensure_ascii=False, indent=1))
