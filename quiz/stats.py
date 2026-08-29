"""薄弱点画像：按知识域统计正确率，判定薄弱域。"""
from __future__ import annotations

import sqlite3

import config
from quiz import store


def domain_accuracy() -> dict[str, tuple[int, float | None]]:
    """{知识域: (答题数, 正确率|None)}。"""
    with sqlite3.connect(config.DB_PATH) as conn:
        rows = conn.execute(
            "SELECT domain, COUNT(*) AS n, SUM(correct) AS c FROM attempts "
            "WHERE domain IS NOT NULL GROUP BY domain"
        ).fetchall()
    return {
        domain: (n, (c / n) if n else None)
        for domain, n, c in rows
    }


def weak_domains(min_attempts: int = config.WEAK_DOMAIN_MIN_ATTEMPTS,
                 threshold: float = config.WEAK_DOMAIN_ACCURACY) -> list[str]:
    return [
        domain for domain, (n, acc) in domain_accuracy().items()
        if n >= min_attempts and acc < threshold
    ]


def summary() -> dict:
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
    return {
        "total_attempts": total_attempts,
        "total_accuracy": round(total_correct / total_attempts, 4) if total_attempts else None,
        "domains": domains_detail,
        "weak_domains": weak_domains(),
    }


if __name__ == "__main__":
    import json
    store.init_db()
    print(json.dumps(summary(), ensure_ascii=False, indent=1))
