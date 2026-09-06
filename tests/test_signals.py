"""提问信号（问答侧薄弱信号）测试：落库、画像融合、抽题加权。"""
import sys

import pytest
from sqlalchemy import create_engine

sys.path.insert(0, ".")


@pytest.fixture()
def tmp_db(tmp_path):
    from storage import db
    from storage.models import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'signals.db'}",
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    db.override_engine(engine)
    yield engine
    engine.dispose()


def _seed_signals():
    from storage import repos

    for q in ["什么是 RTO？", "RPO 和 RTO 的区别？", "业务影响分析怎么做？"]:
        repos.record_question_signal("t1", q, "业务连续性", "knowledge")
    repos.record_question_signal("t1", "闲聊一句", None, "chitchat")  # 无域不统计


def test_signal_rows_aggregate(tmp_db):
    _seed_signals()
    from storage import repos

    rows = {d: (n, last) for d, n, last in repos.signal_rows()}
    assert rows["业务连续性"][0] == 3
    assert rows["业务连续性"][1] > 0


def test_profiles_fuse_ask_signal(tmp_db, monkeypatch):
    """只有提问、没答过题的域也出现在画像里，且标 ask_unverified。"""
    _seed_signals()
    import quiz.stats as stats

    monkeypatch.setattr(stats.repos, "attempt_rows", lambda: [])  # 无答题记录
    profiles = stats.domain_profiles()
    p = profiles["业务连续性"]
    assert p["attempts"] == 0
    assert p["asks"] == 3
    assert p["ask_unverified"] is True
    assert p["insufficient"] is True


def test_ask_signal_boosts_weak_mode_weight(tmp_db, monkeypatch):
    """高频提问未验证的域，在 weak 抽题权重里相对其他无记录域有加成。"""
    _seed_signals()
    from quiz.selector import QuestionBank, domain_weights

    # attempt_rows 置空，让所有域都是"无记录"基线
    import quiz.stats as stats
    monkeypatch.setattr(stats.repos, "attempt_rows", lambda: [])

    bank = QuestionBank.get()
    weights = domain_weights(bank, None)
    # 业务连续性因提问信号 ×1.5，其他无记录域保持 0.5 → 归一化后比值仍 1.5
    others = [w for d, w in weights.items()
              if d != "业务连续性" and d in stats.config_domains()]
    assert others, "题库应覆盖多个知识域"
    ratio = weights["业务连续性"] / (sum(others) / len(others))
    assert ratio == pytest.approx(1.5, rel=0.05)


def test_ask_signal_ignored_below_threshold(tmp_db, monkeypatch):
    """提问次数不足 ASK_SIGNAL_MIN 时不加权：该域只有 Elo 初始缺口权重（0.25），
    低于无记录域基线 0.5 → 比值应为 0.5 而非 1.5。"""
    from storage import repos

    repos.record_question_signal("t1", "什么是 RTO？", "业务连续性", "knowledge")
    import quiz.stats as stats
    monkeypatch.setattr(stats.repos, "attempt_rows", lambda: [])

    from quiz.selector import QuestionBank, domain_weights
    weights = domain_weights(QuestionBank.get(), None)
    others = [w for d, w in weights.items()
              if d != "业务连续性" and d in stats.config_domains()]
    ratio = weights["业务连续性"] / (sum(others) / len(others))
    assert ratio < 1.0  # 无提问加成，也不会被抬到基线之上
