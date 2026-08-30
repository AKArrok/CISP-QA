"""长期记忆（对话落盘）单测 — 独立临时库（SQLAlchemy 引擎级隔离），不碰真实数据。"""
import sys

sys.path.insert(0, ".")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from storage import db as storage_db
from storage.models import Base


def _fresh_memory(tmp_path, monkeypatch):
    """指向临时库的引擎，并重置各模块缓存。"""
    import agents.memory  # noqa: F401  确保模块已加载
    engine = create_engine(f"sqlite:///{tmp_path}/mem_test.db",
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    monkeypatch.setattr(storage_db, "_engine", engine)
    monkeypatch.setattr(storage_db, "_SessionLocal", sessionmaker(bind=engine,
                                                                  expire_on_commit=False))
    monkeypatch.setattr(storage_db, "_backend", "sqlite")
    from storage import cache
    monkeypatch.setattr(cache, "_client", cache._MemClient(), raising=False)
    return sys.modules["agents.memory"]


def test_save_and_load_round(tmp_path, monkeypatch):
    mem = _fresh_memory(tmp_path, monkeypatch)
    mem.save_round("t1", "什么是RTO？", "RTO 是恢复时间目标……", "knowledge")
    hist = mem.load_history("t1")
    assert [h["role"] for h in hist] == ["user", "assistant"]
    assert hist[0]["content"] == "什么是RTO？"
    assert hist[1]["intent"] == "knowledge"


def test_threads_order_and_rounds(tmp_path, monkeypatch):
    mem = _fresh_memory(tmp_path, monkeypatch)
    mem.save_round("tA", "q1", "a1", "knowledge")
    mem.save_round("tB", "q2", "a2", "chitchat")
    mem.save_round("tA", "q3", "a3", "knowledge")  # tA 两轮，且最新
    threads = mem.list_threads()
    assert threads[0]["thread_id"] == "tA"
    assert threads[0]["rounds"] == 2
    assert threads[1]["rounds"] == 1


def test_history_limit_keeps_latest(tmp_path, monkeypatch):
    mem = _fresh_memory(tmp_path, monkeypatch)
    for i in range(3):
        mem.save_round("tX", f"q{i}", f"a{i}", "knowledge")
    hist = mem.load_history("tX", limit=2)
    assert len(hist) == 2
    assert hist[0]["content"] == "q2"  # 保留最新两轮，旧的在前
