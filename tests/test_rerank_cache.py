"""rerank 磁盘缓存：按 (query, 文档) 对命中，同一查询换候选池也能复用。"""
import json

import pytest

import config
from llms import DashScopeReranker

# 与真机一致的确定性假分数：按文档内容给分，与同池其他文档无关
FAKE_SCORES = {"文档甲": 0.2, "文档乙": 0.5, "文档丙": 0.7, "新文档": 0.9}


@pytest.fixture
def reranker(tmp_path, monkeypatch):
    cache_path = str(tmp_path / "rerank_cache.json")
    monkeypatch.setattr(config, "RERANK_CACHE_PATH", cache_path)
    monkeypatch.setattr(config, "RERANK_CACHE_ENABLED", True)
    rr = DashScopeReranker(api_key="test-key", base_url="https://example.invalid",
                           model="test-rerank")
    calls = []

    def fake_post(query, documents, top_n):
        calls.append((query, list(documents)))
        payload = {"output": {"results": [
            {"index": i, "relevance_score": FAKE_SCORES[doc]}
            for i, doc in enumerate(documents)
        ]}}
        return payload

    monkeypatch.setattr(rr, "_post", fake_post)
    return rr, calls, cache_path


def test_second_call_fully_served_from_cache(reranker):
    rr, calls, _ = reranker

    first = rr.rerank("同一查询", ["文档甲", "文档乙"])
    second = rr.rerank("同一查询", ["文档甲", "文档乙"])

    assert len(calls) == 1  # 第二次全部命中缓存，不发请求
    assert first == second == [pytest.approx(0.2), pytest.approx(0.5)]


def test_shuffled_pool_reuses_pair_scores(reranker):
    """同一查询换候选池顺序/子集，已打过的对不重复请求。"""
    rr, calls, _ = reranker
    rr.rerank("查询A", ["文档甲", "文档乙", "文档丙"])

    scores = rr.rerank("查询A", ["文档丙", "文档甲"])

    assert len(calls) == 1  # 没有新请求：池变了但每对都缓存过
    assert scores == [pytest.approx(0.7), pytest.approx(0.2)]


def test_new_query_or_doc_triggers_request(reranker):
    rr, calls, cache_path = reranker
    rr.rerank("查询A", ["文档甲"])

    rr.rerank("查询B", ["文档甲"])  # 查询变了 → 重新请求
    rr.rerank("查询A", ["文档甲", "新文档"])  # 只补新文档那对

    assert len(calls) == 3
    assert calls[1] == ("查询B", ["文档甲"])
    assert calls[2] == ("查询A", ["新文档"])

    with open(cache_path, encoding="utf-8") as fp:
        assert len(json.load(fp)) == 3  # (A,甲) (B,甲) (A,新文档)


def test_cache_persists_across_instances(reranker):
    rr, calls, cache_path = reranker
    rr.rerank("查询A", ["文档甲"])

    fresh = DashScopeReranker(api_key="test-key", base_url="https://example.invalid",
                              model="test-rerank")
    fresh._post = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("缓存命中时不允许发起请求"))
    scores = fresh.rerank("查询A", ["文档甲"])

    assert scores == [pytest.approx(0.2)]
    assert len(calls) == 1
