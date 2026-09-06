"""回答层相关性门限的单元测试（agents/graph._below_relevance_floor）。"""
import unittest

from agents import graph


class TestBelowRelevanceFloor(unittest.TestCase):
    def setUp(self):
        self._enabled = graph.config.ENABLE_RELEVANCE_GATE
        self._floor = graph.config.RELEVANCE_FLOOR

    def tearDown(self):
        graph.config.ENABLE_RELEVANCE_GATE = self._enabled
        graph.config.RELEVANCE_FLOOR = self._floor

    def test_blocks_when_all_below_floor(self):
        contexts = [{"dense_score": 0.2}, {"dense_score": self._floor - 0.01}]
        self.assertTrue(graph._below_relevance_floor(contexts))

    def test_passes_when_any_above_floor(self):
        contexts = [{"dense_score": 0.2}, {"dense_score": self._floor + 0.01}]
        self.assertFalse(graph._below_relevance_floor(contexts))

    def test_boundary_score_passes(self):
        # 恰好等于门限不算低于门限（严格小于才拦截）
        self.assertFalse(graph._below_relevance_floor([{"dense_score": self._floor}]))

    def test_missing_dense_score_passes_conservatively(self):
        # 旧缓存/BM25-only 结果没有 dense_score：放行，交给回答 prompt 兜底
        self.assertFalse(graph._below_relevance_floor([{"text": "旧结果"}]))
        self.assertFalse(graph._below_relevance_floor([]))

    def test_disabled_gate_passes_everything(self):
        graph.config.ENABLE_RELEVANCE_GATE = False
        self.assertFalse(graph._below_relevance_floor([{"dense_score": 0.1}]))


if __name__ == "__main__":
    unittest.main()
