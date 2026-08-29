"""判分逻辑单测（纯函数，无需 API/索引）。运行: pytest tests/test_grader.py -q"""
from quiz.grader import normalize_choice, grade


class FakeStore:
    """替换 store.record_attempt 避免写库。"""
    def __init__(self):
        self.records = []

    def record_attempt(self, **kw):
        self.records.append(kw)


def test_normalize_choice():
    assert normalize_choice("a") == "A"
    assert normalize_choice("BA") == "AB"
    assert normalize_choice("C ") == "C"
    assert normalize_choice("E") == ""


def test_grade_single(monkeypatch):
    import quiz.grader as g
    fake = FakeStore()
    monkeypatch.setattr(g.store, "record_attempt", fake.record_attempt)
    q = {"id": "t1", "source": "test", "domain": "信息安全管理", "stem": "题干",
         "options": {"A": "x", "B": "y", "C": "z", "D": "w"}, "answer": "B", "analysis": "因为B"}
    result = grade(q, "b")                       # 小写也算对
    assert result["correct"] is True
    assert result["answer"] == "B"
    assert "B" in result["analysis"]
    result = grade(q, "A")
    assert result["correct"] is False
    assert len(fake.records) == 2
    assert fake.records[0]["correct"] is True
    assert fake.records[1]["correct"] is False
