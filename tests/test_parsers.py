"""解析产物自检。分两层:
- fixtures 子集: 任何环境（含 CI）都可跑
- 本地全量数据: data/ 不入库，存在时才跑（本地回归用）
"""
import json
import os

import pytest

import config
from domains import DOMAINS

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

REAL_DATA = (
    os.path.exists(config.QUESTIONS_PATH) and os.path.exists(config.KB_CHUNKS_PATH)
)


def _load(path):
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def _fixture_questions():
    return _load(os.path.join(FIXTURE_DIR, "questions_sample.json"))


def _fixture_chunks():
    return _load(os.path.join(FIXTURE_DIR, "kb_chunks_sample.json"))


# ── 通用断言（fixtures 与本地全量共用）────────────────────────────────────

def check_questions(questions):
    for q in questions:
        assert q["id"] and q["source"] and q["stem"].strip(), q["id"]
        if q["answer"]:
            assert set(q["answer"]) <= set("ABCD"), q["id"]
            if not q.get("needs_review"):
                for letter in q["answer"]:
                    assert letter in q["options"], q["id"]
        if q["answer"] and not set(q["answer"]) <= set(q["options"]):
            assert q.get("needs_review"), q["id"]
        assert q["domain"] in DOMAINS or q.get("needs_review"), q["id"]


def check_chunks(chunks):
    for c in chunks:
        assert c["id"] and c["source"] and c["page"] >= 1 and c["text"].strip(), c["id"]
        assert c["domain"] in DOMAINS, c["id"]


# ── fixtures 层（CI 可跑）────────────────────────────────────────────────

def test_fixture_questions():
    check_questions(_fixture_questions())


def test_fixture_chunks():
    check_chunks(_fixture_chunks())


# ── 本地全量层（缺 data/ 时 skip）─────────────────────────────────────────

@pytest.mark.skipif(not REAL_DATA, reason="本地 data/ 不存在（CI 环境）")
class TestLocalData:
    def setup_method(self):
        self.questions = _load(config.QUESTIONS_PATH)
        self.chunks = _load(config.KB_CHUNKS_PATH)

    def test_question_bank_nonempty(self):
        assert len(self.questions) >= 1000

    def test_questions_valid(self):
        check_questions(self.questions)

    def test_kb_chunks_valid_and_cover_all_domains(self):
        check_chunks(self.chunks)
        covered = {c["domain"] for c in self.chunks}
        for d in DOMAINS:
            assert d in covered, d
