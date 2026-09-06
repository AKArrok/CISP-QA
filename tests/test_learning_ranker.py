import json
from collections import Counter

from retrieval.learning_ranker import (
    FEATURE_NAMES,
    feature_vector,
    infer_query_intent,
    rerank_with_learned_model,
    should_apply_learned_ranker,
)
from tests.eval_retrieval_gold import GOLD_PATH
from tests.train_learning_ranker import SPLIT_PATH, load_split


def test_split_is_disjoint_complete_and_domain_balanced():
    gold = json.load(open(GOLD_PATH, encoding="utf-8"))
    train, dev = load_split(SPLIT_PATH, gold)

    assert len(train) == 60
    assert len(dev) == 20
    assert set(case["id"] for case in train).isdisjoint(case["id"] for case in dev)
    assert Counter(case["primary_domain"] for case in train) == {
        domain: 6 for domain in Counter(case["primary_domain"] for case in gold)
    }
    assert Counter(case["primary_domain"] for case in dev) == {
        domain: 2 for domain in Counter(case["primary_domain"] for case in gold)
    }


def test_feature_vector_uses_only_online_candidate_fields():
    chunk = {
        "score": 0.03,
        "dense_score": 0.7,
        "source": "信息安全管理_V4.2",
        "title": "监视和评审ISMS",
        "text": "C1-日常监视和检查。C2-进行有效性测量。C3-实施内部审核。",
        "element": "text",
    }

    values = feature_vector("监视和评审ISMS阶段需要做哪些工作？", chunk)

    assert len(values) == len(FEATURE_NAMES)
    assert values[FEATURE_NAMES.index("base_score")] == 0.03
    assert values[FEATURE_NAMES.index("dense_score")] == 0.7
    assert values[FEATURE_NAMES.index("process_intent_text")] == 1.0


def test_learned_model_reranks_by_model_score():
    model = {
        "features": FEATURE_NAMES,
        "intercept": 0.0,
        "coef": [0.0] * len(FEATURE_NAMES),
        "mean": [0.0] * len(FEATURE_NAMES),
        "scale": [1.0] * len(FEATURE_NAMES),
    }
    model["coef"][FEATURE_NAMES.index("phrase_match")] = 5.0
    generic = {"id": "generic", "score": 0.05, "title": "其他页", "text": "一些相关内容"}
    exact = {"id": "exact", "score": 0.01, "title": "能力级别-2级", "text": "规划执行"}

    ranked = rerank_with_learned_model("SSE-CMM能力级别2级包含哪些公共特征？", [generic, exact], model)

    assert ranked[0]["id"] == "exact"
    assert "learned_score" in ranked[0]


def test_learned_ranker_can_be_gated_by_query_intent():
    assert infer_query_intent("应急响应包括哪些阶段？") == "process"
    assert infer_query_intent("ISMS实施和运行阶段需要做哪些工作？") == "process"
    assert infer_query_intent("PKI由哪些组件组成？") == "acronym"
    assert should_apply_learned_ranker("补丁管理有哪些步骤？", "process")
    assert not should_apply_learned_ranker("CC标准中的EAL如何划分？", "process")
    assert should_apply_learned_ranker("CC标准中的EAL如何划分？", "all")
