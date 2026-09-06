"""轻量学习排序：用可线上复现的特征重排候选页。"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from retrieval.bm25 import tokenize


FEATURE_NAMES = [
    "base_score",
    "dense_score",
    "title_overlap",
    "text_overlap",
    "phrase_match",
    "acronym_hit",
    "source_cisp42",
    "source_v42",
    "overview_penalty",
    "short_detail",
    "composition_intent_title",
    "composition_intent_text",
    "process_intent_title",
    "process_intent_text",
    "table_numeric",
]

_DETAIL_QUERY_CUES = ("哪些", "什么", "如何", "怎么", "区别", "阶段", "步骤", "目的", "方法", "组成", "包括")
_COMPOSITION_QUERY_CUES = ("组成", "构成", "由哪些", "包括哪些")
_PROCESS_QUERY_CUES = ("流程", "过程", "步骤", "阶段")
_FACT_QUERY_CUES = ("几", "多少", "哪一年", "什么时间", "什么时候", "等级", "级别", "分类", "分为", "划分")
_COMPOSITION_TEXT_CUES = ("组成", "构成", "元素", "部件", "部分")
_PROCESS_TEXT_CUES = ("流程", "过程", "步骤", "阶段", "P1-", "D1-", "C1-")
_OVERVIEW_TITLE_PREFIXES = ("知识子域", "课程内容")
_OVERVIEW_VERBS = ("了解", "理解", "掌握")


def _acronyms(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z][A-Za-z0-9+-]{1,}", text))


def _compact(text: str) -> str:
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text.lower()))


def _overlap_score(terms: list[str], text: str) -> float:
    if not terms:
        return 0.0
    normalized = re.sub(r"\s+", "", text.lower())
    return sum(term in normalized for term in terms) / len(terms)


def _phrase_score(query: str, title: str) -> float:
    query_compact = _compact(query)
    title_compact = _compact(title)
    if not query_compact or not title_compact:
        return 0.0
    if title_compact in query_compact or query_compact in title_compact:
        return 1.0
    if len(title_compact) >= 5 and title_compact[:5] in query_compact:
        return 0.6
    return 0.0


def _overview_value(title: str, text: str) -> float:
    if title.startswith(_OVERVIEW_TITLE_PREFIXES):
        return 1.0
    if sum(text.count(verb) for verb in _OVERVIEW_VERBS) >= 3:
        return 0.5
    return 0.0


def infer_query_intent(query: str) -> str:
    """线上查询意图粗分类，用于门控学习排序。"""
    if any(cue in query for cue in _PROCESS_QUERY_CUES):
        return "process"
    if _acronyms(query):
        return "acronym"
    if any(cue in query for cue in _FACT_QUERY_CUES) or re.search(r"\d|第[一二三四五六七八九十]+", query):
        return "fact"
    if any(cue in query for cue in ("区别", "对比", "比较")):
        return "comparison"
    if any(cue in query for cue in ("什么是", "定义", "概念", "含义")):
        return "definition"
    return "other"


def should_apply_learned_ranker(query: str, enabled_query_types: str | None) -> bool:
    """判断学习排序是否应该接管当前查询。"""
    raw = (enabled_query_types or "").strip().lower()
    if raw in {"", "none", "false", "0"}:
        return False
    if raw == "all":
        return True
    allowed = {item.strip() for item in raw.split(",") if item.strip()}
    return infer_query_intent(query) in allowed


def feature_vector(query: str, chunk: dict) -> list[float]:
    """抽取学习排序特征；只使用线上可见信息，不能使用 gold 证据词。"""
    title = chunk.get("title", "")
    text = chunk.get("text", "")
    source = chunk.get("source", "")
    query_terms = [re.sub(r"\s+", "", term.lower()) for term in tokenize(query)]
    query_acronyms = _acronyms(query)
    asks_for_detail = any(cue in query for cue in _DETAIL_QUERY_CUES)
    asks_composition = any(cue in query for cue in _COMPOSITION_QUERY_CUES)
    asks_process = any(cue in query for cue in _PROCESS_QUERY_CUES)
    numeric_query = bool(re.search(r"\d|第[一二三四五六七八九十]+|等级|字段|取值|赋值|位|表", query))
    return [
        float(chunk.get("score", 0.0)),
        float(chunk.get("dense_score") or 0.0),
        _overlap_score(query_terms, title),
        _overlap_score(query_terms, text),
        _phrase_score(query, title),
        float(bool(query_acronyms & _acronyms(title))),
        float("CISP 4.2" in source),
        float("V4.2" in source or "v4.2" in source),
        _overview_value(title, text),
        float(asks_for_detail and len(text) < 120),
        float(asks_composition and any(cue in title for cue in _COMPOSITION_TEXT_CUES)),
        float(asks_composition and any(cue in text for cue in _COMPOSITION_TEXT_CUES)),
        float(asks_process and any(cue in title for cue in _PROCESS_TEXT_CUES)),
        float(asks_process and any(cue in text for cue in _PROCESS_TEXT_CUES)),
        float(chunk.get("element") == "table" and numeric_query),
    ]


def load_model(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fp:
        model = json.load(fp)
    if model.get("features") != FEATURE_NAMES:
        raise ValueError("学习排序模型的特征列表与当前代码不一致")
    return model


def score(query: str, chunk: dict, model: dict) -> float:
    values = feature_vector(query, chunk)
    mean = model["mean"]
    scale = model["scale"]
    coef = model["coef"]
    z = model["intercept"]
    for value, avg, std, weight in zip(values, mean, scale, coef, strict=True):
        z += ((value - avg) / (std or 1.0)) * weight
    return 1.0 / (1.0 + math.exp(-max(min(z, 50), -50)))


def rerank_with_learned_model(query: str, candidates: list[dict], model: dict) -> list[dict]:
    ranked = []
    for index, chunk in enumerate(candidates):
        learned_score = score(query, chunk, model)
        # 保留原始排序作稳定兜底：学习分相同则不打乱候选。
        ranked.append(({**chunk, "learned_score": learned_score}, index))
    ranked.sort(key=lambda item: (-item[0]["learned_score"], item[1]))
    return [item for item, _ in ranked]
