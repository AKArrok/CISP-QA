"""在固定 train/dev split 上训练轻量学习排序器，并诊断 EAL / process 问题。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
sys.path.insert(0, "tests")

import config
from eval_retrieval_gold import (
    GOLD_PATH,
    aggregate,
    embedding_label,
    evaluate_case,
    load_json,
    resolve_relevant_parent_ids,
    validate_gold,
)
from retrieval.bm25 import BM25Index
from retrieval.hybrid import _rrf_fuse
from retrieval.learning_ranker import FEATURE_NAMES, feature_vector, rerank_with_learned_model
from retrieval.learning_ranker import should_apply_learned_ranker
from retrieval.query_rewrite import rewrite_query
from retrieval.rank_features import fold_near_duplicate_pages, rerank_with_features
from retrieval.vector_store import VectorStore

SPLIT_PATH = Path(__file__).parent / "fixtures" / "retrieval_gold_split.json"


class CandidateRetriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = {chunk["id"]: chunk for chunk in chunks}
        self.vector = VectorStore.load()
        self.bm25 = BM25Index.build(chunks)

    def retrieve(self, query: str, query_vec: list[float], top_k: int) -> list[dict]:
        dense = self.vector.search(query_vec, config.DENSE_K)
        dense_scores = {item["id"]: item["score"] for item in dense}
        sparse = self.bm25.search(query, config.SPARSE_K)
        fused = _rrf_fuse(dense, sparse, k=config.RRF_K)
        full = [
            {
                **self.chunks[item["id"]],
                "score": item["rrf_score"],
                "dense_score": dense_scores.get(item["id"]),
            }
            for item in fused
        ]
        unique = []
        seen = set()
        for chunk in full:
            parent_id = chunk.get("parent_id", chunk["id"])
            if parent_id not in seen:
                seen.add(parent_id)
                unique.append(chunk)
        if config.ENABLE_FEATURE_RERANKING:
            unique = fold_near_duplicate_pages(rerank_with_features(query, unique))
        return unique[:top_k]


def load_split(path: str | Path, gold: list[dict]) -> tuple[list[dict], list[dict]]:
    split = json.load(open(path, encoding="utf-8"))
    by_id = {case["id"]: case for case in gold}
    train_ids = split["train"]
    dev_ids = split["dev"]
    all_ids = set(by_id)
    if set(train_ids) & set(dev_ids):
        raise AssertionError("train/dev split 有重叠")
    if set(train_ids) | set(dev_ids) != all_ids:
        missing = sorted(all_ids - (set(train_ids) | set(dev_ids)))
        extra = sorted((set(train_ids) | set(dev_ids)) - all_ids)
        raise AssertionError(f"train/dev split 不覆盖 gold：missing={missing}, extra={extra}")
    return [by_id[item] for item in train_ids], [by_id[item] for item in dev_ids]


def query_vectors_for(gold: list[dict]) -> dict[str, list[float]]:
    from llms import get_embeddings

    questions = [
        rewrite_query(case["question"]) if config.ENABLE_QUERY_REWRITE else case["question"]
        for case in gold
    ]
    vectors = get_embeddings().embed_documents(questions)
    return {case["id"]: vector for case, vector in zip(gold, vectors, strict=True)}


def collect_training_rows(
    cases: list[dict],
    chunks: list[dict],
    query_vectors: dict[str, list[float]],
    retriever: CandidateRetriever,
    candidate_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []
    for case in cases:
        query = rewrite_query(case["question"]) if config.ENABLE_QUERY_REWRITE else case["question"]
        relevant_ids = resolve_relevant_parent_ids(chunks, case)
        for chunk in retriever.retrieve(query, query_vectors[case["id"]], candidate_k):
            rows.append(feature_vector(query, chunk))
            labels.append(int(chunk.get("parent_id", chunk["id"]) in relevant_ids))
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int)


def train_model(x_train: np.ndarray, y_train: np.ndarray) -> tuple[dict, LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=7)
    clf.fit(x_scaled, y_train)
    model = {
        "kind": "logistic_regression",
        "features": FEATURE_NAMES,
        "intercept": float(clf.intercept_[0]),
        "coef": [float(value) for value in clf.coef_[0]],
        "mean": [float(value) for value in scaler.mean_],
        "scale": [float(value) for value in scaler.scale_],
    }
    return model, clf, scaler


def evaluate_cases(
    label: str,
    cases: list[dict],
    chunks: list[dict],
    query_vectors: dict[str, list[float]],
    retriever: CandidateRetriever,
    candidate_k: int,
    model: dict | None,
    learned_query_types: str = "all",
) -> tuple[dict, list[dict]]:
    details = []
    for case in cases:
        query = rewrite_query(case["question"]) if config.ENABLE_QUERY_REWRITE else case["question"]
        relevant_ids = resolve_relevant_parent_ids(chunks, case)
        started = time.perf_counter()
        results = retriever.retrieve(query, query_vectors[case["id"]], candidate_k)
        if model and should_apply_learned_ranker(query, learned_query_types):
            results = rerank_with_learned_model(query, results, model)
        elapsed_ms = (time.perf_counter() - started) * 1000
        details.append(evaluate_case(case, results[:10], relevant_ids, elapsed_ms))
    metrics = aggregate(details)
    print(label, json.dumps(metrics, ensure_ascii=False))
    return metrics, details


def diagnose(details: list[dict], chunks: list[dict], output: str | Path) -> list[dict]:
    by_source_page = {(chunk["source"], chunk["page"]): chunk for chunk in chunks}
    rows = []
    for detail in details:
        if detail["id"] != "assessment_eal" and detail["query_type"] != "process":
            continue
        row = {
            "id": detail["id"],
            "question": detail["question"],
            "query_type": detail["query_type"],
            "first_relevant_rank": detail["first_relevant_rank"],
            "candidates": [],
        }
        for item in detail["retrieved"][:5]:
            chunk = by_source_page.get((item["source"], item["page"]))
            row["candidates"].append({
                **item,
                "snippet": (chunk or {}).get("text", "")[:260].replace("\n", " | "),
            })
        rows.append(row)
    with open(output, "w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=2)
    return rows


def by_query_type(details: list[dict]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for detail in details:
        grouped[detail["query_type"]].append(detail)
    return {key: aggregate(value) for key, value in sorted(grouped.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=GOLD_PATH)
    parser.add_argument("--split", default=str(SPLIT_PATH))
    parser.add_argument("--chunks", default=config.KB_CHUNKS_PATH)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--model-output", default="data/learning_ranker_model.json")
    parser.add_argument("--report", default="data/learning_ranker_report.json")
    parser.add_argument("--diagnosis", default="data/retrieval_failure_diagnosis.json")
    parser.add_argument("--learned-query-types", default=config.LEARNED_RANKER_QUERY_TYPES)
    args = parser.parse_args()

    gold = load_json(args.gold)
    chunks = load_json(args.chunks)
    validation = validate_gold(gold, chunks)
    train_cases, dev_cases = load_split(args.split, gold)
    query_vectors = query_vectors_for(gold)
    retriever = CandidateRetriever(chunks)

    x_train, y_train = collect_training_rows(
        train_cases, chunks, query_vectors, retriever, args.candidate_k
    )
    model, _, _ = train_model(x_train, y_train)

    Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_output, "w", encoding="utf-8") as fp:
        json.dump(model, fp, ensure_ascii=False, indent=2)

    train_base, train_base_details = evaluate_cases(
        "train baseline", train_cases, chunks, query_vectors, retriever, args.candidate_k, None
    )
    train_learned, train_learned_details = evaluate_cases(
        "train learned ", train_cases, chunks, query_vectors, retriever, args.candidate_k, model
    )
    train_gated, train_gated_details = evaluate_cases(
        "train gated   ", train_cases, chunks, query_vectors, retriever,
        args.candidate_k, model, args.learned_query_types
    )
    dev_base, dev_base_details = evaluate_cases(
        "dev baseline  ", dev_cases, chunks, query_vectors, retriever, args.candidate_k, None
    )
    dev_learned, dev_learned_details = evaluate_cases(
        "dev learned   ", dev_cases, chunks, query_vectors, retriever, args.candidate_k, model
    )
    dev_gated, dev_gated_details = evaluate_cases(
        "dev gated     ", dev_cases, chunks, query_vectors, retriever,
        args.candidate_k, model, args.learned_query_types
    )
    all_base, all_base_details = evaluate_cases(
        "all baseline  ", gold, chunks, query_vectors, retriever, args.candidate_k, None
    )
    all_learned, all_learned_details = evaluate_cases(
        "all learned   ", gold, chunks, query_vectors, retriever, args.candidate_k, model
    )
    all_gated, all_gated_details = evaluate_cases(
        "all gated     ", gold, chunks, query_vectors, retriever,
        args.candidate_k, model, args.learned_query_types
    )

    diagnosis_rows = diagnose(all_gated_details, chunks, args.diagnosis)
    report = {
        "validation": validation,
        "embedding_model": embedding_label(),
        "split": {"train": len(train_cases), "dev": len(dev_cases)},
        "candidate_k": args.candidate_k,
        "learned_query_types": args.learned_query_types,
        "positive_rows": int(y_train.sum()),
        "negative_rows": int((1 - y_train).sum()),
        "model_output": args.model_output,
        "diagnosis_output": args.diagnosis,
        "metrics": {
            "train_baseline": train_base,
            "train_learned": train_learned,
            "train_gated": train_gated,
            "dev_baseline": dev_base,
            "dev_learned": dev_learned,
            "dev_gated": dev_gated,
            "all_baseline": all_base,
            "all_learned": all_learned,
            "all_gated": all_gated,
        },
        "by_query_type": {
            "dev_baseline": by_query_type(dev_base_details),
            "dev_learned": by_query_type(dev_learned_details),
            "dev_gated": by_query_type(dev_gated_details),
            "all_baseline": by_query_type(all_base_details),
            "all_learned": by_query_type(all_learned_details),
            "all_gated": by_query_type(all_gated_details),
        },
        "model": model,
        "diagnosis_cases": diagnosis_rows,
    }
    with open(args.report, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    print("模型已保存:", args.model_output)
    print("报告已保存:", args.report)
    print("诊断已保存:", args.diagnosis)


if __name__ == "__main__":
    main()
