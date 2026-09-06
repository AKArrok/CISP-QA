# -*- coding: utf-8 -*-
"""回归测试组：把各评测脚本串成档位，对照基线阈值判定静默劣化。

档位:
  daily  — 单元测试 + holdout/gold 检索 + 问法鲁棒性 + 虚拟学生（零 API 消耗，纯缓存/本地）
  weekly — daily + 回答层质量 + 多轮追问（消耗 DeepSeek，约 1 元/次）

用法:
  python scripts/regression_battery.py --tier daily
  python scripts/regression_battery.py --tier weekly
报告: data/regression_reports/regression_<tier>_<date>.json
退出码: 有回归=1，全绿=0
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "data" / "regression_reports"

# 基线阈值：跌破即判回归（取各文档基线留 1-2pp 余量）
THRESHOLDS = {
    "holdout_recall_at_1": 0.90,      # 基线 92%（live 特征序口径）
    "gold_recall_at_1": 0.88,         # 基线 90%
    "robustness_typo_recall_at_1": 0.78,   # 基线 83.3%
    "robustness_colloquial_recall_at_1": 0.85,  # 基线 90%
    "answer_faithfulness_avg": 1.70,  # 基线 1.79
    "answer_completeness_avg": 1.95,  # 基线 1.99
    "answer_citation_support_avg": 1.75,  # 基线 1.87
    "multiturn_hit_rate": 0.85,       # 基线 100%（变体融合后 10/10），LLM 方差留 1 题余量
}

DAILY_STAGES = ["unit", "holdout", "gold", "robustness", "students"]
WEEKLY_STAGES = DAILY_STAGES + ["answer_quality", "multiturn"]


def _run(cmd: list[str], timeout: int = 3600) -> tuple[int, str]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "ENABLE_RERANKING": "false"}
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          timeout=timeout, env=env)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def run_stage(name: str) -> dict:
    out: dict = {"stage": name}
    if name == "unit":
        code, log = _run([sys.executable, "-m", "pytest", "tests/", "-q",
                          "-p", "no:cacheprovider",
                          "--ignore=tests/eval_retrieval_gold.py",
                          "--ignore=tests/eval_table_ab.py",
                          "--ignore=tests/train_learning_ranker.py",
                          "--ignore=tests/tune_retrieval_gold.py",
                          "--ignore=tests/eval_answer_quality.py",
                          "--ignore=tests/tune_rerank_fusion.py",
                          "--ignore=tests/eval_query_robustness.py",
                          "--ignore=tests/simulate_students.py",
                          "--ignore=tests/eval_multiturn.py",
                          "--basetemp", str(ROOT / "data" / "pytest_regression")])
        out["ok"] = code == 0
        out["detail"] = log.strip().splitlines()[-1] if log.strip() else ""
    elif name == "holdout":
        path = REPORT_DIR / "holdout.json"
        code, _ = _run([sys.executable, "tests/eval_retrieval_gold.py",
                        "--gold", "tests/fixtures/retrieval_holdout.json",
                        "--no-rerank", "--output", str(path)])
        out["ok"] = code == 0
        if path.exists():
            overall = _load_json(path)["overall"]
            out["recall_at_1"] = overall["recall_at_1"]
            out["recall_at_5"] = overall["recall_at_5"]
    elif name == "gold":
        path = REPORT_DIR / "gold.json"
        code, _ = _run([sys.executable, "tests/eval_retrieval_gold.py",
                        "--no-rerank", "--output", str(path)])
        out["ok"] = code == 0
        if path.exists():
            overall = _load_json(path)["overall"]
            out["recall_at_1"] = overall["recall_at_1"]
            out["recall_at_5"] = overall["recall_at_5"]
    elif name == "robustness":
        code, _ = _run([sys.executable, "tests/eval_query_robustness.py"])
        out["ok"] = code == 0
        path = ROOT / "data" / "query_robustness_report.json"
        if path.exists():
            rows = _load_json(path)["rows"]
            for style in ("typo", "colloquial", "terse"):
                style_rows = [r for r in rows if r["style"] == style]
                if style_rows:
                    out[f"{style}_recall_at_1"] = round(
                        sum(r["recall_at_1"] for r in style_rows) / len(style_rows), 4)
    elif name == "students":
        code, log = _run([sys.executable, "tests/simulate_students.py"])
        out["ok"] = code == 0
        out["detail"] = log.strip().splitlines()[-2] if log.strip() else ""
    elif name == "answer_quality":
        path = REPORT_DIR / "answer_quality.json"
        code, _ = _run([sys.executable, "tests/eval_answer_quality.py",
                        "--output", str(path)], timeout=5400)
        out["ok"] = code == 0
        if path.exists():
            summary = _load_json(path)["summary"]
            out["faithfulness_avg"] = summary.get("faithfulness_avg")
            out["completeness_avg"] = summary.get("completeness_avg")
            out["citation_support_avg"] = summary.get("citation_support_avg")
            out["gate_rejected"] = summary.get("gate_rejected")
    elif name == "multiturn":
        code, _ = _run([sys.executable, "tests/eval_multiturn.py"], timeout=3600)
        out["ok"] = code == 0
        path = ROOT / "data" / "multiturn_report.json"
        if path.exists():
            out["hit_rate"] = _load_json(path)["summary"]["hit_rate"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["daily", "weekly"], default="daily")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stages = DAILY_STAGES if args.tier == "daily" else WEEKLY_STAGES
    results = []
    for name in stages:
        print(f"== 运行阶段: {name} ...", flush=True)
        try:
            res = run_stage(name)
        except Exception as exc:
            res = {"stage": name, "ok": False, "error": str(exc)[:200]}
        results.append(res)
        print(f"   -> {'OK' if res.get('ok') else 'FAIL'} "
              f"{json.dumps({k: v for k, v in res.items() if k not in ('stage', 'ok')}, ensure_ascii=False)}",
              flush=True)

    # 阈值判定：阈值 key 形如 "<阶段前缀>_<指标>"，剥前缀后到对应阶段结果里取值
    stage_prefix = {"holdout": "holdout", "gold": "gold", "robustness": "robustness",
                    "answer_quality": "answer", "multiturn": "multiturn"}
    regressions = []
    for stage in results:
        if not stage.get("ok"):
            regressions.append(f"{stage['stage']} 阶段运行失败")
        prefix = stage_prefix.get(stage["stage"])
        if not prefix:
            continue
        for key, threshold in THRESHOLDS.items():
            if not key.startswith(prefix + "_"):
                continue
            metric = key[len(prefix) + 1:]
            value = stage.get(metric)
            if value is None:
                regressions.append(f"{stage['stage']}.{metric} 缺失")
            elif value < threshold:
                regressions.append(f"{stage['stage']}.{metric}={value} < 阈值{threshold}")
        if not stage.get("ok"):
            regressions.append(f"{stage['stage']} 阶段运行失败")

    date = datetime.date.today().isoformat()
    report = {"date": date, "tier": args.tier,
              "rerank_enabled": os.environ.get("ENABLE_RERANKING", "true").lower() == "true",
              "results": results, "regressions": regressions}
    path = REPORT_DIR / f"regression_{args.tier}_{date}.json"
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)

    print(json.dumps({"regressions": regressions,
                      "report": str(path)}, ensure_ascii=False, indent=2))
    print("结论:", "发现回归 ✗" if regressions else "全绿 ✓")
    sys.exit(1 if regressions else 0)


if __name__ == "__main__":
    main()
