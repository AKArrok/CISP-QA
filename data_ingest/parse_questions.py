"""解析 CISP 真题 → data/questions.json（结构化题库）+ data/parse_failures.json

三个来源:
1. cisp试题700道(带答案).pdf — 表格布局，两个题库段（题号各自从 1 开始），
   通过 PyMuPDF find_tables 按行提取；跨页续行合并到上一题
2. CISP216.docx — 与 700题PDF 第二段同源，带"知识点"解析但无答案 → 按题号
   回填解析（题干前缀校验防止错位）
3. 51CTO 六套题 docx — 答案字母跟在题干段末尾

知识域打标: 关键词规则（domains.classify_domain_by_keywords）；
--llm-classify 开启后对规则未命中的题目用轻量 LLM 批量兜底分类。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from domains import DOMAINS, classify_domain_by_keywords, normalize_for_match

NUM_RE = re.compile(r"^\s*(\d{1,3})\s*[.．、\\]\s*")
OPT_RE = re.compile(r"^\s*([A-F])\s*[.．、]\s*(.*)$", re.S)
OPT_CELL_RE = re.compile(r"^\s*[A-F]\s*[.．、]\s*")   # 700题PDF 选项单元格自带字母前缀
ANSWER_RE = re.compile(r"\s*([A-D]{1,4})\s*$")
KNOWN_RE = re.compile(r"^\s*(知识点|解析|考点|答案)\s*[：:]")


def new_question(source: str, num: int) -> dict:
    return {
        "id": f"{source}_{num}",
        "source": source,
        "num": num,
        "stem": "",
        "options": {},
        "answer": "",
        "analysis": "",
        "domain": None,
        "needs_review": False,
    }


def clean(text: str) -> str:
    return re.sub(r"[ \t\u3000]+", " ", text or "").strip()


# ────────────────────────────────────────────────────────────────────────
# 1. 700 题 PDF（表格）
# ────────────────────────────────────────────────────────────────────────

def parse_pdf700() -> list[dict]:
    import pymupdf

    doc = pymupdf.open(os.path.join(config.EXAM_DIR, "cisp试题700道(带答案).pdf"))
    rows = []
    for page in doc:
        for table in page.find_tables().tables:
            rows.extend(table.extract())

    questions: list[dict] = []
    bank = 1
    failures: list[dict] = []
    for r in rows:
        cells = [clean(c).replace("\n", "") for c in r]  # 表格单元格内换行是折行，直接拼回
        stem, opts, answer = cells[0], cells[1:5], cells[5] if len(cells) > 5 else ""
        if stem == "题目内容":          # 表头
            continue
        m = NUM_RE.match(stem)
        if m:
            num = int(m.group(1))
            q = new_question(f"700题_b{bank}", num)
            q["stem"] = stem
            for letter, text in zip("ABCD", opts, strict=False):
                if text:
                    q["options"][letter] = OPT_CELL_RE.sub("", text)
            q["answer"] = answer if re.fullmatch(r"[A-D]{1,4}", answer) else ""
            if not q["answer"]:
                q["needs_review"] = True
                failures.append({"source": q["id"], "reason": "answer_missing_or_invalid", "raw": cells})
            questions.append(q)
        elif questions:                 # 跨页续行：按列拼回上一题
            prev = questions[-1]
            prev["stem"] += stem
            for letter, text in zip("ABCD", opts, strict=False):
                if text and letter in prev["options"]:
                    prev["options"][letter] += text
                elif text:
                    prev["options"][letter] = text
        else:
            failures.append({"source": "700题", "reason": "orphan_row", "raw": cells})

    # 题号 545 之后是第二段（题号重启）：上面 bank 逻辑没用上，这里按序号重置点重打 id
    return questions, failures


def split_banks(questions: list[dict]) -> list[dict]:
    """按题号重起点把 700 题拆成两段并重写 id/source。"""
    out = []
    bank = 1
    prev_num = 0
    for q in questions:
        if q["num"] < prev_num:
            bank += 1
        prev_num = q["num"]
        q["source"] = f"700题_b{bank}"
        q["id"] = f"700题_b{bank}_{q['num']}"
        out.append(q)
    return out


# ────────────────────────────────────────────────────────────────────────
# 2. CISP216.docx（知识点解析回填）
# ────────────────────────────────────────────────────────────────────────

def parse_cisp216() -> tuple[list[dict], list[str]]:
    import docx

    path = os.path.join(config.EXAM_DIR, "CISP216.docx")
    doc = docx.Document(path)
    questions, failures = [], []
    cur = None
    for p in doc.paragraphs:
        text = clean(p.text)
        if not text:
            continue
        if KNOWN_RE.match(text):
            if cur is not None:
                cur["analysis"] += ("\n" if cur["analysis"] else "") + text
            continue
        om = OPT_RE.match(text)
        if om and cur is not None:
            cur["options"][om.group(1)] = clean(om.group(2))
            continue
        nm = NUM_RE.match(text)
        if nm:
            cur = new_question("CISP216", int(nm.group(1)))
            cur["stem"] = text
            questions.append(cur)
            continue
        # 其它行：既不是题号/选项/知识点 → 视为上一条解析的续行
        if cur is not None and cur["analysis"]:
            cur["analysis"] += text
        elif cur is not None:
            cur["stem"] += text
    return questions, failures


def merge_analysis(bank2: list[dict], cisp216: list[dict]) -> None:
    """CISP216 的"知识点"按题号回填到 700题_b2，题干前缀校验防错位。"""
    by_num = {q["num"]: q for q in bank2}
    for q in cisp216:
        target = by_num.get(q["num"])
        if target is None:
            continue
        a, b = normalize_for_match(target["stem"])[:15], normalize_for_match(q["stem"])[:15]
        if a[:10] == b[:10]:
            if not target["analysis"]:
                target["analysis"] = q["analysis"]
        else:
            target.setdefault("_merge_conflicts", []).append(q["num"])


# ────────────────────────────────────────────────────────────────────────
# 3. 51CTO 六套 docx（答案跟在题干末尾）
# ────────────────────────────────────────────────────────────────────────

def parse_51cto_sets() -> tuple[list[dict], list[dict]]:
    """51CTO 六套题，答案有三种形态:
    - 一/二套: 答案字母紧跟题干末尾（'...：C'）
    - 二/三/四/五套: 文末答案对照表（'72\\tD' 整段）
    - 六套: 压缩答案串穿插正文（'1C2A3D4B5C' 每5题一批）
    """
    questions, failures = [], []
    files = sorted(
        f for f in os.listdir(config.EXAM_DIR)
        if f.startswith("51CTO下载-CISP试题及答案") and f.endswith(".docx")
    )
    import docx

    key_line_re = re.compile(r"^\s*(\d{1,3})\s*[\t\.、,，:： ]\s*([A-D]{1,4})\s*$")
    key_compact_re = re.compile(r"^\s*(?:\d{1,3}[A-D]{1,4}){2,}\s*$")

    def parse_compact(text: str) -> dict[int, str]:
        keys = {}
        for num, ans in re.findall(r"(\d{1,3})([A-D]{1,4})", text):
            keys[int(num)] = ans
        return keys

    for fname in files:
        set_name = re.search(r"(一|二|三|四|五|六)套题", fname)
        source = f"51CTO_{set_name.group(1)}套" if set_name else fname
        doc = docx.Document(os.path.join(config.EXAM_DIR, fname))
        cur = None
        key_map: dict[int, str] = {}
        for p in doc.paragraphs:
            text = clean(p.text)
            if not text:
                continue
            # 答案对照表行 / 压缩答案串
            km = key_line_re.match(text)
            if km and (cur is None or len(cur["options"]) >= 2):
                key_map[int(km.group(1))] = km.group(2)
                continue
            if key_compact_re.match(text):
                key_map.update(parse_compact(text))
                continue
            om = OPT_RE.match(text)
            if om and cur is not None:
                cur["options"][om.group(1)] = clean(om.group(2))
                continue
            nm = NUM_RE.match(text)
            if nm and (cur is None or len(cur["options"]) >= 2):
                cur = new_question(source, int(nm.group(1)))
                cur["stem"] = text
                questions.append(cur)
                continue
            if cur is not None:
                cur["stem"] += text  # 题干续行
        # 回填答案: 先题干内联，再对照表/压缩串（回填成功时清掉待复核标记）
        extract_inline_answers([q for q in questions if q["source"] == source])
        for q in questions:
            if q["source"] == source and not q["answer"] and q["num"] in key_map:
                q["answer"] = key_map[q["num"]]
                q["needs_review"] = False
        failures.extend(failures_of_incomplete(questions, source))
    return questions, failures


def failures_of_incomplete(questions: list[dict], source: str) -> list[dict]:
    out = []
    for q in questions:
        if q["source"] != source:
            continue
        if len(q["options"]) < 2 or not q["stem"]:
            q["needs_review"] = True
            out.append({"source": q["id"], "reason": "incomplete_options", "stem": q["stem"][:60]})
    return out


def extract_inline_answers(questions: list[dict]) -> None:
    """51CTO 格式：答案字母紧跟在题干末尾（'...：C'），拆出来。"""
    for q in questions:
        if q["answer"]:
            continue
        m = ANSWER_RE.search(q["stem"])
        if m and len(q["options"]) >= 2:
            q["answer"] = m.group(1)
            q["stem"] = q["stem"][: m.start()].rstrip("：: ，,")
        else:
            q["needs_review"] = True


# ────────────────────────────────────────────────────────────────────────
# 4. 去重 / 打标 / 自检
# ────────────────────────────────────────────────────────────────────────

def dedupe(questions: list[dict]) -> tuple[list[dict], int]:
    """跨来源按题干前缀去重：后出现的重复题标记 dup_of（保留先出现的，其来源题库优先级更高）。"""
    seen: dict[str, str] = {}
    dups = 0
    for q in questions:
        key = normalize_for_match(q["stem"])[:40]
        if key in seen:
            q["dup_of"] = seen[key]
            dups += 1
        else:
            seen[key] = q["id"]
    return questions, dups


def classify_domains(questions: list[dict], use_llm: bool) -> None:
    unclassified = []
    for q in questions:
        text = q["stem"] + " " + " ".join(q["options"].values()) + " " + q["analysis"]
        q["domain"] = classify_domain_by_keywords(text)
        if q["domain"] is None and not q.get("dup_of"):
            unclassified.append(q)
    if not use_llm or not unclassified:
        return
    from pydantic import BaseModel
    from llms import get_simple_llm, invoke_structured

    class Batch(BaseModel):
        labels: list[str]  # 与输入题目一一对应，取值必须是十大知识域之一或 "未知"

    llm = get_simple_llm()
    for i in range(0, len(unclassified), 20):
        batch = unclassified[i:i + 20]
        lines = "\n".join(f"{j}. {q['stem'][:120]}" for j, q in enumerate(batch))
        msg = [
            {"role": "system", "content":
                "你是 CISP 考试题分类器。十大知识域: " + "、".join(DOMAINS) +
                "。对每道题判断其所属知识域。只输出 JSON。"},
            {"role": "user", "content": f"题目列表:\n{lines}"},
        ]
        try:
            result = invoke_structured(llm, Batch, msg)
            for q, label in zip(batch, result.labels, strict=False):
                if label in DOMAINS:
                    q["domain"] = label
        except Exception as e:
            print(f"[警告] LLM 分类批次失败: {e}")
            break

    # 逐题重试仍缺域的（批量 zip 错位/输出"未知"的漏网之鱼）
    class Single(BaseModel):
        domain: str

    still = [q for q in unclassified if q["domain"] is None]
    for q in still:
        try:
            out = invoke_structured(llm, Single, [
                {"role": "system", "content":
                    "你是 CISP 考试题分类器。判断该题所属的十大知识域之一: "
                    + "、".join(DOMAINS) + "。只输出 JSON。"},
                {"role": "user", "content": q["stem"][:200]},
            ])
            if out.domain in DOMAINS:
                q["domain"] = out.domain
        except Exception:
            continue


def self_check(questions: list[dict]) -> dict:
    stats = {"total": len(questions)}
    by_source: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    no_answer = no_domain = needs_review = dups = 0
    for q in questions:
        by_source[q["source"]] = by_source.get(q["source"], 0) + 1
        by_domain[q["domain"] or "未分类"] = by_domain.get(q["domain"] or "未分类", 0) + 1
        if not q["answer"]:
            no_answer += 1
        if not q["domain"]:
            no_domain += 1
        # 答案字母超出实际选项 / 知识域缺失 → 不可用于出题
        if q["answer"] and not set(q["answer"]) <= set(q["options"]):
            q["needs_review"] = True
        if q["domain"] is None:
            q["needs_review"] = True
        if q.get("needs_review"):
            needs_review += 1
        if q.get("dup_of"):
            dups += 1
    stats.update(by_source=by_source, by_domain=by_domain, no_answer=no_answer,
                 no_domain=no_domain, needs_review=needs_review, duplicates=dups)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-classify", action="store_true",
                    help="对关键词规则未命中的题目用轻量 LLM 兜底分类")
    args = ap.parse_args()

    all_questions: list[dict] = []
    failures: list[dict] = []

    # 1) 700 题 PDF → 两段题库
    pdf_qs, pdf_failures = parse_pdf700()
    pdf_qs = split_banks(pdf_qs)
    print(f"700题PDF: {len(pdf_qs)} 题（b1={sum(1 for q in pdf_qs if q['source']=='700题_b1')}, "
          f"b2={sum(1 for q in pdf_qs if q['source']=='700题_b2')}）")
    all_questions.extend(pdf_qs)
    failures.extend(pdf_failures)

    # 2) CISP216 知识点回填到 b2
    cisp_qs, _ = parse_cisp216()
    bank2 = [q for q in pdf_qs if q["source"] == "700题_b2"]
    merge_analysis(bank2, cisp_qs)
    merged = sum(1 for q in bank2 if q["analysis"])
    print(f"CISP216.docx: {len(cisp_qs)} 题，回填解析 {merged} 题")

    # 3) 51CTO 六套
    set_qs, set_failures = parse_51cto_sets()
    print(f"51CTO 六套: {len(set_qs)} 题")
    all_questions.extend(set_qs)
    failures.extend(set_failures)

    # 4) 去重 → 打标 → 自检
    all_questions, dups = dedupe(all_questions)
    classify_domains(all_questions, use_llm=args.llm_classify)
    stats = self_check(all_questions)

    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(config.QUESTIONS_PATH, "w", encoding="utf-8") as fp:
        json.dump(all_questions, fp, ensure_ascii=False, indent=1)
    with open(config.PARSE_FAILURES_PATH, "w", encoding="utf-8") as fp:
        json.dump(failures, fp, ensure_ascii=False, indent=1)
    from data_ingest.manifest import update_manifest, QUESTIONS_SCHEMA_VERSION
    update_manifest(question_count=len(all_questions), questions_schema_version=QUESTIONS_SCHEMA_VERSION)

    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"解析失败/待复核 {len(failures)} 条 → {config.PARSE_FAILURES_PATH}")
    print(f"题库 {len(all_questions)} 题 → {config.QUESTIONS_PATH}")


if __name__ == "__main__":
    main()
