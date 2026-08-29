"""解析 CISP 课件（V4.2 PDF + 知识点总结 PDF + V4.1 PPTX）→ data/kb_chunks.json

chunk 结构: {id, domain, source, page, kind, text}
- 课件 PDF 按页分块（页是天然语义单元，带页码可溯源），单页超限再按句切分
- 知识点总结 PDF 按页再按条目细分
- PPTX 按页(slide)提取标题+正文
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from domains import FILENAME_DOMAINS


def domain_from_filename(name: str) -> str:
    for key, domain in FILENAME_DOMAINS.items():
        if key in name:
            return domain
    return "未分类"


def _clean(text: str) -> str:
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_long(text: str, max_chars: int) -> list[str]:
    """超过 max_chars 的页按句边界切分（移植 AniRAG chunking 思路）。"""
    if len(text) <= max_chars:
        return [text]
    sentences = re.findall(r"[^。！？!?；;\n]+[。！？!?；;\n]?", text)
    parts, current = [], ""
    for s in sentences:
        if len(s) > max_chars:
            if current:
                parts.append(current)
                current = ""
            parts.extend(s[i:i + max_chars] for i in range(0, len(s), max_chars))
        elif len(current) + len(s) <= max_chars:
            current += s
        else:
            parts.append(current)
            current = s
    if current:
        parts.append(current)
    return parts


def parse_pdf(path: str, kind: str, max_chars: int) -> list[dict]:
    import pymupdf

    domain = domain_from_filename(os.path.basename(path))
    source = os.path.splitext(os.path.basename(path))[0]
    chunks = []
    doc = pymupdf.open(path)
    for page_no, page in enumerate(doc, start=1):
        text = _clean(page.get_text())
        if len(text) < 10:
            continue
        for part in _split_long(text, max_chars):
            chunks.append({
                "domain": domain,
                "source": source,
                "page": page_no,
                "kind": kind,
                "text": part,
            })
    doc.close()
    return chunks


def parse_pptx(path: str) -> list[dict]:
    from pptx import Presentation

    domain = domain_from_filename(os.path.basename(path))
    source = os.path.splitext(os.path.basename(path))[0]
    chunks = []
    prs = Presentation(path)
    for slide_no, slide in enumerate(prs.slides, start=1):
        texts = []
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for para in shape.text_frame.paragraphs:
                line = "".join(run.text for run in para.runs).strip()
                if line:
                    texts.append(line)
        text = _clean("\n".join(texts))
        if len(text) < 10:
            continue
        for part in _split_long(text, config.CHUNK_MAX_CHARS):
            chunks.append({
                "domain": domain,
                "source": source,
                "page": slide_no,
                "kind": "pptx",
                "text": part,
            })
    return chunks


def main() -> None:
    course_dir = config.COURSE_DIR
    exam_dir = config.EXAM_DIR
    chunks: list[dict] = []

    # 1. V4.2 课件 PDF（课件目录子目录内）
    for root, _dirs, files in os.walk(course_dir):
        for f in files:
            if f.lower().endswith(".pdf") and "知识点总结" not in f:
                chunks.extend(parse_pdf(os.path.join(root, f), "courseware", config.CHUNK_MAX_CHARS))

    # 2. V4.1 PPTX 课件（存放在试题目录内）
    for f in os.listdir(exam_dir):
        if f.lower().endswith(".pptx"):
            chunks.extend(parse_pptx(os.path.join(exam_dir, f)))

    # 3. 知识点总结 PDF（课件目录内，kind=summary，作为兜底精炼语料）
    for root, _dirs, files in os.walk(course_dir):
        for f in files:
            if f.lower().endswith(".pdf") and "知识点总结" in f:
                chunks.extend(parse_pdf(os.path.join(root, f), "summary", 800))

    # 分配稳定 id
    for i, c in enumerate(chunks):
        c["id"] = f"chunk_{i:05d}"

    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(config.KB_CHUNKS_PATH, "w", encoding="utf-8") as fp:
        json.dump(chunks, fp, ensure_ascii=False, indent=1)
    from data_ingest.manifest import update_manifest, KB_SCHEMA_VERSION
    update_manifest(chunk_count=len(chunks), kb_schema_version=KB_SCHEMA_VERSION)

    # 统计
    by_domain: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for c in chunks:
        by_domain[c["domain"]] = by_domain.get(c["domain"], 0) + 1
        by_kind[c["kind"]] = by_kind.get(c["kind"], 0) + 1
    print(f"共 {len(chunks)} 个知识块 → {config.KB_CHUNKS_PATH}")
    print("按知识域:", json.dumps(by_domain, ensure_ascii=False))
    print("按类型:", json.dumps(by_kind, ensure_ascii=False))
    empty_domains = [d for d in by_domain if d == "未分类"]
    if empty_domains:
        print("[警告] 存在未分类文件，请检查 FILENAME_DOMAINS 映射")


if __name__ == "__main__":
    main()
