"""课件结构化切片：语义 Child 用于检索，完整 Parent 用于回答。"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from data_ingest.tables import split_table_markdown


_BULLET_RE = re.compile(
    r"^(?:[•·●○◆◇■□▪▫]|[-—]|\d+(?:\.\d+)*[、.．)]|"
    r"[（(][一二三四五六七八九十\d]+[）)]|[一二三四五六七八九十]+、)"
)
_PAGE_NUMBER_RE = re.compile(r"^(?:第\s*)?\d{1,4}(?:\s*页)?$")
_SENTENCE_END_RE = re.compile(r"[。！？!?；;：:]$")
_VERSION_RE = re.compile(r"[Vv]?(\d+(?:\.\d+)+)")


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_version(source: str) -> str:
    """从文件名提取版本号（业务连续性_V4.2 → V4.2）。"""
    match = _VERSION_RE.search(source)
    return match.group(0) if match else ""


def strip_repeated_lines(page_texts: list[str]) -> list[str]:
    """删除文档内高频短页眉/页脚和纯页码行。"""
    normalized_pages = [clean_text(text).splitlines() for text in page_texts]
    counts: Counter[str] = Counter()
    for lines in normalized_pages:
        counts.update({line.strip() for line in lines if line.strip()})

    threshold = max(3, math.ceil(len(normalized_pages) * 0.5))
    repeated = {
        line for line, count in counts.items()
        if count >= threshold and len(line) <= 60
    }
    cleaned = []
    for lines in normalized_pages:
        kept = [
            line for line in lines
            if line.strip() not in repeated and not _PAGE_NUMBER_RE.fullmatch(line.strip())
        ]
        cleaned.append(clean_text("\n".join(kept)))
    return cleaned


def split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = re.findall(r"[^。！？!?；;\n]+[。！？!?；;\n]?", text)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                parts.append(current.strip())
                current = ""
            parts.extend(
                sentence[i:i + max_chars].strip()
                for i in range(0, len(sentence), max_chars)
                if sentence[i:i + max_chars].strip()
            )
        elif len(current) + len(sentence) <= max_chars:
            current += sentence
        else:
            parts.append(current.strip())
            current = sentence
    if current.strip():
        parts.append(current.strip())
    return parts


def _looks_like_title(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 45 or _BULLET_RE.match(line):
        return False
    if _SENTENCE_END_RE.search(line):
        return False
    return bool(
        re.match(r"^第[一二三四五六七八九十\d]+[章节部分]", line)
        or re.match(r"^\d+(?:\.\d+){1,3}\s*\S+", line)
        or len(line) <= 24
    )


def semantic_units(text: str) -> tuple[str, list[str]]:
    """按标题、项目符号和段落边界切成尽量完整的语义单元。"""
    lines = [line.strip() for line in clean_text(text).splitlines() if line.strip()]
    if not lines:
        return "", []

    title = lines[0] if _looks_like_title(lines[0]) else ""
    if title:
        lines = lines[1:]

    units: list[str] = []
    current = ""
    for line in lines:
        starts_unit = bool(_BULLET_RE.match(line) or _looks_like_title(line))
        if current and (starts_unit or (_SENTENCE_END_RE.search(current) and len(current) >= 80)):
            units.append(current.strip())
            current = line
        elif current:
            current += "\n" + line
        else:
            current = line
    if current:
        units.append(current.strip())
    # 整页仅一个标题时：保留标题（供跨页继承），不产生内容单元。
    return title, units


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def _pack_children(
    units: list[str], target_chars: int, max_chars: int, overlap_units: int
) -> list[str]:
    expanded = [part for unit in units for part in split_long(unit, max_chars)]
    if not expanded:
        return []

    groups: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for unit in expanded:
        added = len(unit) + (1 if current else 0)
        if current and current_len + added > target_chars:
            groups.append(current)
            current = current[-overlap_units:] if overlap_units else []
            current_len = sum(map(len, current)) + max(0, len(current) - 1)
        current.append(unit)
        current_len += len(unit) + (1 if len(current) > 1 else 0)
    if current:
        groups.append(current)

    # 避免最后产生极短碎片；合并后仍由 max_chars 硬切兜底。
    min_tail_chars = min(80, max(20, target_chars // 2))
    if len(groups) > 1 and len("\n".join(groups[-1])) < min_tail_chars:
        tail = groups.pop()
        groups[-1].extend(tail[overlap_units:] if overlap_units else tail)

    return [part for group in groups for part in split_long("\n".join(group), max_chars)]


def _table_block(table: dict) -> str:
    """表格在 Parent 文本中的呈现：表头标记 + 表题/引导语 + markdown。"""
    caption = (table.get("caption") or "").strip()
    header = f"【表格】{caption}" if caption else "【表格】"
    return f"{header}\n{table['markdown']}"


def _table_children(table: dict, max_chars: int) -> list[str]:
    """表格 Child：注入【表格】标记与表题语境；超长按数据行切分并重复表头行。"""
    caption = (table.get("caption") or "").strip()
    header = f"【表格】{caption}" if caption else "【表格】"
    markdown = table["markdown"]
    if len(header) + 1 + len(markdown) <= max_chars:
        return [f"{header}\n{markdown}"]
    budget = max(max_chars - len(header) - 1, 40)
    parts = split_table_markdown(markdown, markdown.splitlines()[0], budget)
    children = [f"{header}\n{part}" for part in parts]
    # 单行超预算的极端情况由 split_long 兜底硬切
    return [piece for child in children for piece in split_long(child, max_chars)]


def build_structured_chunks(
    pages: list[dict], target_chars: int = 220, max_chars: int = 420,
    overlap_units: int = 1,
) -> list[dict]:
    """从同一文档的页面生成结构化父子 Chunk；合格表格额外产出独立表格 Child。

    表格页的 Parent = 正文 + markdown 表格（回答时表格与正文一起送 LLM）；
    表格 Child 带表题语境独立参与检索，与正文 Child 共享同一 parent_id，
    召回任一即可还原"正文+表格"完整上下文。无表格页面行为与旧版完全一致。
    """
    base_texts = [
        page["body_text"] if page.get("tables") else page["text"]
        for page in pages
    ]
    cleaned_pages = strip_repeated_lines(base_texts)
    chunks: list[dict] = []
    last_title = ""
    for page, cleaned_body in zip(pages, cleaned_pages, strict=True):
        tables = page.get("tables") or []
        parent_parts = [cleaned_body] + [_table_block(table) for table in tables]
        parent_text = "\n\n".join(part for part in parent_parts if part.strip())
        if len(parent_text) < 10:
            continue
        detected_title, units = semantic_units(cleaned_body)
        if detected_title:
            last_title = detected_title
        title = detected_title or last_title  # 续页（无标题）继承上一页标题，补足检索上下文
        children: list[tuple[str, str]] = [
            (child, "text")
            for child in _pack_children(units, target_chars, max_chars, overlap_units)
        ]
        if not children and not tables:
            continue  # 纯标题页不产生 Chunk，仅用于标题继承
        for table in tables:
            children.extend((child, "table") for child in _table_children(table, max_chars))
        parent_id = _stable_id(
            "parent", page["domain"], page["source"], page["page"], parent_text
        )
        for index, (child_text, element) in enumerate(children):
            embedding_text = "\n".join(filter(None, [
                f"知识域：{page['domain']}",
                f"知识点：{title}" if title else "",
                child_text,
            ]))
            chunks.append({
                "id": _stable_id("chunk", parent_id, index, child_text),
                "parent_id": parent_id,
                "domain": page["domain"],
                "source": page["source"],
                "version": extract_version(page["source"]),
                "page": page["page"],
                "page_end": page["page"],
                "kind": page["kind"],
                "title": title,
                "text": parent_text,
                "child_text": child_text,
                "embedding_text": embedding_text,
                "element": element,
                "chunking_strategy": "structured_v1",
            })
    return chunks


def build_fixed_chunks(
    pages: list[dict], target_chars: int = 300, max_chars: int = 500,
    overlap_chars: int = 100,
) -> list[dict]:
    """固定长度滑窗切片（B 基线）：不感知文档结构，仅按字符长度切分 + 字符重叠。"""
    cleaned_pages = strip_repeated_lines([page["text"] for page in pages])
    chunks: list[dict] = []
    for page, parent_text in zip(pages, cleaned_pages, strict=True):
        if len(parent_text) < 10:
            continue
        title, _ = semantic_units(parent_text)
        start = 0
        length = len(parent_text)
        while start < length:
            end = min(start + target_chars, length)
            if end < length:
                # 就近句末截断，避免硬切到半句话；仍是纯字符策略
                for sep in ("。", "！", "？", "\n"):
                    cut = parent_text.rfind(sep, start, end)
                    if cut >= start + target_chars * 0.6:
                        end = cut + 1
                        break
            part = parent_text[start:end].strip()
            if part:
                chunk_id = _stable_id("chunk", page["source"], page["page"], start, part)
                chunks.append({
                    "id": chunk_id,
                    "parent_id": chunk_id,  # 固定切片无父子结构，自身即父
                    "domain": page["domain"],
                    "source": page["source"],
                    "version": extract_version(page["source"]),
                    "page": page["page"],
                    "page_end": page["page"],
                    "kind": page["kind"],
                    "title": title,
                    "text": part,
                    "child_text": part,
                    "embedding_text": "\n".join(filter(None, [
                        f"知识域：{page['domain']}",
                        f"知识点：{title}" if title else "",
                        part,
                    ])),
                    "chunking_strategy": "fixed_v1",
                })
            next_start = end - overlap_chars
            if next_start <= start:
                break
            start = next_start
    return chunks


def deduplicate_chunks(chunks: list[dict]) -> tuple[list[dict], int]:
    """删除跨版本的规范化精确重复 Child，优先保留总结和 V4.2。"""
    def priority(chunk: dict) -> tuple[int, int]:
        source = chunk["source"].lower()
        return (
            3 if chunk["kind"] == "summary" else 2 if "v4.2" in source else 1,
            len(chunk["child_text"]),
        )

    best: dict[str, dict] = {}
    passthrough: list[dict] = []
    for chunk in chunks:
        normalized = re.sub(r"\W+", "", chunk["child_text"].lower())
        if len(normalized) < 6:
            passthrough.append(chunk)
            continue
        existing = best.get(normalized)
        if existing is None or priority(chunk) > priority(existing):
            best[normalized] = chunk
    result = passthrough + list(best.values())
    result.sort(key=lambda c: (c["domain"], c["source"], c["page"], c["id"]))
    return result, len(chunks) - len(result)
