"""轻量查询改写：补足课件检索需要的同义词和章节提示。"""
from __future__ import annotations

import re


_DATE_INTENT = re.compile(r"(什么时候|何时|时间|日期|哪天|开始|实施|施行|生效)")


def rewrite_query(query: str) -> str:
    normalized = query.strip()
    additions: list[str] = []

    if "网络安全法" in normalized and _DATE_INTENT.search(normalized):
        additions.extend(["实施", "施行", "生效", "实施时间", "实施日期", "网络运行安全"])
    if "CRL" in normalized.upper() or "撤销" in normalized:
        additions.extend(["证书撤销列表", "证书库", "已撤销证书"])
    if "BIA" in normalized.upper() or "业务影响分析" in normalized:
        additions.extend(["灾难恢复需求分析", "恢复目标", "关键业务功能"])
    if "EAL" in normalized.upper() and ("划分" in normalized or "评估保证级" in normalized):
        additions.extend(["EAL1", "EAL2", "EAL3", "EAL4", "EAL5", "EAL6", "EAL7", "保护轮廓", "安全目标"])

    if not additions:
        return normalized
    existing = set(re.findall(r"[\w\u4e00-\u9fff]+", normalized.lower()))
    extra = [term for term in additions if term.lower() not in existing]
    return " ".join([normalized, *extra])
