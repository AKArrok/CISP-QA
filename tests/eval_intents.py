# -*- coding: utf-8 -*-
"""意图路由与拒答行为回归评测（golden set）。

改 prompt / 换模型后跑一遍，防止路由和拒答行为静默劣化。
需要 LLM API。运行: python tests/eval_intents.py
"""
from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, ".")

from agents.graph import run

# (问题, 期望意图, 期望行为)
# 行为: "refuse_or_general" = 不编造资料内容（含未覆盖声明/通用补充/闲聊引导均可）
#        "normal"           = 正常知识回答
GOLDEN = [
    ("什么是深度防御？", "knowledge", "normal"),
    ("PKI 的证书作废怎么查？", "knowledge", "normal"),
    ("那它和 RTO 有什么区别？", "knowledge", "normal"),          # 无上文指代也稳定路由
    ("给我出十道题", "quiz", None),
    ("考考我信息安全管理的知识点", "quiz", None),
    ("你好", "chitchat", None),
    ("谢谢啦", "chitchat", None),
    ("今天天气怎么样？", "chitchat", None),
    ("CISP 报名费多少钱？", "knowledge", "refuse_or_general"),   # 考试相关但资料外
    ("用 Python 写个爬虫", "chitchat", None),                    # 完全无关
    ("明天股市开盘吗？", "chitchat", None),
    ("帮我写周报", "chitchat", None),
]


async def main() -> None:
    passed = 0
    for question, want_intent, behavior in GOLDEN:
        r = await run(question, thread_id="eval-intents")
        got_intent = r["intent"]
        answer = r.get("answer", "")
        intent_ok = got_intent == want_intent
        behavior_ok = True
        if behavior == "refuse_or_general":
            behavior_ok = ("未覆盖" in answer or "通用补充" in answer)
        ok = intent_ok and behavior_ok
        passed += ok
        mark = "✅" if ok else "❌"
        detail = "" if intent_ok else f"  期望意图={want_intent}"
        detail += "" if behavior_ok else "  期望拒答/通用补充，实际疑似编造"
        print(f"{mark} [{got_intent}] {question[:30]}{detail}")
    print(f"\n通过 {passed}/{len(GOLDEN)}")


if __name__ == "__main__":
    asyncio.run(main())
