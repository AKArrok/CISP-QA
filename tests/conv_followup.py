# -*- coding: utf-8 -*-
"""追问改写验证：确认指代消解后的独立检索查询。"""
import asyncio
import sys

sys.path.insert(0, ".")

from agents.graph import run


async def main():
    qs = [
        "什么是业务影响分析BIA？",
        "它和RTO是什么关系？",           # 指代：它=BIA
    ]
    for q in qs:
        print("=" * 72)
        print("【用户】", q)
        r = await run(q, thread_id="conv-followup")
        print("[改写后的检索查询]", r.get("search_query", ""))
        print("[意图=%s]" % r["intent"])
        print("【助手】", r.get("answer", "")[:350])


asyncio.run(main())
