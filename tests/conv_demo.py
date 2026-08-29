# -*- coding: utf-8 -*-
"""多轮对话演示：随机问题混合测试（知识问答/追问/闲聊/做题/知识库外）。"""
import asyncio
import sys

sys.path.insert(0, ".")

from agents.graph import run

QUESTIONS = [
    "防火墙和入侵检测系统IDS有什么区别？",      # 知识问答
    "那部署的时候一般放在网络的什么位置？",      # 追问（指代消解）
    "今天天气怎么样？",                        # 闲聊
    "给我出几道题练练手",                      # 想做题
    "CISP考试报名费多少钱？在哪里报名？",       # 考试相关但知识库外
    "用Python写一个快速排序",                  # 完全无关
]


async def main():
    for q in QUESTIONS:
        print("=" * 72)
        print("【用户】", q)
        r = await run(q, thread_id="conv-demo")
        ctxs = r.get("contexts", [])
        if ctxs:
            doms = ["%s·%s p%s" % (c["domain"], c["source"], c["page"]) for c in ctxs[:3]]
            print("[意图=%s | 检索%d块: %s]" % (r["intent"], len(ctxs), "、".join(doms)))
        else:
            print("[意图=%s | 无检索]" % r["intent"])
        print("【助手】", r.get("answer", "(空)")[:700])


asyncio.run(main())
