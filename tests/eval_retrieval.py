"""检索评测：20 条样例问题（覆盖 10 知识域），核对 top5 是否命中正确知识域。

需要已建索引。运行: python tests/eval_retrieval.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

from retrieval.hybrid import HybridRetriever

# (问题, 期望知识域)
EVAL_SET = [
    ("什么是深度防御？", "信息安全保障"),
    ("信息系统安全保障评估框架关注哪些特点？", "信息安全保障"),
    ("我国网络安全等级保护分为几个等级？", "信息安全监管"),
    ("《网络安全法》是什么时候实施的？", "信息安全监管"),
    ("AES 支持哪些密钥长度？", "信息安全支撑技术"),
    ("PKI 中确认证书是否作废查看什么？", "信息安全支撑技术"),
    ("防火墙主要工作在网络体系结构的哪一层？", "物理与网络通信安全"),
    ("机房供配电系统有哪些安全要求？", "物理与网络通信安全"),
    ("操作系统常用的强制访问控制模型有哪些？", "计算环境安全"),
    ("恶意代码的主要传播途径有哪些？", "计算环境安全"),
    ("软件安全开发生命周期包括哪些阶段？", "软件安全开发"),
    ("代码审计主要发现什么问题？", "软件安全开发"),
    ("风险评估的实施过程包括哪些步骤？", "信息安全评估"),
    ("CC 准则的评估保证级 EAL 如何划分？", "信息安全评估"),
    ("什么是业务影响分析 BIA？", "业务连续性"),
    ("RTO 和 RPO 的区别是什么？", "业务连续性"),
    ("信息安全管理体系 ISMS 的 PDCA 循环是什么？", "信息安全管理"),
    ("资产识别和风险评估的关系是什么？", "信息安全管理"),
    ("入侵检测系统 IDS 的部署方式有哪些？", "安全工程与运营"),
    ("应急响应的流程包括哪些环节？", "安全工程与运营"),
]


def main() -> None:
    retriever = HybridRetriever.get()
    hits_top1 = hits_top5 = 0
    for question, expected in EVAL_SET:
        results = retriever.retrieve(question, top_k=5)
        got_domains = [r["domain"] for r in results]
        ok5 = expected in got_domains
        ok1 = bool(got_domains) and got_domains[0] == expected
        hits_top5 += ok5
        hits_top1 += ok1
        mark = "✅" if ok5 else "❌"
        flag = "" if ok1 or not ok5 else "  (top5命中但top1偏差)"
        print(f"{mark} [{expected}] {question}{flag}")
        if not ok5:
            print(f"   实际: {got_domains}")
    print(f"\ntop5 知识域命中: {hits_top5}/{len(EVAL_SET)}")
    print(f"top1 知识域命中: {hits_top1}/{len(EVAL_SET)}  ← 精排主要改善该指标")


if __name__ == "__main__":
    main()
