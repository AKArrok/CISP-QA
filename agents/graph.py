"""LangGraph 主图: route → rewrite → retrieve → answer。

裁剪自 AniRAG 的多 Agent 流水线：CISP 问答以单点知识问答为主，
去掉 Expert 并行/评估重规划/别名/联网回退，保留「路由 → 追问改写 → 混合检索 → 流式回答」。
"""
from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

import config
from agents.state import AgentState
from agents import prompts
from llms import get_answer_llm, get_simple_llm, llm_ainvoke_with_retry, invoke_structured

logger = logging.getLogger(__name__)


# ── 节点 ────────────────────────────────────────────────────────────────

class RouteOut(BaseModel):
    intent: str  # knowledge | quiz | chitchat


async def route(state: AgentState) -> dict:
    """意图路由（轻量 LLM + 结构化输出）。同步 OpenAI 调用放线程池，避免阻塞事件循环。"""
    question = state["question"]
    simple = get_simple_llm()
    try:
        out = await asyncio.to_thread(
            invoke_structured, simple, RouteOut,
            [SystemMessage(content=prompts.ROUTE_SYSTEM), HumanMessage(content=question)],
        )
        intent = out.intent if out.intent in ("knowledge", "quiz", "chitchat") else "knowledge"
    except Exception:
        logger.exception("路由失败，默认走知识问答")
        intent = "knowledge"
    return {"intent": intent}


async def chitchat(state: AgentState) -> dict:
    from langchain_core.messages import AIMessage
    reply = "你好！我是 CISP 备考助手。可以直接问我知识点（比如：什么是深度防御），也可以去「刷题」页签练习真题。"
    # contexts 置空：防止上一轮的检索结果残留在会话状态里
    return {"messages": [AIMessage(content=reply)], "answer": reply,
            "intent": "chitchat", "contexts": []}


async def quiz_hint(state: AgentState) -> dict:
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=prompts.QUIZ_HINT)], "answer": prompts.QUIZ_HINT,
            "contexts": []}


async def persist(state: AgentState) -> dict:
    """长期记忆：本轮（用户问题 + 最终回答）落盘 SQLite。"""
    from agents import memory

    answer = state.get("answer", "")
    if answer:
        try:
            memory.save_round(state.get("thread_id", "default"),
                              state["question"], answer, state["intent"])
        except Exception:
            logger.exception("对话落盘失败（不影响回答）")
    return {}


async def rewrite(state: AgentState) -> dict:
    """追问改写：结合历史把问题改写为独立检索查询。"""
    history = state.get("messages", [])[:-1]  # 去掉刚追加的本轮问题
    question = state["question"]
    search_query = question
    if history:
        simple = get_simple_llm()
        try:
            hist_text = "\n".join(
                f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {str(m.content)[:200]}"
                for m in history[-4:]
            )
            out = await llm_ainvoke_with_retry(simple, [
                SystemMessage(content=prompts.REWRITE_SYSTEM),
                HumanMessage(content=f"对话历史:\n{hist_text}\n\n当前问题: {question}"),
            ])
            rewritten = str(out.content).strip()
            if rewritten and len(rewritten) <= 200:
                search_query = rewritten
        except Exception:
            logger.exception("追问改写失败，使用原始问题")
    return {"search_query": search_query}


async def retrieve(state: AgentState) -> dict:
    from retrieval.hybrid import HybridRetriever

    # 本地 CPU embedding + 检索是同步重计算，放线程池避免阻塞事件循环
    contexts = await asyncio.to_thread(
        HybridRetriever.get().retrieve, state["search_query"], config.RETRIEVER_K)
    return {"contexts": contexts}


async def answer(state: AgentState) -> dict:
    contexts = state.get("contexts", [])
    if not contexts:
        return {
            "messages": [],
            "answer": "知识库中未覆盖该问题。你可以换个问法，或先在「刷题」页签练习对应知识域的题目。",
        }
    context_text = "\n\n".join(
        f"<chunk 来源《{c['source']}》第{c['page']}页 知识域[{c['domain']}]>\n{c['text']}\n</chunk>"
        for c in contexts
    )
    # 传入最近几轮对话背景，供回答时理解指代（回答内容仍以资料为准）
    history = state.get("messages", [])[:-1]
    history_text = ""
    if history:
        lines = "\n".join(
            f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {str(m.content)[:300]}"
            for m in history[-4:]
        )
        history_text = f"\n\n对话背景（仅用于理解指代，回答内容仍必须依据资料）:\n{lines}"
    llm = get_answer_llm()
    messages = [
        SystemMessage(content=prompts.ANSWER_SYSTEM),
        HumanMessage(content=prompts.ANSWER_CONTEXT_TEMPLATE.format(
            contexts=context_text, question=state["question"]) + history_text),
    ]
    resp = await llm_ainvoke_retry(llm, messages)
    usage_meta = getattr(resp, "usage_metadata", None) or {}
    usage = {
        "prompt_tokens": usage_meta.get("input_tokens", 0),
        "completion_tokens": usage_meta.get("output_tokens", 0),
    }
    return {"messages": [resp], "answer": str(resp.content), "usage": usage}


async def llm_ainvoke_retry(llm, messages):
    from llms import llm_ainvoke_with_retry
    return await llm_ainvoke_with_retry(llm, messages)


# ── 路由函数 ────────────────────────────────────────────────────────────

def route_after_intent(state: AgentState) -> str:
    if state["intent"] == "chitchat":
        return "chitchat"
    if state["intent"] == "quiz":
        return "quiz_hint"
    return "rewrite"


# ── 图构建 ──────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("route", route)
    g.add_node("chitchat", chitchat)
    g.add_node("quiz_hint", quiz_hint)
    g.add_node("rewrite", rewrite)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)
    g.add_node("persist", persist)

    g.add_edge(START, "route")
    g.add_conditional_edges(
        "route", route_after_intent,
        {"chitchat": "chitchat", "quiz_hint": "quiz_hint", "rewrite": "rewrite"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "answer")
    # 三条出口统一经 persist 落盘长期记忆再结束
    g.add_edge("answer", "persist")
    g.add_edge("chitchat", "persist")
    g.add_edge("quiz_hint", "persist")
    g.add_edge("persist", END)
    return g


class SessionStore:
    """同一 thread_id 共享 MemorySaver 与已编译图实例（移植 AniRAG 模式）。"""

    _instances: dict[str, tuple] = {}

    @classmethod
    def get(cls, thread_id: str):
        if thread_id not in cls._instances:
            memory = MemorySaver()
            app = build_graph().compile(checkpointer=memory)
            cls._instances[thread_id] = (app, thread_id)
        return cls._instances[thread_id][0]


async def run(question: str, thread_id: str = "default") -> dict:
    """一次性执行（CLI/测试用）。server.py 用 astream_events 做流式。"""
    app = SessionStore.get(thread_id)
    result = await app.ainvoke(
        {"messages": [HumanMessage(content=question)], "question": question,
         "thread_id": thread_id},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "什么是深度防御？"
    result = asyncio.run(run(q, thread_id="cli"))
    print(f"[意图] {result['intent']}")
    print(f"[检索] {len(result.get('contexts', []))} 块")
    print(result.get("answer", ""))
