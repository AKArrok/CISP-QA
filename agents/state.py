"""AgentState — LangGraph 图共享状态。"""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    # 会话线程 id（长期记忆落盘用）
    thread_id: str
    # 消息历史（多轮记忆，MemorySaver 按 thread_id 持久化）
    messages: Annotated[list[AnyMessage], add_messages]
    # 原始用户问题
    question: str
    # 意图: knowledge(知识问答) | quiz(想做题) | chitchat(闲聊)
    intent: Literal["knowledge", "quiz", "chitchat"]
    # 改写后的独立检索查询（追问消解后）
    search_query: str
    # 检索到的知识块
    contexts: list[dict]
    # 最终回答
    answer: str
