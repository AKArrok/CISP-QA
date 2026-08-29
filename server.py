"""FastAPI 服务: SSE 流式问答 + 刷题接口 + 薄弱点统计 + 静态面板。

启动: python server.py → http://localhost:9528
"""
from __future__ import annotations

import json
import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

import config
from agents.graph import SessionStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="CISP 备考问答机器人")


# ── 问答（SSE 流式）────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    thread_id: str = "default"


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    app_graph = SessionStore.get(req.thread_id)
    config_ = {"configurable": {"thread_id": req.thread_id}}

    async def event_gen():
        # 阶段计时（可观测性）: 路由 / 检索 / 首 token / 总耗时
        t0 = time.perf_counter()
        stamps = {"route_ms": None, "retrieval_ms": None, "first_token_ms": None}
        intent_seen = None
        sent_tokens = 0
        async for ev in app_graph.astream_events(
            {"messages": [HumanMessage(content=req.query)], "question": req.query,
             "thread_id": req.thread_id},
            config=config_, version="v2",
        ):
            kind = ev["event"]
            node = (ev.get("metadata") or {}).get("langgraph_node")
            now = time.perf_counter()
            if kind == "on_chain_end" and node == "route":
                out = ev["data"]["output"]
                intent = out.get("intent") if isinstance(out, dict) else getattr(out, "intent", None)
                if intent and intent_seen is None:
                    intent_seen = intent
                    stamps["route_ms"] = round((now - t0) * 1000, 1)
                yield sse({"type": "intent", "intent": intent})
            elif kind == "on_chain_end" and node == "retrieve":
                out = ev["data"]["output"]
                contexts = out.get("contexts", []) if isinstance(out, dict) else []
                stamps["retrieval_ms"] = round((now - t0) * 1000, 1)
                yield sse({"type": "contexts", "contexts": [
                    {"source": c["source"], "page": c["page"], "domain": c["domain"],
                     "text": c["text"][:120]}
                    for c in contexts
                ]})
            elif kind == "on_chat_model_stream" and node == "answer":
                chunk = ev["data"]["chunk"]
                if chunk.content:
                    if stamps["first_token_ms"] is None:
                        stamps["first_token_ms"] = round((now - t0) * 1000, 1)
                    sent_tokens += 1
                    yield sse({"type": "token", "content": str(chunk.content)})
        # 闲聊/做题引导不走 answer 节点、无 token 流：补发完整回复，避免前端空白气泡
        total_ms = round((time.perf_counter() - t0) * 1000, 1)
        final = await app_graph.aget_state(config_)
        answer = (final.values or {}).get("answer", "")
        usage = (final.values or {}).get("usage") or {}
        if sent_tokens == 0 and answer:
            yield sse({"type": "token", "content": answer})
        try:
            from agents.metrics import record
            record(req.thread_id, req.query, intent_seen,
                   stamps["route_ms"], stamps["retrieval_ms"],
                   stamps["first_token_ms"], total_ms,
                   usage.get("prompt_tokens"), usage.get("completion_tokens"))
        except Exception:
            logging.exception("指标写入失败（不影响回答）")
        yield sse({"type": "done"})

    return StreamingResponse(event_gen(), media_type="text/event-stream")


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# ── 刷题 ────────────────────────────────────────────────────────────────

class QuizNextRequest(BaseModel):
    mode: str = "random"          # random | weak
    domain: str | None = None


class QuizAnswerRequest(BaseModel):
    question_id: str
    choice: str


@app.post("/quiz/next")
def quiz_next(req: QuizNextRequest):
    from quiz import get_next_question
    try:
        return get_next_question(mode=req.mode, domain=req.domain)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.post("/quiz/answer")
def quiz_answer(req: QuizAnswerRequest):
    from quiz import submit_answer
    try:
        return submit_answer(req.question_id, req.choice)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/api/stats")
def api_stats():
    from quiz import get_stats
    return get_stats()


@app.get("/api/domains")
def api_domains():
    from domains import DOMAINS
    return DOMAINS


@app.get("/api/history")
def api_history(thread_id: str, limit: int = 50):
    """某会话的落盘历史（重启后仍可查，供会话恢复/续聊）。"""
    from agents.memory import load_history
    return load_history(thread_id, limit=limit)


@app.get("/api/threads")
def api_threads(limit: int = 20):
    """历史会话列表（最近优先）。"""
    from agents.memory import list_threads
    return list_threads(limit=limit)


@app.get("/api/metrics")
def api_metrics():
    """请求级指标聚合：阶段耗时分位数、Token 用量、意图分布。"""
    from agents.metrics import summary
    return summary()


# ── 静态面板 ────────────────────────────────────────────────────────────

app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    config.validate()
    from data_ingest.manifest import validate_index
    validate_index()  # 启动自检：数据资产齐全且与 manifest 一致
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9528, log_level="info")
