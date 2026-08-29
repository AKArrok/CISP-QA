# CISP-QA — CISP 备考问答机器人

基于本地课程资料的 RAG 问答 + 真题刷题 + 薄弱点画像。改造自本人的 AniRAG 项目（LangGraph 多 Agent 架构），详见 `docs/`。

## 功能

- **问答**：基于 10 大知识域课件/知识点总结混合检索（本地向量 + BM25 → RRF），流式回答并标注课件出处；支持多轮追问
- **刷题**：1267 道结构化真题（700题PDF + CISP216 + 51CTO六套），随机练习 / 薄弱强化两种模式；真题不足时 AI 参照课件生成补充题
- **画像**：SQLite 记录答题历史，按知识域统计正确率，答题 ≥3 且正确率 <60% 判为薄弱域，反哺"薄弱强化"出题
- **长期记忆**：每轮对话落盘 SQLite（`conversations` 表），重启不丢；`GET /api/history?thread_id=` / `GET /api/threads` 查询，供会话恢复

## 快速开始

```bash
pip install -r requirements.txt

# 1. 解析原始资料（课件 → 知识块；真题 → 题库）
python data_ingest/parse_courses.py
python data_ingest/parse_questions.py --llm-classify

# 2. 构建向量索引
python data_ingest/build_index.py

# 3. 启动 Web 面板
python server.py        # → http://localhost:9528
```

## 测试

```bash
pytest tests/test_parsers.py tests/test_grader.py -q   # 无 API 依赖
python tests/eval_retrieval.py                          # 检索评测（需索引）
```

## 配置（.env）

| 变量 | 说明 |
|------|------|
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | DeepSeek 问答与路由 |
| `EMBEDDING_BACKEND` | `local`（Qwen3-Embedding-0.6B，当前默认）/ `ark`（doubao-embedding-vision，需 Coding Plan 订阅有效） |
| `CISP_COURSE_DIR` / `CISP_EXAM_DIR` | 原始资料路径 |

## 结构

```
data_ingest/   解析与建库        retrieval/  本地向量库 + BM25 + RRF
agents/        LangGraph 问答    quiz/       抽题/生成/判分/画像
server.py      FastAPI+SSE       static/     双页签面板
docs/          选型/PRD/架构     tests/      自检与评测
```
