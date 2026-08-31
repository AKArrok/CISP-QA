# CISP-QA — CISP 备考问答机器人

基于本地课程资料的 RAG 问答 + 真题刷题 + 薄弱点画像。改造自本人的 AniRAG 项目（LangGraph 多 Agent 架构），详见 `docs/`。

## 功能

- **问答**：基于 10 大知识域课件/知识点总结的混合检索（本地向量 + BM25 → RRF → **bge-reranker 交叉编码精排**），流式回答并标注课件出处；支持多轮追问（指代消解）
- **刷题**：1267 道结构化真题（700题PDF + CISP216 + 51CTO六套），随机练习 / 薄弱强化两种模式；真题不足时 AI 参照课件生成补充题
- **画像**：SQLite 记录答题历史，按知识域统计正确率，答题 ≥3 且正确率 <60% 判为薄弱域，反哺"薄弱强化"出题
- **长期记忆**：每轮对话落盘 SQLite（`conversations` 表），重启不丢；`GET /api/history?thread_id=` / `GET /api/threads` 查询，供会话恢复
- **可观测性**：请求级指标采集（路由/检索/首 token/总耗时 + Token 用量）落 SQLite，`GET /api/metrics` 聚合分位数
- **数据一致性自检**：`data/manifest.json` 版本清单，解析产物与向量索引条数/版本不符时启动即报错并给出修复指令
- **存储层**（仿 Dify 架构）：SQLAlchemy 2.0 单一建模 + Alembic 迁移，`DATABASE_URL` 驱动——MySQL(utf8mb4) 主后端，连接失败自动降级 SQLite；题库/知识块/答题记录/对话/指标全部入库
- **Redis 缓存层**：问答缓存（首轮问题 TTL 1h）、检索结果缓存、画像统计缓存（答题即失效）、固定窗口限流；Redis 不可达自动降级进程内存

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

## 测试与评测

```bash
pytest tests/test_parsers.py tests/test_grader.py tests/test_memory.py -q  # 单测（CI 同款）
python tests/eval_retrieval.py    # 检索评测：top5/top1 知识域命中率
python tests/compare_chunking.py  # 三方案切片 A/B/C 对比（legacy/fixed/structured）
python tests/eval_intents.py      # 意图路由/拒答行为回归（需 LLM API）
```

## Docker 部署

```bash
docker build -t cisp-qa .
docker run -p 9528:9528 --env-file .env \
  -v $(pwd)/data:/app/data -v cisp_hf:/app/.hf_cache cisp-qa
# 首次启动会下载 embedding/精排模型到 .hf_cache 卷；国内构建时可加
# --build-arg 或在 Dockerfile 中设置 HF_ENDPOINT=https://hf-mirror.com
```


## 存储层切换

`.env` 填 `MYSQL_PASSWORD`（或 `DATABASE_URL=mysql+pymysql://user:pwd@host/cisp_qa?charset=utf8mb4`）即用 MySQL；
留空自动降级 SQLite，行为一致。建库导入：

```bash
python scripts/import_to_db.py            # JSON 解析产物 → 数据库（幂等，按 id upsert）
python scripts/import_to_db.py --reset    # 清空 questions/kb_chunks 后重导
python scripts/import_to_db.py --legacy   # 迁移旧 SQLite 的历史答题/对话数据
alembic upgrade head                      # 或用迁移管理 schema
```

## 配置（.env）

| 变量 | 说明 |
|------|------|
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | DeepSeek 问答与路由 |
| `EMBEDDING_BACKEND` | `dashscope`（默认，qwen3.7-text-embedding）/ `local`（Qwen3-Embedding-0.6B）/ `ark`（doubao-embedding-vision，需 Coding Plan 订阅有效） |
| `DASHSCOPE_API_KEY` / `DASHSCOPE_EMBEDDING_BASE_URL` / `DASHSCOPE_EMBEDDING_MODEL` | 百炼 embedding（OpenAI 兼容接口，默认模型 `qwen3.7-text-embedding`，1024 维） |
| `CHUNK_STRATEGY` | `structured`（默认：语义 Child 检索、完整 Parent 回答）/ `fixed`（定长+字符重叠）/ `legacy`（旧按页切片，仅用于 A/B/C 对比） |
| `CISP_COURSE_DIR` / `CISP_EXAM_DIR` | 原始资料路径 |

## 结构

```
data_ingest/   解析与建库        retrieval/  本地向量库 + BM25 + RRF
agents/        LangGraph 问答    quiz/       抽题/生成/判分/画像
server.py      FastAPI+SSE       static/     双页签面板
docs/          选型/PRD/架构     tests/      自检与评测
```
