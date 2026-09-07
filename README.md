# CISP-QA — CISP 备考问答机器人

基于本地课程资料的 RAG 问答 + 真题刷题 + 薄弱点画像，采用基于 LangGraph 的多阶段 RAG 流水线，详见 `docs/`。

## 功能

- **问答**：基于 10 大知识域课件/知识点总结的混合检索（DashScope flash 向量 + BM25 → RRF → 特征重排 → text-rerank 分数融合精排；embedding API 不可用时自动降级 BM25 兜底召回，远端向量与精排分数带磁盘缓存避免重复计费），流式回答并标注课件出处；支持多轮追问（指代消解）
- **刷题**：1267 道结构化真题（700题PDF + CISP216 + 51CTO六套），随机练习 / 薄弱强化两种模式；真题不足时 AI 参照课件生成补充题
- **画像**：SQL 记录答题历史，按知识域统计正确率，答题 ≥3 且正确率 <60% 判为薄弱域，反哺"薄弱强化"出题
- **记忆**：短期上下文保存在 LangGraph `MemorySaver` + Redis/内存缓存；长期记忆把每轮对话 embedding 后写入 `data/memory_vectors.npz`，并通过元数据支持 `GET /api/history?thread_id=` / `GET /api/threads`
- **可观测性**：请求级指标采集（路由/检索/首 token/总耗时 + Token 用量）落 SQL，`GET /api/metrics` 聚合分位数
- **数据一致性自检**：`data/manifest.json` 版本清单，解析产物与课件向量索引条数/版本不符时启动即报错并给出修复指令
- **存储层**（仿 Dify 架构）：SQLAlchemy 2.0 单一建模 + Alembic 迁移，`DATABASE_URL` 驱动——MySQL(utf8mb4) 主后端，连接失败自动降级 SQLite；题库/知识块/答题记录/指标入库，用户长期记忆使用独立向量文件
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
python tests/eval_retrieval.py                 # 旧检索 smoke：top5/top1 知识域命中率
python tests/eval_retrieval_gold.py --no-rerank # 知识点级 gold 检索评测（需 embedding API）
python tests/eval_retrieval_gold.py --gold tests/fixtures/retrieval_holdout.json --no-rerank # 独立 holdout 检索评测
python tests/eval_retrieval_gold.py --rerank    # 知识点级 gold 检索评测 + reranker
python tests/tune_retrieval_gold.py             # 基于 gold 集对 DENSE_K/SPARSE_K/RRF_K 做小网格调参
python tests/train_learning_ranker.py           # 固定 train/dev split 上训练并评估轻量学习排序
python tests/compare_chunking.py               # 三方案切片 A/B/C 对比（legacy/fixed/structured）
python tests/eval_intents.py                   # 意图路由/拒答行为回归（需 LLM API）
```

检索准确性看 `docs/05-检索测试集评测.md`；`tests/eval_retrieval.py` 只适合做回归烟测，不能证明知识点召回。

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
| `DENSE_K` / `SPARSE_K` / `RRF_K` / `ENABLE_QUERY_REWRITE` / `ENABLE_FEATURE_RERANKING` | 检索候选池与轻量重排参数；默认 `10 / 20 / 20 / true / true`，由知识点级 gold 评测确定 |
| `ENABLE_LEARNED_RANKING` / `LEARNED_RANKER_PATH` / `LEARNED_RANKER_QUERY_TYPES` | 轻量学习排序实验开关；默认关闭，当前模型全开会误伤 acronym/definition，门控默认 `process` |
| `ENABLE_RERANKING` / `RERANK_TOP_K` | 可选本地精排；默认关闭，当前 gold 评测显示会增加延迟且可能误伤 Top5 |
| `CHUNK_STRATEGY` | `structured`（默认：语义 Child 检索、完整 Parent 回答）/ `fixed`（定长+字符重叠）/ `legacy`（旧按页切片，仅用于 A/B/C 对比） |
| `CISP_COURSE_DIR` / `CISP_EXAM_DIR` | 原始资料路径 |
| `SHORT_MEMORY_TTL` / `LONG_MEMORY_TOP_K` | 短期上下文缓存 TTL、长期向量记忆召回条数 |

## 结构

```
data_ingest/   解析与建库        retrieval/  本地向量库 + BM25 + RRF
agents/        LangGraph 问答    quiz/       抽题/生成/判分/画像
server.py      FastAPI+SSE       static/     双页签面板
docs/          选型/PRD/架构     tests/      自检与评测
```
