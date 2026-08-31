"""Configuration — all settings from environment variables (CISP-QA, trimmed from AniRAG)."""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ── LLM 通用配置（OpenAI 兼容协议）──
LLM_API_KEY  = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-pro")
SIMPLE_LLM_MODEL = os.getenv("SIMPLE_LLM_MODEL", "deepseek-v4-flash")

# ── Embeddings ────────────────────────────────────────────────────────
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "dashscope")
# 百炼 Model Studio：qwen3.7-text-embedding 走 OpenAI 兼容接口。
# 若未直接配置完整 URL，则由 Workspace ID 和 region 组装。
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_WORKSPACE_ID = os.getenv("DASHSCOPE_WORKSPACE_ID", "")
DASHSCOPE_REGION = os.getenv("DASHSCOPE_REGION", "cn-beijing")
_DASHSCOPE_DEFAULT_BASE_URL = (
    f"https://{DASHSCOPE_WORKSPACE_ID}.{DASHSCOPE_REGION}.maas.aliyuncs.com/compatible-mode/v1"
    if DASHSCOPE_WORKSPACE_ID else ""
)
DASHSCOPE_EMBEDDING_BASE_URL = os.getenv(
    "DASHSCOPE_EMBEDDING_BASE_URL", _DASHSCOPE_DEFAULT_BASE_URL
)
DASHSCOPE_EMBEDDING_MODEL = os.getenv(
    "DASHSCOPE_EMBEDDING_MODEL", "qwen3.7-text-embedding"
)
DASHSCOPE_EMBEDDING_DIMENSIONS = int(os.getenv("DASHSCOPE_EMBEDDING_DIMENSIONS", "1024"))
DASHSCOPE_EMBEDDING_REQUEST_INTERVAL = float(
    os.getenv("DASHSCOPE_EMBEDDING_REQUEST_INTERVAL", "0.1")
)
DASHSCOPE_EMBEDDING_MAX_RETRIES = int(os.getenv("DASHSCOPE_EMBEDDING_MAX_RETRIES", "4"))
DASHSCOPE_EMBEDDING_MAX_BACKOFF = float(os.getenv("DASHSCOPE_EMBEDDING_MAX_BACKOFF", "30"))

# 兼容保留：旧 Ark Coding Plan 配置。
ARK_API_KEY = os.getenv("ARK_API_KEY", "")
ARK_EMBEDDING_API_KEY = os.getenv("ARK_EMBEDDING_API_KEY") or ARK_API_KEY
ARK_EMBEDDING_BASE_URL = os.getenv("ARK_EMBEDDING_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3")
ARK_EMBEDDING_MODEL = os.getenv("ARK_EMBEDDING_MODEL", "doubao-embedding-vision")
ARK_EMBEDDING_DIMENSIONS = int(os.getenv("ARK_EMBEDDING_DIMENSIONS", "1024"))
ARK_EMBEDDING_REQUEST_INTERVAL = float(os.getenv("ARK_EMBEDDING_REQUEST_INTERVAL", "3.0"))
ARK_EMBEDDING_MAX_RETRIES = int(os.getenv("ARK_EMBEDDING_MAX_RETRIES", "6"))
ARK_EMBEDDING_MAX_BACKOFF = float(os.getenv("ARK_EMBEDDING_MAX_BACKOFF", "30"))
# 本地模型（HuggingFace sentence-transformers，EMBEDDING_BACKEND=local 时生效）
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
_LOCAL_EMBEDDING_DEVICE_RAW = os.getenv("LOCAL_EMBEDDING_DEVICE", "cpu")


def _resolve_embedding_device() -> str:
    """auto 时自动检测 CUDA，不可用回退 cpu。"""
    if _LOCAL_EMBEDDING_DEVICE_RAW != "auto":
        return _LOCAL_EMBEDDING_DEVICE_RAW
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


LOCAL_EMBEDDING_DEVICE = _resolve_embedding_device()
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "")
if HF_ENDPOINT:
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# ── 本地数据路径 ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
KB_CHUNKS_PATH = os.path.join(DATA_DIR, "kb_chunks.json")
QUESTIONS_PATH = os.path.join(DATA_DIR, "questions.json")
PARSE_FAILURES_PATH = os.path.join(DATA_DIR, "parse_failures.json")
VECTORS_PATH = os.path.join(DATA_DIR, "vectors.npz")
DB_PATH = os.path.join(DATA_DIR, "cisp_qa.db")

# ── 存储层（MySQL 主后端 + SQLite 降级；Redis 缓存可选）──
MYSQL_URL_OVERRIDE = os.getenv("DATABASE_URL", "")   # 例: mysql+pymysql://user:pwd@127.0.0.1/cisp_qa?charset=utf8mb4
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
ANSWER_CACHE_TTL = int(os.getenv("ANSWER_CACHE_TTL", "3600"))
RETRIEVAL_CACHE_TTL = int(os.getenv("RETRIEVAL_CACHE_TTL", "600"))
STATS_CACHE_TTL = int(os.getenv("STATS_CACHE_TTL", "30"))
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))

# ── 检索参数 ──
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))
DENSE_K = int(os.getenv("DENSE_K", "20"))
SPARSE_K = int(os.getenv("SPARSE_K", "20"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1200"))
SUMMARY_CHUNK_MAX_CHARS = int(os.getenv("SUMMARY_CHUNK_MAX_CHARS", "800"))
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "structured")
CHUNK_TARGET_CHARS = int(os.getenv("CHUNK_TARGET_CHARS", "220"))
STRUCTURED_CHUNK_MAX_CHARS = int(os.getenv("STRUCTURED_CHUNK_MAX_CHARS", "420"))
CHUNK_OVERLAP_UNITS = int(os.getenv("CHUNK_OVERLAP_UNITS", "1"))
# 方案 B（定长+字符重叠）对比基线参数
FIXED_CHUNK_TARGET_CHARS = int(os.getenv("FIXED_CHUNK_TARGET_CHARS", "300"))
FIXED_CHUNK_MAX_CHARS = int(os.getenv("FIXED_CHUNK_MAX_CHARS", "500"))
FIXED_CHUNK_OVERLAP_CHARS = int(os.getenv("FIXED_CHUNK_OVERLAP_CHARS", "100"))
# 精排（交叉编码重排序）：CPU 单次数秒，关闭后保持 RRF 融合顺序
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "10"))
LOCAL_RERANKER_MODEL = os.getenv("LOCAL_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# ── 出题 ──
QUIZ_AI_MIN_BANK = int(os.getenv("QUIZ_AI_MIN_BANK", "5"))   # 域内未做题少于该数时允许 AI 补题
# ── 用户画像（掌握度模型）──
# Beta 后验：掌握度=均值，置信度=样本量带来的区间收敛；薄弱= P(能力<阈值)>概率
MASTERY_WEAK_THRESHOLD = float(os.getenv("MASTERY_WEAK_THRESHOLD", "0.6"))   # 能力低于该值判薄弱
MASTERY_WEAK_PROB = float(os.getenv("MASTERY_WEAK_PROB", "0.7"))             # 后验概率阈值
BETA_PRIOR = float(os.getenv("BETA_PRIOR", "1.0"))                           # Beta 先验（伪计数）
# 遗忘曲线：记忆强度 = 2^(-天数/半衰期)，超过半衰期进入复习队列
MASTERY_HALF_LIFE_DAYS = float(os.getenv("MASTERY_HALF_LIFE_DAYS", "7"))
REVIEW_DUE_STRENGTH = float(os.getenv("REVIEW_DUE_STRENGTH", "0.5"))         # 强度低于该值判到期
# Elo 评级（Pelánek 2016）：按时间顺序 replay attempts 增量更新
ELO_INIT = float(os.getenv("ELO_INIT", "1500"))
ELO_K = float(os.getenv("ELO_K", "32"))

# ── 多轮记忆 ──
MEMORY_MAX_ROUNDS = int(os.getenv("MEMORY_MAX_ROUNDS", "5"))

# ── 原始资料路径（只读）──
COURSE_DIR = os.getenv(
    "CISP_COURSE_DIR",
    r"D:\Users\ASUS\Desktop\学习资料\网安\#5021 CISP注册信息安全专业人员-东方瑞通网站课程资料",
)
EXAM_DIR = os.getenv("CISP_EXAM_DIR", r"D:\Users\ASUS\Desktop\学习资料\网安\CSIP试题")


def validate() -> None:
    missing = [k for k, v in {
        "LLM_API_KEY": LLM_API_KEY,
    }.items() if not v]
    if EMBEDDING_BACKEND == "dashscope" and not DASHSCOPE_API_KEY:
        missing.append("DASHSCOPE_API_KEY")
    if EMBEDDING_BACKEND == "dashscope" and not DASHSCOPE_EMBEDDING_BASE_URL:
        missing.append("DASHSCOPE_WORKSPACE_ID or DASHSCOPE_EMBEDDING_BASE_URL")
    if EMBEDDING_BACKEND == "ark" and not ARK_EMBEDDING_API_KEY:
        missing.append("ARK_EMBEDDING_API_KEY")
    if missing:
        raise EnvironmentError(f"Missing env vars: {missing}. Copy AniRAG/.env keys into .env")
    if EMBEDDING_BACKEND == "ark" and ARK_EMBEDDING_MODEL != "doubao-embedding-vision":
        raise ValueError("Ark Coding Plan embedding model must be doubao-embedding-vision")
    if (EMBEDDING_BACKEND == "dashscope"
            and DASHSCOPE_EMBEDDING_DIMENSIONS not in {256, 512, 768, 1024, 1536, 2048, 2560}):
        raise ValueError("qwen3.7-text-embedding dimensions must be 256/512/768/1024/1536/2048/2560")
