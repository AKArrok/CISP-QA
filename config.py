"""Configuration — all settings from environment variables (CISP-QA, trimmed from AniRAG)."""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ── LLM 通用配置（OpenAI 兼容协议）──
LLM_API_KEY  = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-pro")
SIMPLE_LLM_MODEL = os.getenv("SIMPLE_LLM_MODEL", "deepseek-v4-flash")

# ── Embeddings（Ark Coding Plan, doubao-embedding-vision）──
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "ark")
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

# ── 检索参数 ──
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))
DENSE_K = int(os.getenv("DENSE_K", "20"))
SPARSE_K = int(os.getenv("SPARSE_K", "20"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1200"))

# ── 出题 ──
QUIZ_AI_MIN_BANK = int(os.getenv("QUIZ_AI_MIN_BANK", "5"))   # 域内未做题少于该数时允许 AI 补题
WEAK_DOMAIN_MIN_ATTEMPTS = 3
WEAK_DOMAIN_ACCURACY = 0.6

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
    if EMBEDDING_BACKEND == "ark" and not ARK_EMBEDDING_API_KEY:
        missing.append("ARK_EMBEDDING_API_KEY")
    if missing:
        raise EnvironmentError(f"Missing env vars: {missing}. Copy AniRAG/.env keys into .env")
    if EMBEDDING_BACKEND == "ark" and ARK_EMBEDDING_MODEL != "doubao-embedding-vision":
        raise ValueError("Ark Coding Plan embedding model must be doubao-embedding-vision")
