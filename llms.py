"""LLM and embedding instances — ported from AniRAG (trimmed: no Pinecone/Tavily/local models)."""
import json
import logging
import random
import time
from typing import List, Type, TypeVar

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import APIError, APITimeoutError, RateLimitError
import config

T = TypeVar("T", bound=BaseModel)

# ══════════════════════════════════════════════════════════════════════
# LLM Factory
# ══════════════════════════════════════════════════════════════════════

_answer_llm: ChatOpenAI | None = None
_simple_llm: ChatOpenAI | None = None


def get_answer_llm(**overrides) -> ChatOpenAI:
    global _answer_llm
    if _answer_llm is None or overrides:
        kwargs = dict(
            base_url=overrides.pop("base_url", config.LLM_BASE_URL),
            api_key=overrides.pop("api_key", config.LLM_API_KEY),
            model=overrides.pop("model", config.LLM_MODEL),
            temperature=overrides.pop("temperature", 0.3),
            request_timeout=60,
            max_retries=2,
        )
        kwargs.update(overrides)
        _answer_llm = ChatOpenAI(**kwargs)
    return _answer_llm


def get_simple_llm(**overrides) -> ChatOpenAI:
    global _simple_llm
    if _simple_llm is None or overrides:
        kwargs = dict(
            base_url=overrides.pop("base_url", config.LLM_BASE_URL),
            api_key=overrides.pop("api_key", config.LLM_API_KEY),
            model=overrides.pop("model", config.SIMPLE_LLM_MODEL),
            temperature=overrides.pop("temperature", 0.2),
            request_timeout=45,
            max_retries=2,
        )
        kwargs.update(overrides)
        _simple_llm = ChatOpenAI(**kwargs)
    return _simple_llm


def reset_llms() -> None:
    global _answer_llm, _simple_llm
    _answer_llm = None
    _simple_llm = None


# ══════════════════════════════════════════════════════════════════════
# LLM 调用重试 - 指数退避
# ══════════════════════════════════════════════════════════════════════

_RETRYABLE = (APIError, APITimeoutError, RateLimitError)


def _make_retry(max_attempts: int = 3):
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )


def llm_invoke_with_retry(llm: ChatOpenAI, messages: list[BaseMessage],
                          max_retries: int = 3) -> BaseMessage:
    return _make_retry(max_retries)(llm.invoke)(messages)


async def llm_ainvoke_with_retry(llm: ChatOpenAI, messages: list[BaseMessage],
                                 max_retries: int = 3) -> BaseMessage:
    @_make_retry(max_retries)
    async def _call():
        return await llm.ainvoke(messages)
    return await _call()


# ══════════════════════════════════════════════════════════════════════
# 结构化输出 - 自动降级（with_structured_output -> JSON fallback）
# ══════════════════════════════════════════════════════════════════════

def invoke_structured(llm: ChatOpenAI, output_class: Type[T],
                      messages: list[BaseMessage],
                      max_retries: int = 3) -> T:
    try:
        structured_llm = llm.with_structured_output(output_class)
        return llm_invoke_with_retry(structured_llm, messages, max_retries=max_retries)
    except Exception as e:
        err_str = str(e).lower()
        if "response_format" in err_str or "unavailable" in err_str:
            logging.info("  [降级] with_structured_output 不可用，改用 JSON 模式")
            return _json_fallback_invoke(llm, output_class, messages, max_retries)
        raise


def _json_fallback_invoke(llm: ChatOpenAI, output_class: Type[T],
                          messages: list[BaseMessage],
                          max_retries: int = 3) -> T:
    schema_json = json.dumps(output_class.model_json_schema(), ensure_ascii=False)
    prompt_text = (
        f"请严格按照以下 JSON Schema 输出，不要包含额外文字，不要用 markdown 代码块包裹:\n"
        f"{schema_json}"
    )
    try:
        json_llm = llm.bind(response_format={"type": "json_object"})
        resp = llm_invoke_with_retry(
            json_llm, [*messages, HumanMessage(content=prompt_text)],
            max_retries=max_retries,
        )
    except Exception:
        resp = llm_invoke_with_retry(
            llm,
            [*messages, SystemMessage(
                content="你只输出 JSON，不要包含任何解释、markdown 标记或额外文字。"
            )],
            max_retries=max_retries,
        )

    text = resp.content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return output_class.model_validate_json(text)


# ══════════════════════════════════════════════════════════════════════
# Ark Coding Plan Embeddings（移植自 AniRAG，含限流退避）
# ══════════════════════════════════════════════════════════════════════

class ArkCodingEmbeddings(Embeddings):
    """火山方舟 Coding Plan 的 doubao-embedding-vision 客户端。"""

    def __init__(self, api_key: str, base_url: str, model: str, dimension: int):
        if not api_key:
            raise EnvironmentError("ARK_EMBEDDING_API_KEY is required when EMBEDDING_BACKEND=ark")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.dimension = dimension
        self.request_interval = config.ARK_EMBEDDING_REQUEST_INTERVAL
        self.max_retries = config.ARK_EMBEDDING_MAX_RETRIES
        self.max_backoff = config.ARK_EMBEDDING_MAX_BACKOFF
        self._last_request_at = 0.0

    @property
    def active_model(self) -> str:
        return self.model

    @property
    def model_identity(self) -> str:
        return f"ark-coding:{self.model}:{self.dimension}"

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        message = str(exc).lower()
        return status_code == 429 or any(marker in message for marker in (
            "account_ratelimitexceeded", "accountratelimitexceeded",
            "rate limit", "ratelimit", "too many requests", "toomanyrequests",
        ))

    def _wait_for_request_slot(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.request_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _create_embeddings(self, batch: List[str]):
        for attempt in range(self.max_retries + 1):
            self._wait_for_request_slot()
            try:
                return self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimension,
                    encoding_format="float",
                )
            except Exception as exc:
                if not self._is_rate_limit_error(exc) or attempt >= self.max_retries:
                    raise
                delay = min(max(self.request_interval, 2 ** attempt), self.max_backoff)
                delay += random.uniform(0, 0.5)
                logging.warning("Ark embedding 触发限流，%.1f 秒后重试 (%d/%d)",
                                delay, attempt + 1, self.max_retries)
                time.sleep(delay)
            finally:
                self._last_request_at = time.monotonic()

    def embed_documents(self, texts: List[str], target_dim: int | None = None) -> List[List[float]]:
        expected_dim = target_dim or self.dimension
        if expected_dim != self.dimension:
            raise ValueError(
                f"Ark embedding dimension {self.dimension} does not match target {expected_dim}."
            )
        all_vectors: List[List[float]] = []
        for start in range(0, len(texts), 10):
            batch = texts[start:start + 10]
            response = self._create_embeddings(batch)
            all_vectors.extend(item.embedding for item in response.data)
        return all_vectors

    def embed_query(self, text: str, target_dim: int | None = None) -> List[float]:
        return self.embed_documents([text], target_dim=target_dim)[0]


# ══════════════════════════════════════════════════════════════════════
# 本地 HuggingFace Embeddings（Ark Coding Plan 订阅失效时的兜底，零 API 成本）
# ══════════════════════════════════════════════════════════════════════

class LocalEmbeddings(Embeddings):
    """基于 sentence-transformers 的本地模型（默认 Qwen3-Embedding-0.6B, 1024 维）。"""

    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        logging.info("加载本地 Embedding 模型: %s (device=%s) ...", model_name, device)
        self._model = SentenceTransformer(model_name, device=device)
        self.model = model_name
        self.dimension = len(self._model.encode("test", normalize_embeddings=True))

    @property
    def active_model(self) -> str:
        return self.model

    @property
    def model_identity(self) -> str:
        return f"local:{self.model}:{self.dimension}"

    def embed_documents(self, texts: List[str], target_dim: int | None = None) -> List[List[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True,
                                  show_progress_bar=len(texts) > 50)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str, target_dim: int | None = None) -> List[float]:
        # Qwen3-Embedding 用 query 前缀提升检索质量
        v = self._model.encode(text, prompt_name="query", normalize_embeddings=True)
        return v.tolist()


_embeddings: Embeddings | None = None


def get_embeddings() -> Embeddings:
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    if config.EMBEDDING_BACKEND == "ark":
        _embeddings = ArkCodingEmbeddings(
            api_key=config.ARK_EMBEDDING_API_KEY,
            base_url=config.ARK_EMBEDDING_BASE_URL,
            model=config.ARK_EMBEDDING_MODEL,
            dimension=config.ARK_EMBEDDING_DIMENSIONS,
        )
    elif config.EMBEDDING_BACKEND == "local":
        _embeddings = LocalEmbeddings(
            model_name=config.LOCAL_EMBEDDING_MODEL,
            device=config.LOCAL_EMBEDDING_DEVICE,
        )
    else:
        raise ValueError(f"Unsupported EMBEDDING_BACKEND={config.EMBEDDING_BACKEND!r}; expected ark or local.")
    logging.info("  Embedding: %s | %s", config.EMBEDDING_BACKEND, _embeddings.model)
    return _embeddings
