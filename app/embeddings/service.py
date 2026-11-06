"""Embeddings: OpenAI (production) or a deterministic offline mock (tests/demo). Results are validated, cached by (model, text hash)
and costed. The text is hashed for the embedding cache key; it is never stored."""

import hashlib
import re
import time

import numpy as np

from app.cache.normalize import sha256_hex
from app.config import Settings
from app.similarity.vector import VectorError, validate_vector
from app.storage.backend import Backend, BackendError


class EmbeddingError(Exception):
    """Embedding unavailable or invalid. Safe text only. The cache degrades to exact-match when this happens."""


_WORD = re.compile(r"[a-z0-9']+")
# Tiny synonym lexicon for the mock embedder only: it lets paraphrases land near each other offline. It is NOT how OpenAI
# embeddings behave, so evaluation numbers from the mock say something about the safety pipeline, not about real embeddings.
_SYN = {
    "refund": "money_back", "reimburse": "money_back", "reimbursement": "money_back", "return": "money_back", "back": "money_back",
    "cancel": "cancel", "terminate": "cancel", "close": "cancel",
    "password": "password", "passcode": "password", "reset": "reset", "recover": "reset", "change": "reset",
    "price": "cost", "cost": "cost", "pricing": "cost", "charge": "cost", "fee": "cost",
    "shipping": "ship", "delivery": "ship", "ship": "ship", "deliver": "ship",
    "how": "how", "what": "what", "can": "can", "i": "i", "do": "do", "to": "to", "my": "my", "the": "the", "a": "a", "is": "is",
}  # fmt: skip
_STOP = {"how", "can", "i", "do", "to", "the", "a", "is", "what", "my", "you", "me", "please", "it", "of", "an", "and"}


def mock_embed_text(text: str, dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for w in _WORD.findall(text.lower()):
        if w in _STOP:
            continue
        tok = _SYN.get(w, w)
        h = hashlib.sha256(tok.encode()).digest()
        v[int.from_bytes(h[:4], "big") % dim] += 1.0
        v[int.from_bytes(h[4:8], "big") % dim] += 0.5
    if not v.any():
        v[int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big") % dim] = 1.0
    return v


def make_openai_client(settings: Settings):
    import openai

    key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    return openai.OpenAI(api_key=key, base_url=settings.openai_base_url, timeout=settings.openai_timeout_seconds, max_retries=0)


def _is_transient(e: Exception) -> bool:
    import openai

    return isinstance(e, openai.RateLimitError | openai.APITimeoutError | openai.APIConnectionError) or (
        isinstance(e, openai.APIStatusError) and e.status_code >= 500
    )


def embed(settings: Settings, backend: Backend, client, text: str, *, sleep=time.sleep) -> tuple[np.ndarray, int, bool]:
    """Return (unit vector, token count, served_from_embedding_cache). Raises EmbeddingError."""
    ck = sha256_hex(f"{settings.embedding_model}|{settings.embedding_dim}|{text}")
    try:
        got = backend.emb_get(ck)
    except BackendError:
        got = None  # the embedding cache is an optimization; a failure just means a fresh call
    if got is not None:
        try:
            return validate_vector(got[0], settings.embedding_dim), got[1], True
        except VectorError:
            pass  # corrupt cached vector: ignore and recompute
    if settings.embedding_mode == "mock":
        vec, tokens = validate_vector(mock_embed_text(text, settings.embedding_dim), settings.embedding_dim), max(1, len(text) // 4)
    else:
        vec, tokens = _call_openai(settings, client, text, sleep)
    try:
        backend.emb_put(ck, vec, tokens, settings.embedding_cache_ttl_seconds)
    except BackendError:
        pass
    return vec, tokens, False


def _call_openai(settings: Settings, client, text: str, sleep) -> tuple[np.ndarray, int]:
    import openai

    if client is None:
        raise EmbeddingError("embeddings are not configured")
    last = "unknown"
    for attempt in range(1, settings.max_attempts + 1):
        try:
            kwargs = {"model": settings.embedding_model, "input": text}
            if settings.embedding_model.startswith("text-embedding-3"):
                kwargs["dimensions"] = settings.embedding_dim
            resp = client.embeddings.create(**kwargs)
            vec = validate_vector(resp.data[0].embedding, settings.embedding_dim)
            return vec, int(getattr(resp.usage, "prompt_tokens", 0) or 0)
        except VectorError as e:
            raise EmbeddingError(str(e)) from None
        except (IndexError, AttributeError, TypeError):
            raise EmbeddingError("malformed embeddings response") from None
        except openai.OpenAIError as e:
            last = type(e).__name__
            if not _is_transient(e) or attempt == settings.max_attempts:
                break
            sleep(settings.retry_base_seconds * (2 ** (attempt - 1)))
    raise EmbeddingError(f"embedding request failed ({last})")
