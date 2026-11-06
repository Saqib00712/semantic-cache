import numpy as np
import pytest

from app.config import Settings
from app.embeddings.service import EmbeddingError, embed, mock_embed_text
from app.generation.service import GenerationError, cost_of, generate
from app.similarity.vector import VectorError, cosine, validate_vector
from app.storage.memory import MemoryBackend
from tests.helpers import FakeEmbed, FakeGen


def S(**kw):
    return Settings(_env_file=None, backend="memory", embedding_dim=8, retry_base_seconds=0.0, log_level="ERROR", **kw)


def test_validate_vector():
    v = validate_vector([3, 4, 0, 0, 0, 0, 0, 0], 8)
    assert abs(float(np.linalg.norm(v)) - 1) < 1e-6 and v.dtype == np.float32
    for bad in ([1, 2], [float("nan")] + [0] * 7, [float("inf")] + [0] * 7, [0] * 8, "abc", [[1] * 8]):
        with pytest.raises(VectorError):
            validate_vector(bad, 8)


def test_cosine_of_unit_vectors():
    a = validate_vector([1, 0, 0, 0, 0, 0, 0, 0], 8)
    b = validate_vector([1, 1, 0, 0, 0, 0, 0, 0], 8)
    assert cosine(a, a) == pytest.approx(1) and cosine(a, b) == pytest.approx(0.7071, abs=1e-3)


def test_mock_embedding_is_deterministic_and_paraphrase_close():
    a, b = mock_embed_text("how do I get a refund", 256), mock_embed_text("how do I get a refund", 256)
    assert np.array_equal(a, b)
    close = cosine(
        validate_vector(mock_embed_text("how to get a refund", 256), 256), validate_vector(mock_embed_text("reimbursement how", 256), 256)
    )
    far = cosine(
        validate_vector(mock_embed_text("how to get a refund", 256), 256), validate_vector(mock_embed_text("weather in paris", 256), 256)
    )
    assert close > far


def test_embed_openai_caches_and_bills_once():
    be, c = MemoryBackend(), FakeEmbed()
    s = S()
    v1, t1, cached1 = embed(s, be, c, "hello")
    v2, t2, cached2 = embed(s, be, c, "hello")
    assert (cached1, cached2, c.calls, t1) == (False, True, 1, 5) and np.array_equal(v1, v2)
    assert c.last_kwargs["dimensions"] == 8 and c.last_kwargs["model"] == s.embedding_model


def test_embed_retries_transient_then_succeeds():
    c = FakeEmbed(fail_first=2, error="rate_limit")
    embed(S(), MemoryBackend(), c, "x", sleep=lambda x: None)
    assert c.calls == 3


@pytest.mark.parametrize("error", ["auth", "bad_request"])
def test_embed_does_not_retry_permanent_errors(error):
    c = FakeEmbed(fail_first=9, error=error)
    with pytest.raises(EmbeddingError):
        embed(S(), MemoryBackend(), c, "x", sleep=lambda x: None)
    assert c.calls == 1


def test_embed_gives_up_after_max_attempts():
    c = FakeEmbed(fail_first=99, error="timeout")
    with pytest.raises(EmbeddingError):
        embed(S(max_attempts=3), MemoryBackend(), c, "x", sleep=lambda x: None)
    assert c.calls == 3


@pytest.mark.parametrize("vector", [[1.0, 2.0], [float("nan")] + [0.0] * 7, [0.0] * 8])
def test_embed_rejects_invalid_vectors(vector):
    with pytest.raises(EmbeddingError):
        embed(S(), MemoryBackend(), FakeEmbed(vector=vector), "x")


def test_embed_without_client():
    with pytest.raises(EmbeddingError):
        embed(S(), MemoryBackend(), None, "x")


def test_generate_ok_and_headers():
    g = FakeGen()
    out = generate(S(), g, model="m", query="hi", max_tokens=50, temperature=0.0, request_id="rid1")
    assert out.answer == "ANSWER[hi]" and (out.prompt_tokens, out.completion_tokens) == (20, 10)
    assert g.last_kwargs["extra_headers"] == {"X-Request-ID": "rid1", "X-Correlation-ID": "rid1"}
    assert g.last_kwargs["messages"][0]["role"] == "system" and g.last_kwargs["max_tokens"] == 50


@pytest.mark.parametrize("error,code,status,calls", [
    ("rate_limit", "upstream_rate_limited", 503, 3), ("timeout", "upstream_timeout", 504, 3), ("connection", "upstream_unavailable", 503, 3),
    ("server", "upstream_error", 502, 3), ("auth", "upstream_auth_failed", 502, 1), ("forbidden", "policy_denied", 403, 1),
    ("bad_request", "upstream_rejected_request", 502, 1),
])  # fmt: skip
def test_generation_error_classification_and_retry_policy(error, code, status, calls):
    g = FakeGen(fail_first=99, error=error)
    with pytest.raises(GenerationError) as e:
        generate(S(), g, model="m", query="q", max_tokens=50, temperature=0, request_id="r", sleep=lambda x: None)
    assert (e.value.code, e.value.status, g.calls) == (code, status, calls)


def test_generation_recovers_after_transient_failure():
    g = FakeGen(fail_first=1, error="timeout")
    assert generate(S(), g, model="m", query="q", max_tokens=50, temperature=0, request_id="r", sleep=lambda x: None).answer


@pytest.mark.parametrize("content", ["", "   ", None])
def test_empty_generation_is_an_error(content):
    g = FakeGen(content=content if content is not None else "")
    g.content = content if content is not None else ""
    with pytest.raises(GenerationError) as e:
        generate(S(), g, model="m", query="q", max_tokens=50, temperature=0, request_id="r")
    assert e.value.code == "empty_generation"


def test_cost_arithmetic_is_exact():
    s = S(price_input_per_mtok="1", price_output_per_mtok="2")
    assert str(cost_of(s, 1_000_000, 500_000)) == "2"
