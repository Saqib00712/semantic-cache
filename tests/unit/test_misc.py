import json
import logging

import pytest

from app.config import ConfigurationError, Settings, load_settings
from app.observability import metrics
from app.observability.logging import JsonFormatter, redact, register_secret
from app.storage import ledger
from app.storage.redis_backend import RedisBackend, _tag_escape


def test_unknown_env_names_are_rejected(monkeypatch):
    monkeypatch.setenv("SEMCACHE_SIMILARITY_THRESHOLDD", "0.5")
    with pytest.raises(ConfigurationError):
        load_settings(env_file=None)


def test_invalid_settings_rejected():
    with pytest.raises(Exception):
        Settings(_env_file=None, backend="memory", similarity_threshold=2)
    with pytest.raises(Exception):
        Settings(_env_file=None, backend="memory", system_prompt_version="bad version!")


def test_safe_summary_has_no_secrets():
    s = Settings(_env_file=None, backend="memory", openai_api_key="sk-secretsecretsecret1234", admin_token="z" * 40)
    dump = json.dumps(s.safe_summary())
    assert "sk-secret" not in dump and "zzzz" not in dump


def test_redaction_removes_secrets_and_keys():
    register_secret("supersecretvalue-123")
    assert "supersecretvalue-123" not in redact("x supersecretvalue-123 y")
    assert "sc_deadbeef_abcdefghijklmnop" not in redact("key sc_deadbeef_abcdefghijklmnop here")
    assert "sk-abcdefghijklmnop1234" not in redact("sk-abcdefghijklmnop1234")


def test_log_formatter_drops_unlisted_fields():
    rec = logging.LogRecord("semcache", logging.INFO, "f", 1, "query", None, None)
    rec.prompt = "my private prompt"
    rec.answer = "private answer"
    rec.cache_status = "miss"
    out = JsonFormatter().format(rec)
    assert "private" not in out and "miss" in out


def test_metrics_render_and_labels():
    metrics.inc("semcache_requests_total", cache_status="miss")
    metrics.observe("semcache_request_seconds", 0.03, cache_status="miss")
    text = metrics.render()
    assert 'semcache_requests_total{cache_status="miss"} 1' in text
    assert 'semcache_request_seconds_bucket{cache_status="miss",le="0.05"} 1' in text
    assert "semcache_request_seconds_count" in text


def test_pico_roundtrip():
    from decimal import Decimal

    assert ledger.from_pico(ledger.to_pico(Decimal("0.000123456789"))) == Decimal("0.000123456789")


def test_redis_index_commands_without_a_server():
    """RediSearch is NOT available in the test environment, so the command text is verified (against the documented syntax), not run."""

    class Dummy:
        def register_script(self, s):
            return s

        def execute_command(self, *a):
            raise __import__("redis").ResponseError("unknown command")

    be = RedisBackend(Dummy(), "scan", 1536)
    args = be.create_index_args()
    assert args[:4] == ["sc:idx:v1:1536", "ON", "HASH", "PREFIX"] and "HNSW" in args and "COSINE" in args and "1536" in args
    assert (
        be.knn_query("abc123", "text-embedding-3-small", 5)
        == "(@ns:{abc123} @emb_model:{text\\-embedding\\-3\\-small})=>[KNN 5 @vec $q AS score]"
    )
    assert _tag_escape("a-b.c|d") == "a\\-b\\.c\\|d"
    with pytest.raises(Exception):
        RedisBackend(Dummy(), "search", 8)


def test_blocked_patterns_default_actually_matches_and_commas_are_safe():
    s = Settings(_env_file=None, backend="memory")
    assert any(p.search("key sk-abcdefghijklmnopqrstuv") for p in s.response_patterns)
    s2 = Settings(_env_file=None, backend="memory", blocked_response_patterns="foo{2,3};;bar")
    assert [p.pattern for p in s2.response_patterns] == ["foo{2,3}", "bar"]
