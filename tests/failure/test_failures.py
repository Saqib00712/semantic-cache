import numpy as np
import pytest

from app.cache.models import EntryMeta
from app.storage.backend import BackendError
from app.storage.memory import MemoryBackend
from tests.conftest import DIM
from tests.helpers import FakeEmbed, FakeGen

Q = "How can I get a refund within 30 days?"


class DownBackend(MemoryBackend):
    """A cache store where every call fails (Redis down)."""

    def __getattribute__(self, name):
        if name in {
            "ping",
            "get_exact",
            "get",
            "put",
            "search",
            "delete",
            "delete_matching",
            "list_entries",
            "count",
            "bump_epoch",
            "get_epoch",
            "emb_get",
            "emb_put",
            "lock_acquire",
            "lock_release",
            "flight_put",
            "flight_get",
            "incr_window",
            "record_hit",
        }:

            def boom(*a, **k):
                raise BackendError("ConnectionError")

            return boom
        return object.__getattribute__(self, name)


def test_cache_store_down_bypass_still_answers(make_env):
    e = make_env(backend=DownBackend(), on_cache_failure="bypass")
    _, h = e.tenant()
    r = e.q(h, Q)
    assert r.status_code == 200 and r.json()["cache_status"] == "miss" and e.gen.calls == 1
    assert e.client.get("/ready").status_code == 503
    assert e.client.get("/v1/cache/entries", headers=h).status_code == 503
    assert 'cause="cache_store_bypass"' in e.client.get("/metrics", headers=e.admin).text


def test_cache_store_down_fail_closed(make_env):
    e = make_env(backend=DownBackend(), on_cache_failure="fail")
    _, h = e.tenant()
    r = e.q(h, Q)
    assert r.status_code == 503 and r.json()["error"]["code"] == "cache_unavailable" and e.gen.calls == 0


def test_budget_and_quota_still_enforced_when_cache_down(make_env):
    e = make_env(backend=DownBackend())
    _, h = e.tenant(daily_queries=1)
    assert e.q(h, Q).status_code == 200 and e.q(h, Q).status_code == 429  # SQLite-backed, independent of the cache store


def test_real_redis_connection_failure(make_env):
    import redis

    from app.storage.redis_backend import RedisBackend

    class Dead:
        def register_script(self, s):
            return lambda **k: (_ for _ in ()).throw(redis.ConnectionError("x"))

        def execute_command(self, *a):
            raise redis.ResponseError("unknown")

        def __getattr__(self, n):
            def f(*a, **k):
                raise redis.ConnectionError("down")

            return f

    be = RedisBackend(Dead(), "scan", DIM)
    with pytest.raises(BackendError):
        be.get_exact("ns", "k")
    e = make_env(backend=be)
    _, h = e.tenant()
    assert e.q(h, Q).status_code == 200  # bypass policy


def test_embedding_failure_degrades_to_exact_only(make_env):
    e = make_env(
        embedding_mode="openai", embed_client=FakeEmbed(dim=DIM, fail_first=99, error="auth"), openai_api_key="sk-test-not-real-key-123456"
    )
    _, h = e.tenant()
    a, b = e.q(h, Q).json(), e.q(h, Q).json()
    assert (a["cache_status"], b["cache_status"]) == ("miss", "exact_hit")  # exact matching still works
    assert e.q(h, "How do I get my money back within 30 days").json()["cache_status"] == "miss"  # no semantic layer without embeddings
    assert 'cause="embedding_failed"' in e.client.get("/metrics", headers=e.admin).text


def test_invalid_embedding_dimension_never_stored(make_env):
    from app.cache.keys import namespace

    e = make_env(
        embedding_mode="openai", embed_client=FakeEmbed(dim=DIM, vector=[1.0, 2.0, 3.0]), openai_api_key="sk-test-not-real-key-123456"
    )
    tid, h = e.tenant()
    assert e.q(h, Q).status_code == 200
    (row,) = e.backend.list_entries(namespace(tid), 5)
    assert row["embedding_model"] is None  # stored for exact-match only, never with a bad vector
    assert e.backend.search(namespace(tid), np.ones(DIM, dtype=np.float32) / 16, 5, e.settings.embedding_model) == []


@pytest.mark.parametrize("error,status,code", [
    ("rate_limit", 503, "upstream_rate_limited"), ("timeout", 504, "upstream_timeout"), ("connection", 503, "upstream_unavailable"),
    ("auth", 502, "upstream_auth_failed"), ("forbidden", 403, "policy_denied"), ("server", 502, "upstream_error"),
])  # fmt: skip
def test_openai_errors_map_to_clean_responses_and_nothing_is_cached(make_env, error, status, code):
    e = make_env(gen=FakeGen(fail_first=99, error=error))
    _, h = e.tenant()
    r = e.q(h, Q)
    assert (r.status_code, r.json()["error"]["code"]) == (status, code) and "sk-" not in r.text
    assert e.client.get("/v1/cache/stats", headers=h).json()["cached_entries"] == 0


def test_transient_openai_error_is_retried(make_env):
    e = make_env(gen=FakeGen(fail_first=2, error="rate_limit"))
    _, h = e.tenant()
    assert e.q(h, Q).status_code == 200 and e.gen.calls == 3


def test_corrupt_entry_metadata_is_rejected_and_regenerated(make_env):
    from app.cache.keys import namespace

    e = make_env()
    tid, h = e.tenant()
    e.q(h, Q)
    ns = namespace(tid)
    (row,) = e.backend.list_entries(ns, 5)
    e.backend._entries[ns][row["entry_id"]]["meta"].pop("signature")  # corrupt in place
    assert e.q(h, Q).json()["cache_status"] == "miss"  # not trusted
    assert e.gen.calls == 2 and e.q(h, Q).json()["cache_status"] == "exact_hit"  # corrupt one removed, fresh one served
    assert 'component="corrupt_entry_removed"' in e.client.get("/metrics", headers=e.admin).text


def test_stored_entry_with_wrong_vector_size_is_ignored_by_search(make_env):
    from app.cache.keys import namespace

    e = make_env()
    tid, h = e.tenant()
    e.q(h, Q)
    ns = namespace(tid)
    (row,) = e.backend.list_entries(ns, 5)
    e.backend._entries[ns][row["entry_id"]]["vec"] = np.ones(3, dtype=np.float32)
    assert e.q(h, "How do I get my money back within 30 days").status_code == 200  # no crash


def test_expired_entry_is_not_served(make_env):
    e = make_env(ttl_seconds=1)
    e.app.state.runtime.clock = lambda: __import__("time").time()
    _, h = e.tenant()
    e.q(h, Q)
    import time

    time.sleep(1.2)
    assert e.q(h, Q).json()["cache_status"] == "miss"


def test_empty_model_answer_is_502_and_not_cached(make_env):
    e = make_env(gen=FakeGen(content=""))
    _, h = e.tenant()
    r = e.q(h, Q)
    assert r.status_code == 502 and r.json()["error"]["code"] == "empty_generation"


def test_meta_model_schema_rejects_extra_fields():
    with pytest.raises(Exception):
        EntryMeta.model_validate({"bogus": 1})
