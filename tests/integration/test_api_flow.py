import json

import pytest

Q = "How can I get a refund within 30 days?"


def test_miss_then_exact_then_semantic(env):
    _, h = env.tenant()
    r1 = env.q(h, Q).json()
    r2 = env.q(h, Q).json()
    r3 = env.q(h, "How do I get my money back within 30 days").json()
    assert [r1["cache_status"], r2["cache_status"], r3["cache_status"]] == ["miss", "exact_hit", "semantic_hit"]
    assert r2["answer"] == r1["answer"] == r3["answer"] and env.gen.calls == 1
    assert r3["similarity_score"] >= env.settings.similarity_threshold


@pytest.mark.parametrize("variant", ["How can I get a refund within 60 days?", "How can I get a refund, it is not within 30 days?"])
def test_changed_meaning_is_not_served_from_cache(env, variant):
    _, h = env.tenant()
    env.q(h, Q)
    assert env.q(h, variant).json()["cache_status"] == "miss" and env.gen.calls == 2


@pytest.mark.parametrize(
    "body", [{"context_version": "c2"}, {"prompt_template_version": "v2"}, {"tool_version": "t1"}, {"max_output_tokens": 64}]
)
def test_version_or_param_change_misses(env, body):
    _, h = env.tenant()
    env.q(h, Q)
    assert env.q(h, Q, **body).json()["cache_status"] == "miss"
    assert env.q(h, "How do I get my money back within 30 days", **body).json()["cache_status"] in ("semantic_hit", "miss")


def test_personal_and_high_temperature_bypass(env):
    _, h = env.tenant()
    for body in (
        {"query": "where is my order 12345"},
        {"query": "tell a story", "temperature": 1.0},
        {"query": "hello", "personalized": True},
    ):
        a, b = env.q(h, **body).json(), env.q(h, **body).json()
        assert a["cache_status"] == b["cache_status"] == "ineligible"
    assert env.client.get("/v1/cache/entries", headers=h).json()["count"] == 0


def test_truncated_answer_not_stored(make_env):
    from tests.helpers import FakeGen

    e = make_env(gen=FakeGen(finish_reason="length"))
    _, h = e.tenant()
    e.q(h, Q)
    assert e.q(h, Q).json()["cache_status"] == "miss" and e.gen.calls == 2


def test_blocked_pattern_answer_not_stored(make_env):
    from tests.helpers import FakeGen

    e = make_env(gen=FakeGen(content="here is a key sk-abcdefghijklmnopqrstuv"))
    _, h = e.tenant()
    e.q(h, Q)
    assert e.q(h, Q).json()["cache_status"] == "miss"


def test_model_allow_list(env):
    _, h = env.tenant()
    r = env.q(h, Q, model="gpt-unknown")
    assert r.status_code == 422 and r.json()["error"]["code"] == "model_not_allowed" and env.gen.calls == 0


def test_invalidation_by_entry_filter_and_all(env):
    _, h = env.tenant()
    env.q(h, Q, context_version="c1")
    env.q(h, "other question here", context_version="c2")
    rows = env.client.get("/v1/cache/entries", headers=h).json()["entries"]
    assert len(rows) == 2 and "answer" not in json.dumps(rows) and Q not in json.dumps(rows)
    assert env.client.delete("/v1/cache", params={"context_version": "c1"}, headers=h).json() == {"deleted": 1}
    assert env.q(h, Q, context_version="c1").json()["cache_status"] == "miss"
    eid = env.client.get("/v1/cache/entries", headers=h).json()["entries"][0]["entry_id"]
    assert env.client.delete(f"/v1/cache/entries/{eid}", headers=h).json() == {"deleted": 1}
    assert env.client.delete(f"/v1/cache/entries/{eid}", headers=h).status_code == 404
    assert env.client.delete("/v1/cache", headers=h).status_code == 422
    assert env.client.delete("/v1/cache", params={"bogus": "x"}, headers=h).status_code == 422
    assert env.client.delete("/v1/cache", params={"all": "true"}, headers=h).json()["deleted"] >= 1
    assert env.client.get("/v1/cache/entries", headers=h).json()["count"] == 0


def test_stats_and_cost_accounting(env):
    _, h = env.tenant()
    env.q(h, Q)
    env.q(h, Q)
    env.q(h, "How do I get my money back within 30 days")
    st = env.client.get("/v1/cache/stats", headers=h).json()
    assert (
        st["requests"] == 3 and st["by_status"]["miss"] == 1 and st["by_status"]["exact_hit"] == 1 and st["by_status"]["semantic_hit"] == 1
    )
    assert st["hit_rate"] == pytest.approx(2 / 3) and st["cached_entries"] == 1
    gen, avoided = float(st["generation_cost_usd"]), float(st["gross_generation_cost_avoided_usd"])
    assert avoided == pytest.approx(2 * gen)  # two hits each avoid one original generation
    assert float(st["estimated_net_savings_usd"]) == pytest.approx(avoided - float(st["embedding_cost_usd"]) - float(st["infra_cost_usd"]))


def test_health_ready_metrics_ui(env):
    assert env.client.get("/health").json()["status"] == "ok"
    assert env.client.get("/ready").json()["status"] == "ready"
    _, h = env.tenant()
    env.q(h, Q)
    m = env.client.get("/metrics", headers=env.admin).text
    assert 'semcache_requests_total{cache_status="miss"} 1' in m
    assert "default-src 'none'" in env.client.get("/ui").headers["content-security-policy"]


def test_request_id_propagates_to_generation(env):
    _, h = env.tenant()
    r = env.q({**h, "X-Request-ID": "my-req-id-1"}, Q)
    assert r.headers["x-request-id"] == "my-req-id-1" and r.json()["request_id"] == "my-req-id-1"
    assert env.gen.last_kwargs["extra_headers"]["X-Request-ID"] == "my-req-id-1"
