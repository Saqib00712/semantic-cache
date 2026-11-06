import io
import logging

from app.observability.logging import JsonFormatter

Q = "How can I get a refund within 30 days?"


def test_auth_required_everywhere(env):
    for method, path in (("post", "/v1/cache/query"), ("get", "/v1/cache/entries"), ("get", "/v1/cache/stats"), ("delete", "/v1/cache")):
        r = getattr(env.client, method)(path)
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert env.client.post("/v1/cache/query", headers={"Authorization": "Bearer sc_deadbeef_x"}, json={"query": "q"}).status_code == 401
    assert env.client.post("/v1/cache/query", data="not json").status_code == 401  # auth before body parsing


def test_tenant_isolation_exact_semantic_and_management(env):
    _, a = env.tenant("a")
    _, b = env.tenant("b")
    env.q(a, Q)
    assert env.q(b, Q).json()["cache_status"] == "miss"  # same text, different tenant: no hit
    _, c = env.tenant("c")
    assert (
        env.q(c, "How do I get my money back within 30 days").json()["cache_status"] == "miss"
    )  # paraphrase: no cross-tenant semantic hit
    eid_a = env.client.get("/v1/cache/entries", headers=a).json()["entries"][0]["entry_id"]
    assert env.client.delete(f"/v1/cache/entries/{eid_a}", headers=b).status_code == 404  # cannot delete another tenant's entry
    assert env.q(a, Q).json()["cache_status"] == "exact_hit"
    env.client.delete("/v1/cache", params={"all": "true"}, headers=b)
    assert env.q(a, Q).json()["cache_status"] == "exact_hit"  # b's wipe left a untouched
    assert env.client.get("/v1/cache/stats", headers=b).json()["requests"] == 1  # only b's own query


def test_tenant_id_in_body_is_rejected(env):
    _, h = env.tenant()
    r = env.client.post("/v1/cache/query", headers=h, json={"query": "hi", "tenant_id": "other"})
    assert r.status_code == 422 and "other" not in r.text


def test_revoked_and_suspended(env):
    tid, h = env.tenant()
    assert env.q(h, Q).status_code == 200
    assert env.client.post(f"/admin/tenants/{tid}/suspend", headers=env.admin).status_code == 200
    assert env.q(h, Q).status_code == 403
    env.client.post(f"/admin/tenants/{tid}/activate", headers=env.admin)
    assert env.q(h, Q).status_code == 200
    assert env.client.post(f"/admin/tenants/{tid}/revoke-keys", headers=env.admin).json()["revoked"] == 1
    assert env.q(h, Q).status_code == 401


def test_admin_requires_token_and_is_hidden_when_unset(env, make_env):
    assert env.client.post("/admin/tenants", json={"name": "x"}).status_code == 401
    assert env.client.post("/admin/tenants", json={"name": "x"}, headers={"Authorization": "Bearer " + "b" * 40}).status_code == 401
    _, h = env.tenant()
    assert env.client.get("/admin/tenants", headers=h).status_code == 401  # a tenant key is not an admin token
    off = make_env(admin_token=None)
    assert off.client.get("/admin/tenants").status_code == 404


def test_created_key_shown_once_and_stored_hashed(env):
    r = env.client.post("/admin/tenants", json={"name": "n"}, headers=env.admin).json()
    from app.storage.database import db

    with db(env.settings.database_path) as conn:
        dump = " ".join(str(x) for row in conn.execute("SELECT * FROM api_keys") for x in row)
    assert r["api_key"] not in dump and r["api_key"].startswith("sc_")


def test_logs_never_contain_prompts_answers_or_keys(make_env):
    e = make_env()
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("semcache")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        _, h = e.tenant()
        secret_prompt = "tell me about zebra-giraffe-unique-topic"
        e.q(h, secret_prompt)
        e.q(h, secret_prompt)
        e.client.post("/v1/cache/query", headers={"Authorization": "Bearer sc_aaaaaaaa_bad"}, json={"query": secret_prompt})
    finally:
        logger.removeHandler(handler)
    out = buf.getvalue()
    assert out and "zebra" not in out and "ANSWER[" not in out and h["Authorization"].split(" ")[1] not in out and "sc_aaaaaaaa" not in out


def test_metrics_have_no_tenant_prompt_or_key(env):
    tid, h = env.tenant()
    env.q(h, "zebra-giraffe-unique-topic")
    m = env.client.get("/metrics", headers=env.admin).text
    assert tid not in m and "zebra" not in m and "sc_" not in m


def test_metrics_protected_when_admin_configured(env):
    assert env.client.get("/metrics").status_code == 401


def test_error_responses_do_not_echo_input(env):
    _, h = env.tenant()
    r = env.client.post("/v1/cache/query", headers=h, json={"query": "ok", "temperature": "SECRETVALUE"})
    assert r.status_code == 422 and "SECRETVALUE" not in r.text
    r = env.client.post("/v1/cache/query", headers=h, json={"query": "ok", "SECRETFIELD": 1})
    assert r.status_code == 422 and "SECRETFIELD" not in r.text
    assert env.client.post("/v1/cache/query", headers=h, content=b"{bad").status_code == 400
    assert env.client.post("/v1/cache/query", headers=h, json={"query": "bad\x01char"}).status_code == 422
    assert env.client.post("/v1/cache/query", headers=h, json={"query": "   "}).status_code == 422


def test_body_limit(env):
    _, h = env.tenant()
    r = env.client.post("/v1/cache/query", headers={**h, "Content-Type": "application/json"}, content=b'{"query":"' + b"a" * 70_000 + b'"}')
    assert r.status_code == 413


def test_deeply_nested_json_is_a_client_error(env):
    _, h = env.tenant()
    r = env.client.post("/v1/cache/query", headers=h, content=b"[" * 20000 + b"]" * 20000)
    assert r.status_code in (400, 422)


def test_rate_limit_and_quota(make_env):
    e = make_env()
    _, h = e.tenant(rate_per_minute=2)
    assert [e.q(h, f"q{i}").status_code for i in range(3)] == [200, 200, 429]
    _, h2 = e.tenant("q", daily_queries=1)
    assert [e.q(h2, "x").status_code for i in range(2)] == [200, 429]


def test_daily_budget_enforced(make_env):
    from decimal import Decimal

    e = make_env(price_input_per_mtok="1000000", price_output_per_mtok="1000000")  # 30 USD per call
    _, h = e.tenant(daily_budget_usd=Decimal("1"))
    assert e.q(h, "first question").status_code == 200
    r = e.q(h, "second question")
    assert r.status_code == 429 and r.json()["error"]["code"] == "budget_exceeded" and e.gen.calls == 1


def test_unknown_routes_and_methods(env):
    assert env.client.get("/nope").status_code == 404
    assert env.client.get("/docs").status_code == 404 and env.client.get("/openapi.json").status_code == 404
