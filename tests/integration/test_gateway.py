import pytest
from fastapi.testclient import TestClient

from app.integrations.gateway import gateway_ready, is_gateway, request_headers
from app.main import create_app
from app.storage.memory import MemoryBackend
from tests.integration.mock_gateway import MockGateway

Q = "How can I get a refund within 30 days?"


@pytest.fixture
def gw():
    g = MockGateway().start()
    yield g
    g.stop()


@pytest.fixture
def gw_env(gw, make_settings):
    s = make_settings(generation_base_url=gw.url, generation_api_key="gw-test-key", allowed_models="gpt-4o-mini,gpt-denied", max_attempts=1)
    app = create_app(s, backend=MemoryBackend())
    from tests.conftest import Env

    with TestClient(app) as c:
        yield Env(app, c, None, s, None)


def test_misses_go_through_the_gateway_and_hits_do_not(gw_env, gw):
    _, h = gw_env.tenant()
    assert gw_env.q(h, Q).json()["answer"] == f"GW[{Q}]"
    assert gw_env.q(h, Q).json()["cache_status"] == "exact_hit"
    assert len(gw.seen) == 1


def test_request_and_correlation_ids_reach_the_gateway(gw_env, gw):
    _, h = gw_env.tenant()
    gw_env.q({**h, "X-Request-ID": "trace-abc-123"}, Q)
    hdrs = gw.seen[0]["headers"]
    assert hdrs["x-request-id"] == "trace-abc-123" and hdrs["x-correlation-id"] == "trace-abc-123"


def test_gateway_policy_denial_is_surfaced_and_not_bypassed(gw_env, gw):
    _, h = gw_env.tenant()
    r = gw_env.q(h, Q, model="gpt-denied")
    assert r.status_code == 403 and r.json()["error"]["code"] == "policy_denied"
    assert len(gw.seen) == 1  # exactly one call, only to the gateway: no retry, no direct-to-OpenAI fallback
    assert gw_env.client.get("/v1/cache/stats", headers=h).json()["cached_entries"] == 0


def test_wrong_gateway_key_is_an_upstream_auth_error(gw, make_settings):
    s = make_settings(generation_base_url=gw.url, generation_api_key="wrong", max_attempts=1)
    from tests.conftest import Env

    with TestClient(create_app(s, backend=MemoryBackend())) as c:
        e = Env(None, c, None, s, None)
        _, h = e.tenant()
        r = e.q(h, Q)
        assert r.status_code == 502 and r.json()["error"]["code"] == "upstream_auth_failed" and "wrong" not in r.text


def test_gateway_down_is_a_clean_503(make_settings):
    s = make_settings(generation_base_url="http://127.0.0.1:9/v1", generation_api_key="k", max_attempts=1)
    from tests.conftest import Env

    with TestClient(create_app(s, backend=MemoryBackend())) as c:
        e = Env(None, c, None, s, None)
        _, h = e.tenant()
        assert e.q(h, Q).status_code == 503


def test_adapter_helpers(gw, make_settings):
    assert request_headers("r1") == {"X-Request-ID": "r1", "X-Correlation-ID": "r1"}
    assert is_gateway(make_settings(generation_base_url=gw.url)) and not is_gateway(make_settings())
    assert gateway_ready(make_settings(generation_base_url=gw.url)) is True
    assert gateway_ready(make_settings(generation_base_url="http://127.0.0.1:9/v1")) is False
    assert gateway_ready(make_settings()) is False
