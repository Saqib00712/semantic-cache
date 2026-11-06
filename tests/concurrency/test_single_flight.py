import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.cache import lifecycle
from tests.helpers import FakeGen

Q = "Explain how DNS caching works"


def fire(env, h, n, query=Q):
    barrier = threading.Barrier(n)

    def one(_):
        barrier.wait()
        return env.q(h, query)

    with ThreadPoolExecutor(n) as ex:
        return list(ex.map(one, range(n)))


def test_identical_concurrent_requests_generate_once(make_env, backend):
    e = make_env(backend=backend, gen=FakeGen(delay=0.4), lock_poll_seconds=0.01)
    e.app.state.runtime.sleep = time.sleep
    _, h = e.tenant()
    res = fire(e, h, 12)
    assert all(r.status_code == 200 for r in res) and e.gen.calls == 1
    statuses = sorted(r.json()["cache_status"] for r in res)
    assert statuses.count("miss") == 1 and len({r.json()["answer"] for r in res}) == 1
    assert set(statuses) <= {"miss", "coalesced", "exact_hit"}
    st = e.client.get("/v1/cache/stats", headers=h).json()
    assert st["by_status"]["miss"] == 1 and st["requests"] == 12
    assert float(st["gross_generation_cost_avoided_usd"]) > 0  # waiters avoided real generations


def test_different_queries_do_not_block_each_other(make_env):
    e = make_env(gen=FakeGen(delay=0.3))
    e.app.state.runtime.sleep = time.sleep
    _, h = e.tenant()
    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda i: e.q(h, f"distinct question number {i} about topic{i}"), range(4)))
    assert e.gen.calls == 4 and time.time() - t0 < 1.0  # ran in parallel (4 x 0.3s serial would be 1.2s)


def test_leader_failure_waiters_recover(make_env):
    e = make_env(
        gen=FakeGen(delay=0.3, fail_first=3, error="auth"), max_attempts=1
    )  # first request fails permanently, others retry as leader
    e.app.state.runtime.sleep = time.sleep
    _, h = e.tenant()
    res = fire(e, h, 4)
    codes = sorted(r.status_code for r in res)
    assert codes.count(502) >= 1 and codes.count(200) >= 1  # the failure did not poison the waiters
    assert e.client.get("/ready").json()["status"] == "ready"
    assert e.q(h, Q).json()["cache_status"] in ("exact_hit", "miss")


def test_lock_released_after_failure(make_env, backend):
    e = make_env(backend=backend, gen=FakeGen(fail_first=1, error="auth"), max_attempts=1)
    _, h = e.tenant()
    assert e.q(h, Q).status_code == 502
    assert e.q(h, Q).status_code == 200  # not stuck behind a leaked lock


def test_expired_lock_with_slow_leader_does_not_corrupt(make_env):
    """Lock TTL shorter than generation: a second request becomes leader too. Both finish; the stored entry is consistent and the
    first leader's late release does not free the second leader's lock (owner-checked)."""
    e = make_env(gen=FakeGen(delay=0.5), lock_ttl_seconds=0.2, lock_wait_seconds=3.0)
    e.app.state.runtime.sleep = time.sleep
    _, h = e.tenant()
    res = fire(e, h, 2)
    assert all(r.status_code == 200 for r in res) and e.gen.calls == 2
    assert e.q(h, Q).json()["cache_status"] == "exact_hit"
    assert e.client.get("/v1/cache/entries", headers=h).json()["count"] == 1


def test_invalidation_during_generation_is_not_overwritten(make_env, backend):
    """A generation that began BEFORE an invalidation and ends AFTER it must not write its (possibly stale) answer back."""
    gen = FakeGen(delay=0.5)
    e = make_env(backend=backend, gen=gen)
    tid, h = e.tenant()
    out = {}
    t = threading.Thread(target=lambda: out.update(r=e.q(h, Q)))
    t.start()
    time.sleep(0.2)
    lifecycle.wipe_tenant(backend, tid)  # operator invalidates while the model call is in flight
    t.join()
    assert out["r"].status_code == 200  # the caller still gets its answer
    assert e.client.get("/v1/cache/entries", headers=h).json()["count"] == 0  # ... but it was not cached
    assert e.q(h, Q).json()["cache_status"] == "miss"


def test_concurrent_distinct_tenants_isolated(make_env):
    e = make_env(gen=FakeGen(delay=0.2))
    e.app.state.runtime.sleep = time.sleep
    heads = [e.tenant(f"t{i}")[1] for i in range(4)]
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda h: e.q(h, Q), heads))
    assert e.gen.calls == 4  # same text, four tenants: four generations, never shared


@pytest.mark.parametrize("n", [8])
def test_hits_under_load_never_call_the_model(make_env, n):
    e = make_env()
    _, h = e.tenant()
    e.q(h, Q)
    res = fire(e, h, n)
    assert {r.json()["cache_status"] for r in res} == {"exact_hit"} and e.gen.calls == 1
