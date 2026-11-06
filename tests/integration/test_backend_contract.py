"""The same contract tests run against the memory backend and a real Redis server."""

import time
import uuid

import numpy as np
import pytest

from app.cache.keys import namespace
from tests.conftest import DIM


def unit(*idx):
    v = np.zeros(DIM, dtype=np.float32)
    for i in idx:
        v[i] = 1
    return v / np.linalg.norm(v)


def meta(ek, **o):
    return {"created_at": time.time(), "embedding_model": "m1", "exact_key": ek, "model": "gpt", "context_version": "c1", **o}


def put(be, ns, vec=None, ek=None, entry_id=None, ttl=60, cap=100, epoch=0, **m):
    entry_id, ek = entry_id or uuid.uuid4().hex, ek or uuid.uuid4().hex
    ok = be.put(ns, entry_id, ek, meta(ek, entry_id=entry_id, **m), "the answer", vec, ttl, cap, epoch)
    return entry_id, ek, ok


def test_put_get_exact_roundtrip(backend):
    ns = "ns1"
    eid, ek, ok = put(backend, ns, unit(1))
    assert ok and backend.ping()
    e = backend.get_exact(ns, ek)
    assert e.answer == "the answer" and e.meta["entry_id"] == eid and np.allclose(e.vec, unit(1))
    assert backend.get(ns, eid).answer == "the answer" and backend.get_exact(ns, "missing") is None and backend.count(ns) == 1


def test_namespaces_are_isolated(backend):
    _, ek, _ = put(backend, "nsA", unit(1))
    assert backend.get_exact("nsB", ek) is None
    assert backend.search("nsB", unit(1), 5, "m1") == []
    assert backend.count("nsB") == 0 and backend.delete_matching("nsB", {}) == 0 and backend.count("nsA") == 1


def test_search_ranks_by_cosine_and_filters_embedding_model(backend):
    ns = "ns1"
    a, _, _ = put(backend, ns, unit(1))
    b, _, _ = put(backend, ns, unit(1, 2))
    c, _, _ = put(backend, ns, unit(5))
    put(backend, ns, unit(1), embedding_model="other")
    res = backend.search(ns, unit(1), 3, "m1")
    assert [r[0] for r in res][:2] == [a, b] and len(res) == 3 and res[0][1] == pytest.approx(1.0, abs=1e-5)
    assert res[1][1] == pytest.approx(0.7071, abs=1e-3) and c in [r[0] for r in res]
    assert len(backend.search(ns, unit(1), 1, "m1")) == 1


def test_search_skips_entries_without_a_vector(backend):
    ns = "ns1"
    put(backend, ns, None, embedding_model=None)
    assert backend.search(ns, unit(1), 5, "m1") == []


def test_same_exact_key_replaces_previous_entry(backend):
    ns = "ns1"
    ek = uuid.uuid4().hex
    old, _, _ = put(backend, ns, unit(1), ek=ek)
    new, _, _ = put(backend, ns, unit(2), ek=ek)
    assert backend.get(ns, old) is None and backend.get_exact(ns, ek).meta["entry_id"] == new and backend.count(ns) == 1


def test_ttl_expiry(backend):
    ns = "ns1"
    eid, ek, _ = put(backend, ns, unit(1), ttl=1)
    time.sleep(1.3)
    assert backend.get_exact(ns, ek) is None and backend.get(ns, eid) is None
    assert backend.search(ns, unit(1), 5, "m1") == []
    backend.cleanup_expired(ns)
    assert backend.count(ns) == 0


def test_eviction_removes_oldest_beyond_cap(backend):
    ns = "ns1"
    ids = []
    for i in range(5):
        eid, _, _ = put(backend, ns, unit(i), cap=3, created_at=1000.0 + i)
        ids.append(eid)
    assert backend.count(ns) == 3
    assert backend.get(ns, ids[0]) is None and backend.get(ns, ids[1]) is None and backend.get(ns, ids[4]) is not None


def test_epoch_rejects_stale_store(backend):
    ns = "ns1"
    e0 = backend.get_epoch(ns)
    assert backend.bump_epoch(ns) == e0 + 1
    _, ek, ok = put(backend, ns, unit(1), epoch=e0)  # generation started before the invalidation
    assert ok is False and backend.get_exact(ns, ek) is None
    _, _, ok2 = put(backend, ns, unit(1), epoch=backend.get_epoch(ns))
    assert ok2 is True


def test_delete_and_delete_matching(backend):
    ns = "ns1"
    a, ka, _ = put(backend, ns, unit(1), context_version="c1")
    b, kb, _ = put(backend, ns, unit(2), context_version="c2")
    c, kc, _ = put(backend, ns, unit(3), context_version="c2", model="other")
    assert backend.delete_matching(ns, {"context_version": "c2", "model": "gpt"}) == 1
    assert backend.get(ns, b) is None and backend.get_exact(ns, kb) is None and backend.get(ns, a) and backend.get(ns, c)
    assert backend.delete(ns, a) is True and backend.delete(ns, a) is False and backend.get_exact(ns, ka) is None
    assert backend.delete_matching(ns, {}) == 1 and backend.count(ns) == 0


def test_list_entries_has_hits_and_no_answer(backend):
    ns = "ns1"
    eid, _, _ = put(backend, ns, unit(1))
    backend.record_hit(ns, eid)
    backend.record_hit(ns, eid)
    (row,) = backend.list_entries(ns, 10)
    assert row["hits"] == 2 and row["entry_id"] == eid and "answer" not in row


def test_embedding_cache(backend):
    assert backend.emb_get("k") is None
    backend.emb_put("k", unit(3), 7, 60)
    v, t = backend.emb_get("k")
    assert t == 7 and np.allclose(v, unit(3))


def test_lock_is_exclusive_and_owner_only(backend):
    assert backend.lock_acquire("ns", "k", "tokA", 5) is True
    assert backend.lock_acquire("ns", "k", "tokB", 5) is False
    assert backend.lock_release("ns", "k", "tokB") is False  # not the owner
    assert backend.lock_acquire("ns", "k", "tokB", 5) is False  # still held by A
    assert backend.lock_release("ns", "k", "tokA") is True
    assert backend.lock_acquire("ns", "k", "tokB", 5) is True


def test_lock_expiry_does_not_let_old_owner_release_new_lock(backend):
    assert backend.lock_acquire("ns", "k", "old", 0.2)
    time.sleep(0.4)
    assert backend.lock_acquire("ns", "k", "new", 5)
    assert backend.lock_release("ns", "k", "old") is False
    assert backend.lock_acquire("ns", "k", "x", 5) is False  # "new" still holds it
    assert backend.lock_release("ns", "k", "new") is True


def test_flight_and_rate_window(backend):
    backend.flight_put("ns", "k", "payload", 5)
    assert backend.flight_get("ns", "k") == "payload" and backend.flight_get("ns", "zz") is None
    assert [backend.incr_window("ns", "w", 60) for _ in range(3)] == [1, 2, 3]
    assert backend.incr_window("other", "w", 60) == 1


def test_namespace_function_used_for_keys():
    assert len(namespace("t")) == 24
