"""In-process backend. Same contract as Redis; state lives in this Python process only (lost on exit, not shared between
processes). Used by tests, and as the no-Redis mode on Windows. Thread-safe through one re-entrant lock."""

import threading
import time

import numpy as np

from app.storage.backend import Backend, StoredEntry


class MemoryBackend(Backend):
    name = "memory"
    vector_mode = "scan"

    def __init__(self, clock=time.monotonic):
        self._lock = threading.RLock()
        self._clock = clock
        self._entries: dict[str, dict[str, dict]] = {}  # ns -> id -> {meta, answer, vec, hits, exp, created}
        self._exact: dict[tuple[str, str], tuple[str, float]] = {}
        self._epoch: dict[str, int] = {}
        self._emb: dict[str, tuple[np.ndarray, int, float]] = {}
        self._locks: dict[tuple[str, str], tuple[str, float]] = {}
        self._flight: dict[tuple[str, str], tuple[str, float]] = {}
        self._windows: dict[tuple[str, str], tuple[int, float]] = {}

    def ping(self) -> bool:
        return True

    def _live(self, ns: str, entry_id: str) -> dict | None:
        e = self._entries.get(ns, {}).get(entry_id)
        if e is None:
            return None
        if e["exp"] <= self._clock():
            self._drop(ns, entry_id)
            return None
        return e

    def _drop(self, ns: str, entry_id: str) -> None:
        e = self._entries.get(ns, {}).pop(entry_id, None)
        if e is not None:
            xk = (ns, e["meta"].get("exact_key", ""))
            if self._exact.get(xk, ("", 0))[0] == entry_id:
                del self._exact[xk]

    @staticmethod
    def _out(e: dict) -> StoredEntry:
        return StoredEntry(meta=dict(e["meta"]), answer=e["answer"], vec=e["vec"], hits=e["hits"])

    def get_exact(self, ns, exact_key):
        with self._lock:
            ref = self._exact.get((ns, exact_key))
            e = self._live(ns, ref[0]) if ref else None
            return self._out(e) if e else None

    def get(self, ns, entry_id):
        with self._lock:
            e = self._live(ns, entry_id)
            return self._out(e) if e else None

    def put(self, ns, entry_id, exact_key, meta, answer, vec, ttl, max_entries, expect_epoch):
        with self._lock:
            if self._epoch.get(ns, 0) != expect_epoch:
                return False  # an invalidation happened while this answer was being generated: do not store it
            old = self._exact.get((ns, exact_key))
            if old and old[0] != entry_id:
                self._drop(ns, old[0])
            now = self._clock()
            self._entries.setdefault(ns, {})[entry_id] = {"meta": dict(meta), "answer": answer, "vec": vec, "hits": 0, "exp": now + ttl,
                                                          "created": now}  # fmt: skip
            self._exact[(ns, exact_key)] = (entry_id, now + ttl)
            bucket = self._entries[ns]
            while len(bucket) > max_entries:
                self._drop(ns, min(bucket, key=lambda i: bucket[i]["created"]))
            return True

    def search(self, ns, vec, k, embedding_model):
        with self._lock:
            scored = []
            for entry_id in list(self._entries.get(ns, {})):
                e = self._live(ns, entry_id)
                if e is None or e["vec"] is None or e["meta"].get("embedding_model") != embedding_model or e["vec"].shape != vec.shape:
                    continue
                scored.append((entry_id, float(np.dot(e["vec"], vec))))
            return sorted(scored, key=lambda t: -t[1])[:k]

    def delete(self, ns, entry_id):
        with self._lock:
            existed = entry_id in self._entries.get(ns, {})
            self._drop(ns, entry_id)
            return existed

    def delete_matching(self, ns, filters):
        with self._lock:
            doomed = [i for i, e in self._entries.get(ns, {}).items() if all(e["meta"].get(k) == v for k, v in filters.items())]
            for i in doomed:
                self._drop(ns, i)
            return len(doomed)

    def list_entries(self, ns, limit):
        with self._lock:
            out = []
            for i in list(self._entries.get(ns, {})):
                e = self._live(ns, i)
                if e:
                    out.append({**e["meta"], "hits": e["hits"]})
            return sorted(out, key=lambda m: -m.get("created_at", 0))[:limit]

    def count(self, ns):
        with self._lock:
            return sum(1 for i in list(self._entries.get(ns, {})) if self._live(ns, i))

    def record_hit(self, ns, entry_id):
        with self._lock:
            e = self._live(ns, entry_id)
            if e:
                e["hits"] += 1

    def bump_epoch(self, ns):
        with self._lock:
            self._epoch[ns] = self._epoch.get(ns, 0) + 1
            return self._epoch[ns]

    def get_epoch(self, ns):
        with self._lock:
            return self._epoch.get(ns, 0)

    def cleanup_expired(self, ns):
        with self._lock:
            now, n = self._clock(), 0
            for i in list(self._entries.get(ns, {})):
                if self._entries[ns][i]["exp"] <= now:
                    self._drop(ns, i)
                    n += 1
            return n

    def emb_get(self, key):
        with self._lock:
            v = self._emb.get(key)
            if v is None or v[2] <= self._clock():
                self._emb.pop(key, None)
                return None
            return v[0], v[1]

    def emb_put(self, key, vec, tokens, ttl):
        with self._lock:
            self._emb[key] = (vec, tokens, self._clock() + ttl)

    def lock_acquire(self, ns, key, token, ttl):
        with self._lock:
            held = self._locks.get((ns, key))
            if held and held[1] > self._clock():
                return False
            self._locks[(ns, key)] = (token, self._clock() + ttl)
            return True

    def lock_release(self, ns, key, token):
        with self._lock:
            held = self._locks.get((ns, key))
            if held and held[0] == token:  # only the owner can release; a stale owner whose lock expired must not free someone else's
                del self._locks[(ns, key)]
                return True
            return False

    def flight_put(self, ns, key, payload, ttl):
        with self._lock:
            self._flight[(ns, key)] = (payload, self._clock() + ttl)

    def flight_get(self, ns, key):
        with self._lock:
            v = self._flight.get((ns, key))
            return v[0] if v and v[1] > self._clock() else None

    def incr_window(self, ns, name, ttl):
        with self._lock:
            n, exp = self._windows.get((ns, name), (0, 0.0))
            if exp <= self._clock():
                n, exp = 0, self._clock() + ttl
            self._windows[(ns, name)] = (n + 1, exp)
            return n + 1
