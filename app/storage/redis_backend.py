"""Redis backend. Key layout (every key carries the tenant namespace `ns`, a hash of the tenant id):

  sc:e:{ns}:{entry_id}   HASH  meta (JSON), answer, vec (float32 bytes), ns, emb_model, hits     TTL = entry ttl
  sc:x:{ns}:{exact_key}  STRING entry_id                                                           TTL = entry ttl
  sc:ids:{ns}            ZSET   entry ids scored by creation time (eviction order, listing, size)
  sc:epoch:{ns}          STRING invalidation counter (see put: an in-flight answer from before an invalidation is not stored)
  sc:emb:{sha}           HASH   vec, tokens                                                        embedding cache (text hash, model)
  sc:lock:{ns}:{key}     STRING owner token, SET NX PX                                             single-flight lock
  sc:flight:{ns}:{key}   STRING short-lived result handed to waiting identical requests
  sc:rl:{ns}:{name}      STRING rate-limit window counter

Vector search has two modes, chosen at startup (SEMCACHE_VECTOR_MODE=auto|search|scan):
  search: the RediSearch module (redis-stack): FT.CREATE ... VECTOR HNSW COSINE and a KNN query pre-filtered by tenant tag.
  scan:   works on plain Redis: read the tenant's vectors and score them with numpy (exact, O(entries per tenant)).
Redis Cluster is not supported (the multi-key Lua script spans hash slots)."""

import numpy as np
import redis

from app.storage.backend import Backend, BackendError, StoredEntry

INDEX_SCHEMA_VERSION = 1
MAX_SCAN = 50_000

_PUT = """
local cur = redis.call('GET', KEYS[1])
if not cur then cur = '0' end
if cur ~= ARGV[1] then return 0 end
local old = redis.call('GET', KEYS[3])
if old and old ~= ARGV[3] then
  redis.call('DEL', ARGV[10] .. old)
  redis.call('ZREM', KEYS[4], old)
end
redis.call('DEL', KEYS[2])
redis.call('HSET', KEYS[2], 'meta', ARGV[4], 'answer', ARGV[5], 'ns', ARGV[7], 'emb_model', ARGV[8], 'hits', 0)
if ARGV[6] ~= '' then redis.call('HSET', KEYS[2], 'vec', ARGV[6]) end
redis.call('EXPIRE', KEYS[2], ARGV[2])
redis.call('SET', KEYS[3], ARGV[3], 'EX', ARGV[2])
redis.call('ZADD', KEYS[4], ARGV[9], ARGV[3])
return 1
"""
_RELEASE = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) else return 0 end"


def _tag_escape(value: str) -> str:
    return "".join("\\" + c if not (c.isalnum() or c == "_") else c for c in value)


def _guard(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except redis.RedisError as e:
            raise BackendError(type(e).__name__) from None

    wrapper.__name__ = fn.__name__
    return wrapper


class RedisBackend(Backend):
    name = "redis"

    def __init__(self, client: redis.Redis, vector_mode: str = "auto", dim: int = 1536):
        self.r, self.dim = client, dim
        self._put = client.register_script(_PUT)
        self._release = client.register_script(_RELEASE)
        self.search_available = self._detect_search()
        if vector_mode == "search" and not self.search_available:
            raise BackendError("vector_mode=search but this Redis has no RediSearch module (use redis-stack, or vector_mode=scan)")
        self.vector_mode = "search" if (vector_mode == "search" or (vector_mode == "auto" and self.search_available)) else "scan"
        self.index = f"sc:idx:v{INDEX_SCHEMA_VERSION}:{dim}"
        if self.vector_mode == "search":
            self._ensure_index()

    # ---- vector search capability -------------------------------------------------------------------------------------------
    def _detect_search(self) -> bool:
        try:
            self.r.execute_command("FT._LIST")
            return True
        except redis.ResponseError:
            return False  # "unknown command": no RediSearch module
        except redis.RedisError:
            return False

    def create_index_args(self) -> list:
        """FT.CREATE arguments. Cosine distance on float32 vectors; HNSW graph. The index name embeds the schema version and the
        dimension: changing either creates a NEW index (old entries are simply not in it and age out by TTL), which is the migration."""
        return [self.index, "ON", "HASH", "PREFIX", "1", "sc:e:", "SCHEMA", "ns", "TAG", "emb_model", "TAG", "vec", "VECTOR", "HNSW", "6",
                "TYPE", "FLOAT32", "DIM", str(self.dim), "DISTANCE_METRIC", "COSINE"]  # fmt: skip

    def _ensure_index(self) -> None:
        try:
            self.r.execute_command("FT.INFO", self.index)
        except redis.ResponseError:
            self.r.execute_command("FT.CREATE", *self.create_index_args())

    def knn_query(self, ns: str, embedding_model: str, k: int) -> str:
        return f"(@ns:{{{ns}}} @emb_model:{{{_tag_escape(embedding_model)}}})=>[KNN {k} @vec $q AS score]"

    # ---- helpers ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _ekey(ns: str, entry_id: str) -> str:
        return f"sc:e:{ns}:{entry_id}"

    @staticmethod
    def _decode(h: dict) -> StoredEntry | None:
        if not h or b"meta" not in h:
            return None
        import json

        try:
            meta = json.loads(h[b"meta"])
        except ValueError:
            return StoredEntry(meta={"_corrupt": True}, answer="", vec=None)  # the engine rejects it; the caller deletes it
        vec = np.frombuffer(h[b"vec"], dtype=np.float32) if b"vec" in h else None
        return StoredEntry(meta=meta, answer=h.get(b"answer", b"").decode("utf-8", "replace"), vec=vec, hits=int(h.get(b"hits", 0)))

    # ---- backend contract ---------------------------------------------------------------------------------------------------
    @_guard
    def ping(self) -> bool:
        return bool(self.r.ping())

    @_guard
    def get_exact(self, ns, exact_key):
        entry_id = self.r.get(f"sc:x:{ns}:{exact_key}")
        return self.get(ns, entry_id.decode()) if entry_id else None

    @_guard
    def get(self, ns, entry_id):
        return self._decode(self.r.hgetall(self._ekey(ns, entry_id)))

    @_guard
    def put(self, ns, entry_id, exact_key, meta, answer, vec, ttl, max_entries, expect_epoch):
        import json

        ok = self._put(
            keys=[f"sc:epoch:{ns}", self._ekey(ns, entry_id), f"sc:x:{ns}:{exact_key}", f"sc:ids:{ns}"],
            args=[str(expect_epoch), ttl, entry_id, json.dumps(meta), answer, vec.astype(np.float32).tobytes() if vec is not None else "",
                  ns, meta.get("embedding_model") or "none", meta["created_at"], f"sc:e:{ns}:"],
        )  # fmt: skip
        if not ok:
            return False
        over = self.r.zcard(f"sc:ids:{ns}") - max_entries
        if over > 0:
            self._delete_ids(ns, [i.decode() for i in self.r.zrange(f"sc:ids:{ns}", 0, over - 1)])
        return True

    def _delete_ids(self, ns: str, ids: list[str]) -> int:
        n = 0
        for entry_id in ids:
            key = self._ekey(ns, entry_id)
            raw = self.r.hget(key, "meta")
            pipe = self.r.pipeline()
            if raw:
                import json

                try:
                    xk = json.loads(raw).get("exact_key")
                except ValueError:
                    xk = None
                if xk and (self.r.get(f"sc:x:{ns}:{xk}") or b"").decode() == entry_id:
                    pipe.delete(f"sc:x:{ns}:{xk}")
            pipe.delete(key)
            pipe.zrem(f"sc:ids:{ns}", entry_id)
            res = pipe.execute()
            n += 1 if res[-2] or res[-1] else 0
        return n

    @_guard
    def search(self, ns, vec, k, embedding_model):
        q = vec.astype(np.float32)
        if self.vector_mode == "search":
            res = self.r.execute_command("FT.SEARCH", self.index, self.knn_query(ns, embedding_model, k), "PARAMS", "2", "q", q.tobytes(),
                                         "SORTBY", "score", "RETURN", "1", "score", "DIALECT", "2")  # fmt: skip
            out = []
            for i in range(1, len(res), 2):
                key = res[i].decode()
                fields = res[i + 1]
                score = float(fields[fields.index(b"score") + 1]) if b"score" in fields else 1.0
                out.append((key.rsplit(":", 1)[1], 1.0 - score))  # COSINE distance -> cosine similarity
            return out
        ids = [i.decode() for i in self.r.zrange(f"sc:ids:{ns}", 0, MAX_SCAN - 1)]
        if not ids:
            return []
        pipe = self.r.pipeline()
        for i in ids:
            pipe.hmget(self._ekey(ns, i), "vec", "emb_model")
        rows, keep, mats, stale = pipe.execute(), [], [], []
        for i, (raw, model) in zip(ids, rows, strict=True):
            if raw is None:
                stale.append(i)  # its hash expired: drop the leftover index entry
            elif model is not None and model.decode() == embedding_model and len(raw) == q.shape[0] * 4:
                keep.append(i)
                mats.append(np.frombuffer(raw, dtype=np.float32))
        if stale:
            self.r.zrem(f"sc:ids:{ns}", *stale)
        if not keep:
            return []
        scores = np.vstack(mats) @ q
        order = np.argsort(-scores)[:k]
        return [(keep[int(j)], float(scores[int(j)])) for j in order]

    @_guard
    def delete(self, ns, entry_id):
        return self._delete_ids(ns, [entry_id]) > 0

    @_guard
    def delete_matching(self, ns, filters):
        import json

        doomed = []
        for entry_id in (i.decode() for i in self.r.zrange(f"sc:ids:{ns}", 0, -1)):
            raw = self.r.hget(self._ekey(ns, entry_id), "meta")
            try:
                meta = json.loads(raw) if raw else None
            except ValueError:
                meta = {}
            if meta is not None and all(meta.get(k) == v for k, v in filters.items()):
                doomed.append(entry_id)
        return self._delete_ids(ns, doomed)

    @_guard
    def list_entries(self, ns, limit):
        import json

        out = []
        for entry_id in (i.decode() for i in self.r.zrevrange(f"sc:ids:{ns}", 0, limit - 1)):
            raw, hits = self.r.hmget(self._ekey(ns, entry_id), "meta", "hits")
            if raw:
                try:
                    out.append({**json.loads(raw), "hits": int(hits or 0)})
                except ValueError:
                    continue
        return out

    @_guard
    def count(self, ns):
        return int(self.r.zcard(f"sc:ids:{ns}"))

    @_guard
    def record_hit(self, ns, entry_id):
        self.r.hincrby(self._ekey(ns, entry_id), "hits", 1) if self.r.exists(self._ekey(ns, entry_id)) else None

    @_guard
    def bump_epoch(self, ns):
        return int(self.r.incr(f"sc:epoch:{ns}"))

    @_guard
    def get_epoch(self, ns):
        return int(self.r.get(f"sc:epoch:{ns}") or 0)

    @_guard
    def cleanup_expired(self, ns):
        ids = [i.decode() for i in self.r.zrange(f"sc:ids:{ns}", 0, -1)]
        gone = [i for i in ids if not self.r.exists(self._ekey(ns, i))]
        if gone:
            self.r.zrem(f"sc:ids:{ns}", *gone)
        return len(gone)

    @_guard
    def emb_get(self, key):
        h = self.r.hgetall(f"sc:emb:{key}")
        return (np.frombuffer(h[b"vec"], dtype=np.float32), int(h[b"tokens"])) if h and b"vec" in h else None

    @_guard
    def emb_put(self, key, vec, tokens, ttl):
        pipe = self.r.pipeline()
        pipe.hset(f"sc:emb:{key}", mapping={"vec": vec.astype(np.float32).tobytes(), "tokens": tokens})
        pipe.expire(f"sc:emb:{key}", ttl)
        pipe.execute()

    @_guard
    def lock_acquire(self, ns, key, token, ttl):
        return bool(self.r.set(f"sc:lock:{ns}:{key}", token, nx=True, px=max(1, int(ttl * 1000))))

    @_guard
    def lock_release(self, ns, key, token):
        return bool(self._release(keys=[f"sc:lock:{ns}:{key}"], args=[token]))  # compare-and-delete: only the owner releases

    @_guard
    def flight_put(self, ns, key, payload, ttl):
        self.r.set(f"sc:flight:{ns}:{key}", payload, ex=ttl)

    @_guard
    def flight_get(self, ns, key):
        v = self.r.get(f"sc:flight:{ns}:{key}")
        return v.decode() if v else None

    @_guard
    def incr_window(self, ns, name, ttl):
        pipe = self.r.pipeline()
        pipe.incr(f"sc:rl:{ns}:{name}")
        pipe.expire(f"sc:rl:{ns}:{name}", ttl)
        return int(pipe.execute()[0])

    def close(self) -> None:
        self.r.close()
