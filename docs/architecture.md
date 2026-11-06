# Architecture

```
client ──HTTP──> FastAPI (app/main.py)
                   │ auth (API key -> tenant)            app/security/auth.py   (SQLite: tenants, key hashes)
                   │ rate limit / daily quota / budget   app/cache/service.py  (rate: store; quota+budget: SQLite)
                   ▼
            handle_query (app/cache/service.py)
   normalize ─ signature ─ exact key ─ bypass check (personal data / temperature / caller flag)
        │ eligible
        ├─ exact lookup ──────────────► hit (exact_hit)
        ├─ embedding (cache, retries) ─► vector search (top-k, tenant-scoped)
        │        └─ eligibility engine per candidate ► hit (semantic_hit) or named rejection
        └─ miss ─ single-flight lock ─► generation (OpenAI SDK, or LLM gateway) ─ validate ─ store (epoch-guarded)
                                      └ waiters get the leader's result (coalesced)
   every request -> one ledger row (SQLite) + metrics + a log line without prompt/answer/key
```

| Layer | Module | Notes |
|---|---|---|
| Storage contract | `app/storage/backend.py` | Two implementations behind one interface: Redis and in-memory (single process; Windows quick start, tests). |
| Redis | `app/storage/redis_backend.py` | Key schema in the module docstring. Lua script makes "epoch check + replace old entry + write + index" atomic. |
| Vector search | `redis_backend.py` | `search` mode = RediSearch HNSW (redis-stack). `scan` mode = numpy over one tenant's vectors (plain Redis). Auto-detected. |
| Embeddings | `app/embeddings/service.py` | OpenAI or a deterministic mock; validated (dimension, NaN, zero), cached by (model, dim, text hash). |
| Eligibility | `app/eligibility/` | Pure functions returning a verdict with a reason. |
| Generation | `app/generation/service.py` | Error classification; only transient errors retry. |
| Lifecycle | `app/cache/lifecycle.py` | Invalidation always bumps the tenant epoch first. |
| Metadata/cost | `app/storage/{database,ledger}.py` | SQLite only: tenants, key hashes, usage ledger (integer pico-USD). Never stores prompts or answers. |
| Observability | `app/observability/` | JSON logs with an allow-list of fields, Prometheus text, OTel spans. |
| Gateway | `app/integrations/gateway.py` | HTTP contract only. See `gateway_integration.md`. |

## Decisions
- **Functions + Pydantic models**; classes only where state or an interface demands (backends, runtime holder, mock gateway).
- **Entry metadata is self-describing.** Everything the eligibility decision uses is stored with the entry (`EntryMeta`), so the decision is reproducible and a malformed entry is rejected, not trusted.
- **Two vector modes** because RediSearch needs redis-stack. Scan is exact but O(entries per tenant): fine for thousands, not millions.
- **Budget and quota live in SQLite**, so they hold even when Redis is down. Rate limiting falls back to an in-process counter (per-process, so weaker with several workers).
- **Embeddings never go through the gateway**; generation does when `SEMCACHE_GENERATION_BASE_URL` is set.
- **Redis Cluster is not supported** (the put script touches keys in several slots).
