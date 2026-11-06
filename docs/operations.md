# Operations

**Run (Windows or no Redis)**: copy `.env.example` to `.env`, set `SEMCACHE_BACKEND=memory`, your key and an admin token, then
`python -m uvicorn app.main:create_app --factory --port 8080`. Memory mode is single-process and loses the cache on restart.
**Run (Redis)**: `docker compose up --build` (redis-stack gives vector search). On plain Redis set `SEMCACHE_VECTOR_MODE=scan`.
**Tenants**: `POST /admin/tenants` (Bearer admin token) or `python scripts/admin.py create-tenant NAME`.
**Endpoints**: `POST /v1/cache/query`, `GET /v1/cache/entries|stats`, `DELETE /v1/cache/entries/{id}`, `DELETE /v1/cache?<filter>=<v>` or `?all=true`, `/admin/*`, `/health`, `/ready`, `/metrics`, `/ui`.
**Monitor**: `semcache_requests_total{cache_status}`, `semcache_rejections_total{reason}` (a spike in one reason tells you what is blocking hits), `semcache_degraded_total{cause}`, `semcache_errors_total`, latency histograms.
**Set real prices** (`SEMCACHE_PRICE_*`); defaults are placeholders and savings figures are only as good as these.
**Changing the embedding model/dimension**: a new RediSearch index name is used automatically; old entries are not matched (wrong model tag) and age out by TTL.
**Backups**: the cache is disposable; back up `data/semcache.db` (tenants, keys, ledger).
