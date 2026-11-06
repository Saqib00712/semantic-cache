# Failure modes (all covered by tests in `tests/failure` and `tests/concurrency` unless marked)

| Failure | Behavior |
|---|---|
| Redis down | `SEMCACHE_ON_CACHE_FAILURE=bypass` (default): answer from the model, no cache, metric `degraded`; `fail`: 503 `cache_unavailable`. Budget/quota still enforced (SQLite). `/ready` = 503. |
| Embedding call fails | Degrades to exact-match only; the entry is stored without a vector. |
| Embedding invalid (wrong size, NaN, zero) | Rejected; never stored or searched. |
| OpenAI rate limit / timeout / connection / 5xx | Retried with backoff up to `MAX_ATTEMPTS`, then 503/504/502. Nothing cached. |
| OpenAI auth / bad request / gateway 403 | No retry; 502 / 403 `policy_denied`. |
| Empty or truncated answer | Empty = 502; truncated is returned but not stored. |
| Leader fails during single-flight | Waiters are released and retry as leader; no permanent lock. |
| Lock TTL shorter than generation | A second leader may also generate (extra cost, not corruption); a late release cannot free the new owner's lock. |
| Invalidation during generation | The in-flight answer is returned to its caller but not stored (epoch mismatch). |
| Corrupt stored entry | Rejected, deleted, regenerated. |
| Process crash mid-store | The Lua put is atomic; an orphaned lock expires by TTL. |
| Unverified | RediSearch failures (no server available), Redis failover/replication lag, Redis Cluster (unsupported). |
