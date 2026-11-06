# LLM Gateway integration (HTTP contract)

No code is imported from any gateway repository. Set `SEMCACHE_GENERATION_BASE_URL` (e.g. `http://gateway:8000/v1`) and `SEMCACHE_GENERATION_API_KEY`.

**Requests**: `POST {base}/chat/completions` (OpenAI schema) with `Authorization: Bearer <gateway key>`, `X-Request-ID` and `X-Correlation-ID` equal to the cache request id (the client's `X-Request-ID` if it sent a safe one).
**Responses expected**: OpenAI chat-completion JSON with `choices[0].message.content`, `finish_reason`, `usage`.
**Policy is never bypassed**: 401/403/429/5xx from the gateway are returned to the caller (`upstream_auth_failed`, `policy_denied`, `upstream_rate_limited`, `upstream_error`); there is no direct-to-OpenAI fallback. Only transient errors retry, using the configured limit.
**Cache hits do not reach the gateway**, so the gateway sees only misses; account for that in its per-key quotas and analytics.
**Optional**: `GET {root}/health` is used by `app.integrations.gateway.gateway_ready`.
Tested against `tests/integration/mock_gateway.py` over real HTTP (headers preserved, denial surfaced and not retried, wrong key, gateway down). Not tested against the real LLM gateway project.
