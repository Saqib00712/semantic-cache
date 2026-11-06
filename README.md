# semantic-cache

A semantic cache in front of OpenAI chat models: exact and similarity lookup, **eligibility checks before any answer is reused**, multi-tenant isolation, stampede protection, cost accounting, metrics, and an optional LLM-gateway route for misses.

> A semantic match is never, by itself, a reason to reuse an answer. See `docs/cache_policy.md`.

## Quick start (no Redis, no Docker: works on Windows)
```
pip install -r requirements.txt
copy .env.example .env      # then set SEMCACHE_OPENAI_API_KEY and SEMCACHE_ADMIN_TOKEN (32+ chars) in .env
python -m uvicorn app.main:create_app --factory --port 8080
```
Create a tenant: `python scripts/admin.py create-tenant "me"` (key shown once). Open `http://127.0.0.1:8080/ui`, paste the key, ask, then ask a reworded version.
Offline demo with no keys: `python scripts/demo.py`.

## API
`POST /v1/cache/query` `{"query": "...", "model"?, "context_version"?, "prompt_template_version"?, "tool_version"?, "temperature"?, "max_output_tokens"?, "personalized"?}`
-> `{"answer","cache_status": exact_hit|semantic_hit|miss|ineligible|coalesced,"similarity_score","model","usage","request_id"}`.
Also `GET /v1/cache/entries|stats`, `DELETE /v1/cache/entries/{id}`, `DELETE /v1/cache?context_version=..` / `?all=true`, `/admin/*`, `/health`, `/ready`, `/metrics`.

## Production-like
`docker compose up --build` (redis-stack for vector search). Plain Redis: `SEMCACHE_VECTOR_MODE=scan`. Gateway: `SEMCACHE_GENERATION_BASE_URL` (see `docs/gateway_integration.md`).

## Quality
`python scripts/quality.py` runs lint, format check, 187 tests and the evaluation. Honest status of what was and was not verified: `docs/definition_of_done.md` and `docs/final_engineering_report.md`.

## Docs
architecture, cache_policy, security, failure_modes, operations, cost_model, gateway_integration, evaluation_report, definition_of_done, final_engineering_report (all in `docs/`).
