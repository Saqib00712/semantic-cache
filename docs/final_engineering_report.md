# Final engineering report

## What was run (build environment: Linux, Python 3.13, plain Redis 7.0.15 with **no RediSearch**)
- Clean virtualenv from `requirements-dev.txt`, `pip check` clean, `python -m pytest`: **187 passed** (unit, integration on memory **and** real Redis, concurrency, security, failure, gateway, evaluation).
- `python scripts/quality.py` (ruff check, ruff format --check, pytest, evaluation): passed. Coverage of `app/`: 90% overall (redis_backend 88%, tracing 52%, backend base class 62%).
- Manual end-to-end: real `uvicorn` process + real `redis-server` (scan mode) + mock gateway over HTTP + mock embeddings: ready check ok; miss, exact_hit, semantic_hit (0.9186), miss on changed number; the gateway received 2 calls for 4 requests with request ids forwarded.
- `scripts/demo.py`: 5 requests, 3 model calls (exact hit, semantic hit, number-change miss, personal request bypassed).
- Evaluation (mock embedder, 28 cases): 28/28 expected outcomes; **0** unsafe hits with the pipeline vs **8** a similarity-only cache would serve at threshold 0.8.

## NOT verified
Docker build/compose, GitHub Actions, RediSearch (HNSW) mode (command text unit-tested only), real OpenAI embeddings/generation, a real LLM gateway, Windows, Redis Cluster/failover, any performance or load figure, `pip-audit` (CI step only).

## Bug found by testing
`SEMCACHE_BLOCKED_RESPONSE_PATTERNS` was split on commas, which broke the default regex (`{12,}`) so keys in answers were never blocked. Now separated by `;;`, with a regression test.

## Phases
| # | Phase | Implemented in | Key tests |
|---|---|---|---|
| 1 | Foundation/config | `config.py`, `main.py`, `observability/`, `api/` | `test_misc.py` |
| 2 | Exact-match cache | `cache/{normalize,keys}.py`, `storage/` | `test_normalize_keys.py`, `test_backend_contract.py` |
| 3 | OpenAI embeddings | `embeddings/service.py` | `test_services.py` |
| 4 | Vector search | `similarity/vector.py`, `storage/redis_backend.py`, `memory.py` | contract tests (both backends) |
| 5 | Eligibility | `eligibility/{engine,personal}.py` | `test_eligibility.py` |
| 6 | Generation | `generation/service.py` | `test_services.py`, `test_failures.py` |
| 7 | `POST /v1/cache/query` | `main.py`, `cache/service.py` | `test_api_flow.py` |
| 8 | Invalidation/lifecycle | `cache/lifecycle.py` | `test_api_flow.py`, concurrency invalidation race |
| 9 | Multi-tenant security | `security/auth.py`, namespaces | `tests/security/` |
| 10 | Single-flight | `cache/service.py`, backend locks | `tests/concurrency/` |
| 11 | Cost analytics | `storage/ledger.py`, `/v1/cache/stats` | `test_api_flow.py`, `docs/cost_model.md` |
| 12 | Evaluation | `evaluation/` | `test_evaluation.py`, `evaluation/report.md` |
| 13 | Observability | `observability/` | `test_misc.py`, `test_security.py` |
| 14 | Failure injection | `tests/failure/` | Redis down, embedding failure, OpenAI errors, corruption, invalid dims |
| 15 | Gateway adapter | `integrations/gateway.py`, `docs/gateway_integration.md` | `test_gateway.py` |
| 16 | Docker/CI/docs | `Dockerfile`, `docker-compose.yml`, `.github/`, `docs/` | not run |

## Architecture decisions, limitations, concepts
See `architecture.md`, `cache_policy.md`, `failure_modes.md`, `security.md`. Concepts to be able to explain: why similarity alone is unsafe; exact key vs semantic candidate vs eligibility verdict; the critical-signature guard and its limits; single-flight and lock ownership; epoch-based invalidation races; fail-open vs fail-closed; net vs gross savings and break-even hit rate; why entries store their own metadata; why scan vs RediSearch.

## Limitations
Signature heuristic; mock-embedder evaluation; scan mode is O(entries per tenant); memory backend is single-process; per-process rate-limit fallback; one SQLite writer; plaintext values in Redis; budget can overshoot by one in-flight generation per concurrent request; prices are placeholders until configured.

## Manual verification checklist (for you)
1. `python scripts/demo.py` (no keys needed). 2. Windows: `python -m pytest` (Redis tests skip without `redis-server`). 3. Set your key in `.env`, run with `SEMCACHE_BACKEND=memory`, open `/ui`, ask a question and a paraphrase. 4. `docker compose up --build`, then check `/ready` shows `vector_mode: search`. 5. `python -m evaluation.run_eval --live` with your key.
