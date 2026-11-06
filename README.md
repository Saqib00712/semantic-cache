# semantic-cache · Project 8

> **Production AI Semantic Cache & Cost Optimization System**  
> A smart cache layer in front of OpenAI — reuses answers for semantically similar questions, cuts redundant API calls, and enforces eligibility rules so stale or unsafe answers are never served.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![Redis Stack](https://img.shields.io/badge/Redis%20Stack-vector%20search-red.svg)](https://redis.io/docs/stack/)
[![Tests](https://img.shields.io/badge/tests-187%20pass-brightgreen.svg)](#test-results)
[![Ruff](https://img.shields.io/badge/lint-ruff%20clean-blue.svg)](https://docs.astral.sh/ruff/)

---

## What It Does

```
Client  ──► POST /v1/cache/query
                │
                ├─ Exact match?        ──► return cached answer immediately
                ├─ Semantic match?
                │    └─ Eligibility check  ──► pass → return cached answer
                │                          └─ fail → forward to OpenAI
                └─ Miss  ──► forward to OpenAI ──► cache result ──► return
```

> **A semantic match is never, by itself, a reason to reuse an answer.**  
> Every cache hit passes an eligibility gate that checks context version, prompt template version, tool version, temperature, personalization flags, and freshness. See `docs/cache_policy.md`.

---

## Phases Built

| Phase | What Was Built |
|-------|---------------|
| 0 | Architecture, design decisions, cache policy design |
| 1 | Foundation: FastAPI skeleton, config, `/health`, `/ready`, structured logging |
| 2 | Exact-match cache: SHA-256 key on normalized query + tenant + config hash |
| 3 | Embedding generation: OpenAI `text-embedding-3-small`, normalized vectors |
| 4 | Vector similarity search: Redis Stack (HNSW index) + scan fallback for plain Redis |
| 5 | Eligibility checker: version gates, temperature, personalization, TTL |
| 6 | Cache miss path: OpenAI call, store result, return |
| 7 | Multi-tenant isolation: per-tenant key namespace, separate stats |
| 8 | Stampede protection: coalescing — one OpenAI call for N concurrent identical queries |
| 9 | Cost accounting: token counting, USD saved vs spent per tenant |
| 10 | Cache invalidation: by entry ID, by context version, full flush |
| 11 | Admin API: tenant CRUD, key management, cache stats, invalidation |
| 12 | Prometheus `/metrics`: hit rate, miss rate, latency histograms, cost saved |
| 13 | Optional LLM-gateway route: forward misses through Project 6 instead of OpenAI directly |
| 14 | Web UI: query interface, hit/miss display, stats dashboard |
| 15 | Evaluation framework: hit rate, precision, eligibility accuracy, regression gate |
| 16 | Docker, CI files, docs, final report |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     semantic-cache                       │
│                                                          │
│  ┌────────────┐    ┌──────────────┐    ┌─────────────┐  │
│  │  Auth &    │    │  Exact Match │    │  Embedding  │  │
│  │  Tenants   │───►│  (SHA-256)   │    │  (OpenAI)   │  │
│  └────────────┘    └──────┬───────┘    └──────┬──────┘  │
│                           │ miss              │          │
│                           ▼                   ▼          │
│                   ┌──────────────┐    ┌─────────────┐   │
│                   │  Vector      │    │ Eligibility │   │
│                   │  Search      │───►│  Checker    │   │
│                   │ (Redis HNSW) │    └──────┬──────┘   │
│                   └──────────────┘           │ pass     │
│                                              ▼          │
│                   ┌──────────────────────────────────┐  │
│                   │  OpenAI / LLM-gateway (misses)   │  │
│                   └──────────────────────────────────┘  │
│                                                          │
│  Cost Accounting · Stampede Protection · Prometheus      │
└──────────────────────────────────────────────────────────┘
```

---

## Cache Status Values

| Status | Meaning |
|--------|---------|
| `exact_hit` | Query matched exactly — answer returned, no OpenAI call |
| `semantic_hit` | Similar query found, eligibility passed — answer returned |
| `miss` | No match — OpenAI called, result cached |
| `ineligible` | Similar query found but failed eligibility check — OpenAI called |
| `coalesced` | Duplicate concurrent query — waited for first request's result |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API framework | FastAPI + Uvicorn |
| Vector store | Redis Stack (HNSW index) / plain Redis (scan fallback) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Exact match | SHA-256 normalized key |
| Auth | HMAC-hashed tenant API keys, admin bearer token |
| Cost tracking | tiktoken + configurable pricing table |
| Observability | Prometheus client, structured JSON logs |
| Config | Pydantic Settings (`SEMCACHE_*` prefix) |
| Testing | pytest, pytest-asyncio, fakeredis |
| Lint | ruff |

---

## Project Structure

```
semantic-cache/
├── app/
│   ├── main.py              # create_app() factory
│   ├── config.py            # Pydantic settings (SEMCACHE_* vars)
│   ├── api/
│   │   ├── cache.py         # POST /v1/cache/query, GET entries/stats, DELETE
│   │   └── admin.py         # tenant CRUD, key management, invalidation
│   ├── auth/                # API key validation, tenant isolation
│   ├── cache/
│   │   ├── exact.py         # SHA-256 exact match
│   │   ├── semantic.py      # embedding + vector search
│   │   ├── eligibility.py   # version gates, TTL, personalization checks
│   │   ├── stampede.py      # request coalescing
│   │   └── store.py         # Redis read/write, TTL management
│   ├── generation/
│   │   ├── openai_client.py
│   │   └── gateway_client.py  # optional: route through llm-gateway
│   ├── cost/                # token counting, USD saved/spent ledger
│   ├── metrics.py           # Prometheus counters and histograms
│   └── ui/                  # web UI: query, hit/miss, stats
├── scripts/
│   ├── quality.py           # lint + tests + coverage + evaluation
│   ├── admin.py             # CLI: create-tenant, create-key
│   └── demo.py              # offline demo (no keys, no cost)
├── docs/
│   ├── architecture.md
│   ├── cache_policy.md      # eligibility rules — read this first
│   ├── security.md
│   ├── failure_modes.md
│   ├── operations.md
│   ├── cost_model.md
│   ├── gateway_integration.md
│   ├── evaluation_report.md
│   ├── definition_of_done.md
│   └── final_engineering_report.md
├── tests/                   # 187 tests
├── Dockerfile
├── docker-compose.yml       # redis-stack + api
├── .env.example
└── pyproject.toml
```

---

## Getting Started

### Quickstart — no Redis, no Docker (works on Windows)

```bash
pip install -r requirements.txt

cp .env.example .env
# Edit .env — set at minimum:
#   SEMCACHE_OPENAI_API_KEY=<your key>     ← never paste into chat
#   SEMCACHE_ADMIN_TOKEN=<32+ char secret>

python -m uvicorn app.main:create_app --factory --port 8080
```

Create a tenant and key:
```bash
python scripts/admin.py create-tenant "me"
# Copy the key shown — it appears ONCE
```

Open `http://127.0.0.1:8080/ui`, paste the key, ask a question, then ask it again in different words — you should see `semantic_hit`.

### Offline demo (no keys, no cost)

```bash
python scripts/demo.py
# Runs with fake OpenAI — shows exact hits, semantic hits, misses, coalescing
```

### Production — Docker with Redis Stack (vector search)

```bash
export SEMCACHE_OPENAI_API_KEY=...
export SEMCACHE_ADMIN_TOKEN=...

docker compose up --build
# Starts: redis-stack + api on port 8080
```

Plain Redis (no vector index, scan fallback):
```env
SEMCACHE_VECTOR_MODE=scan
```

Route misses through LLM-gateway (Project 6):
```env
SEMCACHE_GENERATION_BASE_URL=http://localhost:8000
```

---

## API Reference

### Query the cache

```bash
curl -X POST http://127.0.0.1:8080/v1/cache/query \
  -H "Authorization: Bearer $SEMCACHE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "model": "gpt-4o-mini",
    "context_version": "v1",
    "temperature": 0
  }'
```

Response:
```json
{
  "answer": "Paris.",
  "cache_status": "semantic_hit",
  "similarity_score": 0.97,
  "model": "gpt-4o-mini",
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
  "request_id": "req_..."
}
```

### Cache management

```bash
# List cached entries
GET /v1/cache/entries

# Per-tenant stats (hit rate, cost saved)
GET /v1/cache/stats

# Invalidate one entry
DELETE /v1/cache/entries/{id}

# Invalidate by context version (e.g. after a prompt update)
DELETE /v1/cache?context_version=v1

# Full flush for a tenant
DELETE /v1/cache?all=true
```

---

## Eligibility Rules

Before returning a semantic match, the eligibility checker verifies:

| Check | Fail behaviour |
|-------|---------------|
| `context_version` mismatch | Miss — forward to OpenAI |
| `prompt_template_version` mismatch | Miss |
| `tool_version` mismatch | Miss |
| `temperature` > threshold | Ineligible — high-temp queries need fresh answers |
| `personalized=true` | Ineligible — user-specific queries must not be reused across users |
| Entry TTL expired | Miss |

Full policy: `docs/cache_policy.md`.

---

## Cost Accounting

Every request records:

- **Cost saved:** tokens that would have been used if the cache missed
- **Cost spent:** tokens actually used (embeddings + any OpenAI call)
- **Net saving:** per tenant, cumulative

```bash
GET /v1/cache/stats
# → { "hit_rate": 0.73, "cost_saved_usd": 1.42, "cost_spent_usd": 0.18, ... }
```

---

## Quality Gate

```bash
python scripts/quality.py
# Runs: ruff lint + format check + 187 tests + evaluation suite
```

The evaluation suite measures hit rate, precision (no wrong answers served), and eligibility accuracy on a fixed test set — results in `docs/evaluation_report.md`.

---

## Test Results

| Metric | Result |
|--------|--------|
| Tests | **187 pass** |
| Lint | ruff clean |
| Evaluation | hit rate, precision, eligibility accuracy — see `docs/evaluation_report.md` |

---

## Key Concepts

**Semantic similarity ≠ safe to reuse** — Two questions can be 97% similar but need different answers (different context version, different user, high temperature). The eligibility gate is what makes the cache safe, not the similarity score alone.

**Stampede protection** — When 50 clients ask the same question at once, only one OpenAI call is made. The other 49 wait and get the same result — `cache_status: coalesced`.

**Version-based invalidation** — When you update your prompt template, set `context_version=v2` and old cached answers are automatically ineligible. No manual flush needed.

**Redis Stack vs plain Redis** — Redis Stack runs the HNSW vector index for fast ANN search. Plain Redis (`SEMCACHE_VECTOR_MODE=scan`) does a linear scan — works fine for small caches, slower at scale.

**Honest limitations** — Single-host Redis (no HA), embeddings cost money (every miss + every new query), scan mode degrades at scale, Docker files written but not run by the author.

---

## Related Projects

| Project | Role |
|---------|------|
| [llm-gateway](../llm-gateway) | Optional: route cache misses through the gateway for cost tracking + circuit breaker |
| [guardrail-middleware](../guardrail-middleware) | Inspect queries and answers before caching |
| [observability-dashboard](../observability-dashboard) | Ingest Prometheus metrics from the cache |
| [eval-harness](../eval-harness) | Run the cache evaluation suite in CI |

---

## Certifications

Built as part of a self-directed 43-project AI Engineering curriculum underpinned by IBM / Coursera certifications in Generative AI, RAG, Agentic Workflows, and LLM deployment.

---

## Author

**Muhammad Saqib** · [GitHub](https://github.com/Saqib00712) · [LinkedIn](https://linkedin.com/in/muhammad-saqib-68b9b3374)
