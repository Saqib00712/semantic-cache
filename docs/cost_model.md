# Cost model

Per request: `embedding_cost` always (a hit still embeds; embedding-cache hits cost 0) and `generation_cost` on a miss/ineligible request. Money is `Decimal` in code and integer pico-USD in SQLite.
- `gross_generation_cost_avoided` = for each hit, the **original generation cost stored with that entry** (an estimate of what a fresh call would cost).
- `estimated_net_savings` = gross avoided − all embedding cost − configured infra cost (`SEMCACHE_INFRA_COST_PER_REQUEST_USD`, default 0 = *not measured*; Redis/hosting cost is not included unless you set it).
- **Break-even hit rate** = embedding cost per request ÷ generation cost per request. Below it the cache loses money.
- Prices are configuration. Use your real model prices.
