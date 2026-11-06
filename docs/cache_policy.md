# Cache policy

A semantic match is a **candidate**, never a decision.

**Skips the cache entirely (no lookup, no store)**: `personalized: true`; temperature above `SEMCACHE_MAX_CACHEABLE_TEMPERATURE` (0.3); query longer than `MAX_QUERY_CHARS`; personal-data signals (email, phone, "my <order/account/password/...>", self-statements, mixed identifiers, 5+ digit numbers).

**Exact key** = SHA-256 of canonical JSON: schema, tenant, model, system-prompt version, template version, context version, tool version, params hash, normalized query. Normalization is NFC + line endings + whitespace only (no lower-casing, punctuation or number changes).

**Semantic hit requires all of**: valid metadata, same tenant, active, unexpired, not personalized, same embedding model + dimension, same model / system prompt / template / context / tool versions / params, similarity ≥ threshold, and an equal *critical signature* (numbers, negation, quoted strings, identifiers). Each failure has a named reason (`semcache_rejections_total{reason}`).

**Stored only if**: finish_reason is `stop`, answer matches no blocked pattern (default: OpenAI-key-like strings), the tenant epoch did not change during generation.

**Lifecycle**: TTL; per-tenant cap with oldest-first eviction; invalidation by entry, by filter (model, context_version, prompt_template_version, system_prompt_version, tool_version), or whole tenant. Change `context_version` when your documents change: it is part of every key.

**Known weakness**: the signature is a heuristic. "thirty" vs "30", antonyms ("allowed"/"forbidden"), and meaning changes without a number, negation or identifier are not caught. Raise the threshold or lower TTL for high-stakes content.
