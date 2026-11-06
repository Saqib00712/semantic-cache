# Security

- **Auth**: `Authorization: Bearer sc_<prefix>_<secret>`. Only a SHA-256 of the key is stored; constant-time compare; revoked/suspended tenants are refused. Unauthenticated callers get 401 before the body is read.
- **Tenant isolation**: the tenant comes only from the key (a `tenant_id` in a body is rejected). Every storage key embeds `sha256("tenant:"+id)[:24]`; every backend call is namespace-scoped; a tenant cannot read, search, list or delete another's entries (a foreign entry id returns the same 404 as a missing one). Tested for exact, semantic and management paths.
- **Admin**: separate token; if unset, admin routes answer 404 exactly like unknown paths.
- **No secrets in output**: logs use an allow-list of fields plus regex/value redaction; metrics labels are fixed vocabularies; error bodies never echo input; operator entry listings return metadata only, never answers or queries.
- **Inputs**: strict Pydantic models (unknown fields rejected), control characters rejected, request body size limited (also for chunked bodies), nested-JSON `RecursionError` handled.
- **Prompts are sent to OpenAI (or your gateway)** on a miss, and embeddings of normalized prompts are computed by OpenAI. Cached answers are stored in Redis in plaintext. Use Redis auth/TLS and disk encryption; this project does not encrypt values at rest.
- **Not done**: key rotation UI, per-key scopes, audit log of admin actions, TLS termination (put a proxy in front), protection against a malicious tenant crafting many distinct queries beyond rate/quota/budget.
