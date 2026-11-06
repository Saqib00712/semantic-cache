# ruff: noqa: E501
"""Invalidation and lifecycle. Every bulk invalidation BUMPS THE TENANT EPOCH FIRST: a generation that started before the
invalidation and finishes after it carries the old epoch, and the store rejects it (otherwise a stale answer would be written back
right after the operator deleted it). Then entries are deleted. All operations are scoped to one tenant namespace."""

from app.cache.keys import namespace
from app.observability import metrics
from app.storage.backend import Backend

FILTER_FIELDS = ("model", "context_version", "prompt_template_version", "system_prompt_version", "tool_version")


def invalidate_entry(be: Backend, tenant_id: str, entry_id: str) -> bool:
    ns = namespace(tenant_id)
    be.bump_epoch(ns)  # an in-flight identical generation must not re-create it
    ok = be.delete(ns, entry_id)
    metrics.inc("semcache_invalidations_total", scope="entry")
    return ok


def invalidate_matching(be: Backend, tenant_id: str, filters: dict[str, str]) -> int:
    if not filters or set(filters) - set(FILTER_FIELDS):
        raise ValueError("filters must be a non-empty subset of " + ", ".join(FILTER_FIELDS))
    ns = namespace(tenant_id)
    be.bump_epoch(ns)
    n = be.delete_matching(ns, filters)
    metrics.inc("semcache_invalidations_total", scope="filter")
    return n


def wipe_tenant(be: Backend, tenant_id: str) -> int:
    ns = namespace(tenant_id)
    be.bump_epoch(ns)
    n = be.delete_matching(ns, {})
    metrics.inc("semcache_invalidations_total", scope="tenant")
    return n


def list_metadata(be: Backend, tenant_id: str, limit: int = 100) -> list[dict]:
    """Operator view: metadata and hit counts only. Neither the cached answer nor the original query text is ever returned."""
    keep = ("entry_id", "model", "context_version", "prompt_template_version", "system_prompt_version", "tool_version", "created_at", "expires_at",
            "status", "finish_reason", "prompt_tokens", "completion_tokens", "hits", "embedding_model")  # fmt: skip
    return [{k: e.get(k) for k in keep} for e in be.list_entries(namespace(tenant_id), limit)]
