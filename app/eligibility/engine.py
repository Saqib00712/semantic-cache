"""Eligibility: a semantic match is only a CANDIDATE. This module decides, with an explicit reason, whether the candidate's answer may
be served for the new request. A semantic hit requires ALL of: a valid entry, same tenant, active, unexpired, same embedding model and
dimension, same answer-defining versions (model, system prompt, template, context, tools, params), not personalized, similarity at or
above the threshold, and a matching critical signature (numbers, negation, quoted text, identifiers). Anything else is a rejection with
a named reason. The reasons are what the logs/metrics/evaluation report."""

from dataclasses import dataclass

from pydantic import ValidationError

from app.cache.keys import params_hash
from app.cache.models import EntryMeta, QueryRequest
from app.config import Settings
from app.eligibility.personal import personal_data_reasons


@dataclass(frozen=True)
class Verdict:
    decision: str  # semantic_hit | rejected_incompatible | expired
    reason: str
    meta: EntryMeta | None = None


def request_bypass_reason(req: QueryRequest, settings: Settings) -> str | None:
    """Reasons a request must skip the cache entirely (no lookup, no store)."""
    if req.personalized:
        return "caller_marked_personalized"
    if req.temperature > settings.max_cacheable_temperature:
        return "temperature_above_cacheable_limit"
    if len(req.query) > settings.max_query_chars:
        return "query_too_long"
    signals = personal_data_reasons(req.query)
    return f"personal_data:{','.join(signals)}" if signals else None


def parse_meta(raw: dict) -> EntryMeta | None:
    try:
        return EntryMeta.model_validate(raw)
    except (ValidationError, TypeError):
        return None


def check_exact(raw: dict, *, tenant_id: str, now: float) -> Verdict:
    meta = parse_meta(raw)
    if meta is None:
        return Verdict("rejected_incompatible", "malformed_metadata")
    if meta.tenant_id != tenant_id:
        return Verdict("rejected_incompatible", "tenant_mismatch")
    if meta.status != "active":
        return Verdict("rejected_incompatible", "invalidated")
    if meta.expires_at <= now:
        return Verdict("expired", "ttl_elapsed")
    if meta.personalized:
        return Verdict("rejected_incompatible", "entry_personalized")
    return Verdict("semantic_hit", "ok", meta)  # the caller labels it exact_hit


def check_semantic(
    raw: dict,
    score: float,
    *,
    req: QueryRequest,
    tenant_id: str,
    signature: dict,
    model: str,
    max_tokens: int,
    settings: Settings,
    now: float,
) -> Verdict:
    v = check_exact(raw, tenant_id=tenant_id, now=now)
    if v.meta is None:
        return v
    m = v.meta
    if score < settings.similarity_threshold:
        return Verdict("rejected_incompatible", "below_threshold")
    if m.embedding_model != settings.embedding_model or m.embedding_dim != settings.embedding_dim:
        return Verdict("rejected_incompatible", "embedding_model_mismatch")
    if m.model != model:
        return Verdict("rejected_incompatible", "model_mismatch")
    if m.system_prompt_version != settings.system_prompt_version:
        return Verdict("rejected_incompatible", "system_prompt_version_mismatch")
    if m.prompt_template_version != req.prompt_template_version:
        return Verdict("rejected_incompatible", "template_version_mismatch")
    if m.context_version != req.context_version:
        return Verdict("rejected_incompatible", "context_version_mismatch")
    if m.tool_version != req.tool_version:
        return Verdict("rejected_incompatible", "tool_version_mismatch")
    if m.params_hash != params_hash(req.temperature, max_tokens):
        return Verdict("rejected_incompatible", "params_mismatch")
    s = m.signature
    for field, reason in (
        ("numbers", "number_mismatch"),
        ("negated", "negation_mismatch"),
        ("quoted", "quoted_text_mismatch"),
        ("idents", "identifier_mismatch"),
    ):
        if s.get(field) != signature.get(field):
            return Verdict("rejected_incompatible", reason)
    return Verdict("semantic_hit", "ok", m)
