"""Deterministic cache keys. The key is a SHA-256 of a canonical JSON document that contains EVERYTHING that can change the answer:
tenant, model, system prompt version, prompt template version, context (document) version, tool configuration, generation
parameters and the normalized query. Two requests share a key only if all of those are identical. Canonical JSON (sorted keys, fixed
separators, no ASCII escaping ambiguity) means field order and spacing can never produce different keys for the same request."""

import json

from app.cache.models import QueryRequest
from app.cache.normalize import normalize_query, sha256_hex

KEY_SCHEMA = 1


def namespace(tenant_id: str) -> str:
    """Per-tenant namespace used in every storage key. Hashing keeps tenant ids out of key names and makes delimiter tricks
    ("a:b" vs "a" + ":b") impossible."""
    return sha256_hex("tenant:" + tenant_id)[:24]


def params_hash(temperature: float, max_output_tokens: int) -> str:
    return sha256_hex(json.dumps({"t": round(temperature, 4), "m": max_output_tokens}, sort_keys=True, separators=(",", ":")))[:32]


def canonical_json(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def exact_key(tenant_id: str, req: QueryRequest, *, model: str, system_prompt_version: str, max_output_tokens: int) -> str:
    doc = {
        "v": KEY_SCHEMA,
        "tenant": tenant_id,
        "model": model,
        "system_prompt_version": system_prompt_version,
        "prompt_template_version": req.prompt_template_version,
        "context_version": req.context_version,
        "tool_version": req.tool_version,
        "params": params_hash(req.temperature, max_output_tokens),
        "query": normalize_query(req.query),
    }
    return sha256_hex(canonical_json(doc))
