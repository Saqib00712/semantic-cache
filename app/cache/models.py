"""Request, response and cache-entry models. Every model forbids unknown fields, so a misspelt or smuggled field (for example a
`tenant_id` in the body) is rejected rather than ignored."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.cache.normalize import has_control_chars

VERSION_PATTERN = r"^[A-Za-z0-9._:\-]{1,64}$"
_VERSION = re.compile(VERSION_PATTERN)
CacheStatus = Literal["exact_hit", "semantic_hit", "miss", "ineligible", "coalesced"]
Decision = Literal["exact_hit", "semantic_hit", "miss", "rejected_incompatible", "expired", "ineligible"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QueryRequest(Strict):
    query: str = Field(min_length=1, max_length=100_000)  # the real limit (SEMCACHE_MAX_QUERY_CHARS) is checked in the route
    model: str | None = Field(default=None, max_length=100)
    context_version: str = Field(default="none", pattern=VERSION_PATTERN)  # version of the documents/knowledge base the answer depends on
    prompt_template_version: str = Field(default="v1", pattern=VERSION_PATTERN)  # the calling app's prompt template version
    tool_version: str | None = Field(default=None, pattern=VERSION_PATTERN)  # tool/function configuration, if the app uses any
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=16, le=16384)
    personalized: bool = False  # the caller says this request depends on one user's data: it will bypass the cache

    @field_validator("query")
    @classmethod
    def _query(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        if has_control_chars(v):
            raise ValueError("query contains control characters")
        return v


class Usage(Strict):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class QueryResponse(Strict):
    answer: str
    cache_status: CacheStatus
    similarity_score: float | None = None
    model: str
    usage: Usage
    request_id: str


class EntryMeta(Strict):
    """What is stored with a cached answer. Everything the eligibility decision needs is here, so the decision can be reproduced
    later from the entry alone (and a stored entry with a missing or malformed field is rejected, not trusted)."""

    schema_version: Literal[1] = 1
    entry_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    tenant_id: str = Field(min_length=1, max_length=64)
    exact_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    query_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    signature: dict
    model: str
    system_prompt_version: str
    prompt_template_version: str
    context_version: str
    tool_version: str | None
    params_hash: str
    embedding_model: str | None  # None: stored without a vector (embedding was unavailable), exact-match only
    embedding_dim: int | None
    personalized: bool
    scope: Literal["tenant"]
    status: Literal["active", "invalidated"]
    created_at: float
    expires_at: float
    ttl_seconds: int
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    gen_cost_pico: int  # what generating this answer cost: the "avoided" amount when it is served again (an estimate)

    @field_validator("model", "system_prompt_version", "prompt_template_version", "context_version", "tool_version")
    @classmethod
    def _version_like(cls, v):
        if v is not None and not _VERSION.match(v) and not re.match(r"^[A-Za-z0-9._:\-/]{1,100}$", v):
            raise ValueError("malformed")
        return v
