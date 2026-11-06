"""Settings from environment variables (prefix SEMCACHE_) and an optional .env file.

* Unknown SEMCACHE_* names are an error (a typo cannot silently fall back to a default).
* Secrets are SecretStr: never printed, never in error messages. Errors name the variable, never the value.
* Money values are Decimal: floats are never used for prices or costs.
"""

import os
import re
from decimal import Decimal
from typing import Literal
from urllib.parse import urlsplit

from dotenv import dotenv_values
from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PREFIX = "SEMCACHE_"
VERSION_RE = re.compile(r"^[A-Za-z0-9._:\-]{1,64}$")


class ConfigurationError(Exception):
    """Invalid configuration. The text is safe to print."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, extra="ignore", frozen=True, env_file_encoding="utf-8")

    environment: Literal["development", "test", "production"] = "production"
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    database_path: str = "data/semcache.db"  # tenants, API keys and the cost ledger (SQLite)

    # Cache storage. "redis" is the real backend; "memory" is single-process (tests, Windows demo, no Redis).
    backend: Literal["redis", "memory"] = "redis"
    redis_url: SecretStr | None = None
    redis_timeout_seconds: float = Field(default=2.0, gt=0, le=30)
    vector_mode: Literal["auto", "search", "scan"] = "auto"  # search = RediSearch module (redis-stack); scan = numpy over one tenant
    on_cache_failure: Literal["bypass", "fail"] = "bypass"  # when the cache store breaks: generate directly, or return 503

    # OpenAI (the only provider)
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None  # embeddings endpoint override (tests/proxy)
    generation_base_url: str | None = None  # where generation goes: unset = OpenAI directly, or an OpenAI-compatible LLM gateway
    generation_api_key: SecretStr | None = None  # key for the gateway; falls back to openai_api_key
    generation_model: str = Field(default="gpt-4o-mini", min_length=1, max_length=100)
    allowed_models: str = ""  # comma separated; empty = only generation_model
    embedding_model: str = Field(default="text-embedding-3-small", min_length=1, max_length=100)
    embedding_dim: int = Field(default=1536, ge=8, le=4096)
    embedding_mode: Literal["openai", "mock"] = "openai"  # mock = offline deterministic embedder (tests/demo only)
    system_prompt: str = Field(default="You are a helpful assistant. Answer concisely and accurately.", max_length=4000)
    system_prompt_version: str = "sp-1"
    openai_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    max_attempts: int = Field(default=3, ge=1, le=6)  # total tries for a transient OpenAI error
    retry_base_seconds: float = Field(default=0.5, ge=0, le=30)
    max_output_tokens: int = Field(default=512, ge=16, le=16384)

    # Prices in USD per 1M tokens: SET THESE TO YOUR MODELS' REAL PRICES; the defaults are placeholders.
    price_input_per_mtok: Decimal = Decimal("0.15")
    price_output_per_mtok: Decimal = Decimal("0.60")
    price_embedding_per_mtok: Decimal = Decimal("0.02")
    infra_cost_per_request_usd: Decimal = Decimal("0")  # cache infrastructure cost per request; 0 means "not measured"

    # Cache policy
    similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1, le=50)
    ttl_seconds: int = Field(default=86_400, ge=1, le=31_536_000)
    embedding_cache_ttl_seconds: int = Field(default=604_800, ge=1, le=31_536_000)
    max_cacheable_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    blocked_response_patterns: str = r"sk-[A-Za-z0-9_\-]{12,}"  # regexes separated by ";;": a matching answer is never cached
    max_query_chars: int = Field(default=4000, ge=1, le=100_000)
    max_request_bytes: int = Field(default=65_536, ge=1024, le=4_194_304)

    # Tenants: defaults (a tenant row may override)
    default_max_entries: int = Field(default=1000, ge=1, le=10_000_000)
    default_daily_queries: int = Field(default=10_000, ge=1, le=100_000_000)
    default_rate_per_minute: int = Field(default=120, ge=1, le=1_000_000)
    default_daily_budget_usd: Decimal = Decimal("5")  # max generation+embedding spend per tenant per UTC day
    admin_token: SecretStr | None = None

    # Single-flight (stampede prevention)
    lock_ttl_seconds: float = Field(default=30.0, gt=0, le=600)
    lock_wait_seconds: float = Field(default=20.0, gt=0, le=600)
    lock_poll_seconds: float = Field(default=0.05, gt=0, le=5)

    otel_exporter: Literal["none", "console", "otlp"] = "none"

    @field_validator("redis_url")
    @classmethod
    def _redis_url(cls, v: SecretStr | None) -> SecretStr | None:
        if v is None:
            return v
        try:
            parts = urlsplit(v.get_secret_value())
            ok = parts.scheme in ("redis", "rediss", "unix") and (bool(parts.hostname) or parts.scheme == "unix")
            _ = parts.port
        except ValueError:
            ok = False
        if not ok:
            raise ValueError("must look like redis://host:port/db")  # never echo the URL
        return v

    @field_validator("admin_token")
    @classmethod
    def _admin(cls, v: SecretStr | None) -> SecretStr | None:
        if v is not None and len(v.get_secret_value()) < 32:
            raise ValueError("must be at least 32 characters")
        return v

    @field_validator("system_prompt_version")
    @classmethod
    def _version(cls, v: str) -> str:
        if not VERSION_RE.match(v):
            raise ValueError("letters, digits and . _ : - only, 1 to 64 characters")
        return v

    @field_validator("blocked_response_patterns")
    @classmethod
    def _patterns(cls, v: str) -> str:
        for p in [x for x in v.split(";;") if x]:
            try:
                re.compile(p)
            except re.error:
                raise ValueError("contains an invalid regular expression") from None
        return v

    @model_validator(mode="after")
    def _checks(self) -> "Settings":
        if self.backend == "redis" and self.redis_url is None and self.environment == "production":
            raise ValueError("production with backend=redis requires SEMCACHE_REDIS_URL")
        return self

    @property
    def effective_redis_url(self) -> str:
        return self.redis_url.get_secret_value() if self.redis_url is not None else "redis://127.0.0.1:6379/0"

    @property
    def models(self) -> tuple[str, ...]:
        extra = tuple(m.strip() for m in self.allowed_models.split(",") if m.strip())
        return tuple(dict.fromkeys((self.generation_model, *extra)))

    @property
    def response_patterns(self) -> list[re.Pattern]:
        return [re.compile(p) for p in self.blocked_response_patterns.split(";;") if p]

    def secret_values(self) -> list[str]:
        values = []
        for s in (self.redis_url, self.openai_api_key, self.generation_api_key, self.admin_token):
            if s is not None:
                values.append(s.get_secret_value())
        return values

    def safe_summary(self) -> dict[str, object]:
        return {
            "environment": self.environment,
            "backend": self.backend,
            "vector_mode": self.vector_mode,
            "generation_model": self.generation_model,
            "embedding_model": self.embedding_model,
            "embedding_mode": self.embedding_mode,
            "similarity_threshold": self.similarity_threshold,
            "openai_key_configured": self.openai_api_key is not None,
            "admin_enabled": self.admin_token is not None,
            "generation_via_gateway": self.generation_base_url is not None,
        }


def _unknown_names(environ, env_file: str | None) -> list[str]:
    known = {ENV_PREFIX + name.upper() for name in Settings.model_fields}
    names = {k.upper() for k in environ if k.upper().startswith(ENV_PREFIX)}
    if env_file and os.path.exists(env_file):
        names |= {k.upper() for k in dotenv_values(env_file) if k.upper().startswith(ENV_PREFIX)}
    return sorted(names - known)


def load_settings(env_file: str | None = ".env") -> Settings:
    unknown = _unknown_names(os.environ, env_file)
    if unknown:
        raise ConfigurationError("unknown setting(s): " + ", ".join(unknown))
    try:
        return Settings(_env_file=env_file if env_file and os.path.exists(env_file) else None)  # type: ignore[call-arg]
    except ValidationError as e:
        problems = []
        for err in e.errors():
            loc = err.get("loc") or ()
            name = ENV_PREFIX + str(loc[0]).upper() if loc else "settings"
            problems.append(f"{name}: {str(err.get('msg', 'invalid')).removeprefix('Value error, ')}")
        raise ConfigurationError("invalid configuration: " + "; ".join(problems)) from None
