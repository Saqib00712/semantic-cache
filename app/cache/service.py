# ruff: noqa: E501
"""The cache pipeline. One function, `handle_query`, runs: limits -> bypass check -> exact lookup -> embedding -> vector search ->
eligibility -> hit, or single-flight generation -> validate -> store -> cost ledger. Every request ends with exactly one ledger event
and one metrics update. No prompt, answer or key is logged."""

import json
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np

from app.api.errors import ApiError
from app.cache.keys import exact_key, namespace, params_hash
from app.cache.models import EntryMeta, QueryRequest, QueryResponse, Usage
from app.cache.normalize import critical_signature, normalize_query, sha256_hex
from app.config import Settings
from app.eligibility import engine
from app.embeddings.service import EmbeddingError, embed
from app.generation.service import GenerationError, cost_of, embedding_cost, generate
from app.observability import metrics
from app.observability.logging import get_logger
from app.observability.tracing import span
from app.storage import ledger
from app.storage.backend import Backend, BackendError
from app.storage.database import db

log = get_logger()


@dataclass
class Runtime:
    settings: Settings
    backend: Backend
    embed_client: object = None
    gen_client: object = None
    sleep: object = time.sleep
    clock: object = time.time
    local_windows: dict = field(default_factory=dict)  # in-process rate-limit fallback if the shared store is down


@dataclass
class _Outcome:
    answer: str
    status: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    score: float | None = None
    gen_cost: Decimal = Decimal(0)
    avoided: Decimal = Decimal(0)
    reported_prompt: int = 0
    reported_completion: int = 0


def resolve_model(settings: Settings, req: QueryRequest) -> str:
    model = req.model or settings.generation_model
    if model not in settings.models:
        raise ApiError(422, "model_not_allowed", "that model is not allowed")
    return model


def check_limits(rt: Runtime, tenant: dict, ns: str) -> None:
    """Rate limit (shared store, with an in-process fallback) and the SQLite-enforced daily query quota and spend budget."""
    window = int(rt.clock() // 60)
    try:
        n = rt.backend.incr_window(ns, f"m{window}", 120)
    except BackendError:
        metrics.inc("semcache_degraded_total", cause="rate_limit_local")
        n = rt.local_windows[(ns, window)] = rt.local_windows.get((ns, window), 0) + 1
        for k in [k for k in rt.local_windows if k[1] < window]:
            del rt.local_windows[k]
    if n > tenant["rate_per_minute"]:
        raise ApiError(429, "rate_limited", "too many requests", {"Retry-After": str(60 - int(rt.clock() % 60))})
    with db(rt.settings.database_path) as conn:
        if ledger.queries_today(conn, tenant["id"]) >= tenant["daily_queries"]:
            raise ApiError(429, "quota_exceeded", "the daily query quota is used up")
        if ledger.spend_today(conn, tenant["id"]) >= ledger.from_pico(tenant["daily_budget_pico"]):
            raise ApiError(429, "budget_exceeded", "the daily spend budget is used up")


def _cacheable(settings: Settings, text: str, finish_reason: str) -> str | None:
    """None if the answer may be stored, else the reason it may not."""
    if finish_reason != "stop":
        return f"finish_reason_{finish_reason}"  # truncated or filtered answers are never reused
    if any(p.search(text) for p in settings.response_patterns):
        return "blocked_pattern"
    return None


def _hit_outcome(meta: EntryMeta, answer: str, status: str, score: float | None) -> _Outcome:
    return _Outcome(
        answer, status, meta.model, meta.prompt_tokens, meta.completion_tokens, score, avoided=ledger.from_pico(meta.gen_cost_pico)
    )


def _lookup(rt: Runtime, ctx: dict) -> tuple[_Outcome | None, np.ndarray | None, int, bool]:
    """(hit, vector, embedding tokens billed, embedding_failed). Raises BackendError."""
    s, be, ns, tenant_id = rt.settings, rt.backend, ctx["ns"], ctx["tenant_id"]
    now = rt.clock()
    with span("cache.exact_lookup"):
        got = be.get_exact(ns, ctx["ek"])
        if got is not None:
            v = engine.check_exact(got.meta, tenant_id=tenant_id, now=now)
            if v.meta is not None and v.meta.exact_key == ctx["ek"]:
                be.record_hit(ns, v.meta.entry_id)
                return _hit_outcome(v.meta, got.answer, "exact_hit", 1.0), None, 0, False
            metrics.inc("semcache_rejections_total", reason="exact_" + v.reason)
            if v.reason == "malformed_metadata":
                _drop_corrupt(be, ns, got.meta)
    try:
        with span("cache.embed"):
            vec, tokens, cached = embed(s, be, rt.embed_client, ctx["norm"], sleep=rt.sleep)
    except EmbeddingError:
        metrics.inc("semcache_degraded_total", cause="embedding_failed")
        return None, None, 0, True
    billed = 0 if cached else tokens
    with span("cache.vector_search"):
        cands = be.search(ns, vec, s.top_k, s.embedding_model)
    for entry_id, score in cands:
        e = be.get(ns, entry_id)
        if e is None:
            continue
        v = engine.check_semantic(
            e.meta,
            score,
            req=ctx["req"],
            tenant_id=tenant_id,
            signature=ctx["sig"],
            model=ctx["model"],
            max_tokens=ctx["max_tokens"],
            settings=s,
            now=now,
        )
        if v.meta is not None:
            be.record_hit(ns, entry_id)
            return _hit_outcome(v.meta, e.answer, "semantic_hit", round(score, 4)), vec, billed, False
        metrics.inc("semcache_rejections_total", reason=v.reason)
        if v.reason == "malformed_metadata":
            _drop_corrupt(be, ns, e.meta, entry_id)
    return None, vec, billed, False


def _drop_corrupt(be: Backend, ns: str, meta: dict, entry_id: str | None = None) -> None:
    eid = entry_id or meta.get("entry_id")
    if isinstance(eid, str) and len(eid) <= 64:
        try:
            be.delete(ns, eid)
            metrics.inc("semcache_errors_total", component="corrupt_entry_removed")
        except BackendError:
            pass


def _generate_and_store(rt: Runtime, ctx: dict, vec: np.ndarray | None, store: bool) -> _Outcome:
    s, req = rt.settings, ctx["req"]
    epoch = rt.backend.get_epoch(ctx["ns"]) if store else 0
    with span("openai.generate"):
        g = generate(
            s,
            rt.gen_client,
            model=ctx["model"],
            query=req.query,
            max_tokens=ctx["max_tokens"],
            temperature=req.temperature,
            request_id=ctx["request_id"],
            sleep=rt.sleep,
        )
    cost = cost_of(s, g.prompt_tokens, g.completion_tokens)
    metrics.inc("semcache_openai_calls_total", kind="generation", result="ok")
    why_not = _cacheable(s, g.answer, g.finish_reason)
    if store and why_not is None:
        entry_id, now = uuid.uuid4().hex, rt.clock()
        meta = EntryMeta(
            entry_id=entry_id, tenant_id=ctx["tenant_id"], exact_key=ctx["ek"], query_hash=sha256_hex(ctx["norm"]), signature=ctx["sig"], model=ctx["model"],
            system_prompt_version=s.system_prompt_version, prompt_template_version=req.prompt_template_version, context_version=req.context_version,
            tool_version=req.tool_version, params_hash=params_hash(req.temperature, ctx["max_tokens"]),
            embedding_model=s.embedding_model if vec is not None else None, embedding_dim=s.embedding_dim if vec is not None else None,
            personalized=False, scope="tenant", status="active", created_at=now, expires_at=now + s.ttl_seconds, ttl_seconds=s.ttl_seconds,
            prompt_tokens=g.prompt_tokens, completion_tokens=g.completion_tokens, finish_reason=g.finish_reason, gen_cost_pico=ledger.to_pico(cost),
        )  # fmt: skip
        with span("cache.store"):
            ok = rt.backend.put(
                ctx["ns"], entry_id, ctx["ek"], meta.model_dump(mode="json"), g.answer, vec, s.ttl_seconds, ctx["max_entries"], epoch
            )
        if not ok:
            metrics.inc("semcache_rejections_total", reason="store_skipped_invalidated_during_generation")
    elif store:
        metrics.inc("semcache_rejections_total", reason="not_stored_" + (why_not or ""))
    return _Outcome(g.answer, "miss", ctx["model"], g.prompt_tokens, g.completion_tokens, None, gen_cost=cost)


def _single_flight(rt: Runtime, ctx: dict, vec: np.ndarray | None) -> _Outcome:
    """Only one identical request generates; the others wait for its result instead of paying for the same generation."""
    s, be, ns, ek = rt.settings, rt.backend, ctx["ns"], ctx["ek"]
    token, started = uuid.uuid4().hex, rt.clock()
    deadline = started + s.lock_wait_seconds
    while True:
        if be.lock_acquire(ns, ek, token, s.lock_ttl_seconds):
            try:
                got = be.get_exact(ns, ek)  # someone may have finished between our lookup and our lock
                if got is not None:
                    v = engine.check_exact(got.meta, tenant_id=ctx["tenant_id"], now=rt.clock())
                    if v.meta is not None:
                        return _hit_outcome(v.meta, got.answer, "exact_hit", 1.0)
                out = _generate_and_store(rt, ctx, vec, store=True)
                be.flight_put(
                    ns,
                    ek,
                    json.dumps(
                        {
                            "t": rt.clock(),
                            "a": out.answer,
                            "m": out.model,
                            "p": out.prompt_tokens,
                            "c": out.completion_tokens,
                            "cost": str(out.gen_cost),
                        }
                    ),
                    10,
                )
                return out
            except GenerationError as e:
                try:
                    be.flight_put(ns, ek, json.dumps({"t": rt.clock(), "err": e.code}), 10)  # waiters stop waiting and retry on their own
                except BackendError:
                    pass
                raise
            finally:
                try:
                    be.lock_release(ns, ek, token)  # compare-and-delete: never releases a lock someone else now owns
                except BackendError:
                    pass
        got = be.get_exact(ns, ek)
        if got is not None:
            v = engine.check_exact(got.meta, tenant_id=ctx["tenant_id"], now=rt.clock())
            if v.meta is not None:
                o = _hit_outcome(v.meta, got.answer, "coalesced", None)
                return o
        raw = be.flight_get(ns, ek)
        if raw:
            try:
                f = json.loads(raw)
            except ValueError:
                f = {}
            if f.get("t", 0) >= started and "a" in f:
                return _Outcome(
                    f["a"], "coalesced", f.get("m", ctx["model"]), f.get("p", 0), f.get("c", 0), None, avoided=Decimal(f.get("cost", "0"))
                )
        if rt.clock() >= deadline:
            metrics.inc("semcache_degraded_total", cause="single_flight_wait_timeout")
            return _generate_and_store(rt, ctx, vec, store=True)
        rt.sleep(s.lock_poll_seconds)


def handle_query(rt: Runtime, tenant: dict, req: QueryRequest, request_id: str) -> QueryResponse:
    t0 = time.perf_counter()
    s = rt.settings
    model = resolve_model(s, req)
    max_tokens = req.max_output_tokens or s.max_output_tokens
    ns = namespace(tenant["id"])
    check_limits(rt, tenant, ns)
    norm = normalize_query(req.query)
    ctx = dict(
        ns=ns, tenant_id=tenant["id"], req=req, norm=norm, sig=critical_signature(norm), model=model, max_tokens=max_tokens, request_id=request_id,
        max_entries=tenant["max_entries"],
        ek=exact_key(tenant["id"], req, model=model, system_prompt_version=s.system_prompt_version, max_output_tokens=max_tokens),
    )  # fmt: skip
    outcome: _Outcome | None = None
    emb_tokens, lookup_ms, degraded = 0, 0, False
    try:
        bypass = engine.request_bypass_reason(req, s)
        if bypass:
            metrics.inc("semcache_bypass_total", reason=bypass.split(":")[0])
            outcome = _generate_and_store(rt, ctx, None, store=False)
            outcome.status = "ineligible"
        else:
            tl = time.perf_counter()
            try:
                outcome, vec, emb_tokens, _ = _lookup(rt, ctx)
                lookup_ms = int((time.perf_counter() - tl) * 1000)
                if outcome is None:
                    outcome = _single_flight(rt, ctx, vec)
            except BackendError:
                metrics.inc("semcache_errors_total", component="cache_store")
                if s.on_cache_failure == "fail":
                    raise ApiError(503, "cache_unavailable", "the cache store is unavailable") from None
                degraded = True
                metrics.inc("semcache_degraded_total", cause="cache_store_bypass")
                outcome = _generate_and_store(rt, ctx, None, store=False)
    except GenerationError as e:
        metrics.inc("semcache_openai_calls_total", kind="generation", result=e.code)
        _record(rt, tenant, "miss", t0, lookup_ms, emb_tokens, None, error=True)
        raise ApiError(e.status, e.code, e.message) from None
    latency_ms = int((time.perf_counter() - t0) * 1000)
    _record(rt, tenant, outcome.status, t0, lookup_ms, emb_tokens, outcome)
    metrics.inc("semcache_requests_total", cache_status=outcome.status)
    metrics.observe("semcache_request_seconds", latency_ms / 1000, cache_status=outcome.status)
    log.info(
        "query",
        extra={
            "event": "query",
            "cache_status": outcome.status,
            "decision": outcome.status,
            "duration_ms": latency_ms,
            "tenant_id": tenant["id"],
            "component": "degraded" if degraded else "ok",
        },
    )
    return QueryResponse(
        answer=outcome.answer, cache_status=outcome.status, similarity_score=outcome.score, model=outcome.model,
        usage=Usage(prompt_tokens=outcome.prompt_tokens, completion_tokens=outcome.completion_tokens), request_id=request_id,
    )  # fmt: skip


def _record(
    rt: Runtime, tenant: dict, status: str, t0: float, lookup_ms: int, emb_tokens: int, o: _Outcome | None, error: bool = False
) -> None:
    s = rt.settings
    gen = o.gen_cost if o else Decimal(0)
    with db(s.database_path) as conn:
        ledger.record_event(
            conn, tenant_id=tenant["id"], cache_status=status, latency_ms=int((time.perf_counter() - t0) * 1000), lookup_ms=lookup_ms,
            gen_prompt_tokens=o.prompt_tokens if o and o.status in ("miss", "ineligible") else 0,
            gen_completion_tokens=o.completion_tokens if o and o.status in ("miss", "ineligible") else 0,
            emb_tokens=emb_tokens, gen_cost=gen, emb_cost=embedding_cost(s, emb_tokens), infra_cost=s.infra_cost_per_request_usd,
            avoided_cost=o.avoided if o and not error else Decimal(0),
        )  # fmt: skip
