# ruff: noqa: E501
"""Application factory and routes. Auth runs before the body is read, so an unauthenticated caller always gets 401 whatever the body."""

import time
from decimal import Decimal

from fastapi import Depends, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, ValidationError
from pydantic import ConfigDict as _CD

from app.api.body_limit import BodyLimitMiddleware
from app.api.errors import ApiError, error_response, install_error_handlers, internal_error_response, safe_details
from app.cache import lifecycle, service
from app.cache.keys import namespace
from app.cache.models import QueryRequest
from app.config import Settings, load_settings
from app.embeddings.service import make_openai_client
from app.generation.service import make_generation_client
from app.observability import metrics
from app.observability.context import accept_request_id, new_request_id, request_id_var
from app.observability.logging import configure_logging, get_logger, register_secret
from app.observability.tracing import configure_tracing
from app.security import auth
from app.storage import ledger
from app.storage.backend import Backend, BackendError
from app.storage.database import check_database, db, init_database
from app.storage.memory import MemoryBackend

VERSION = "1.0.0"
log = get_logger()


class _Body(BaseModel):
    model_config = _CD(extra="forbid")


class TenantCreate(_Body):
    name: str = Field(min_length=1, max_length=100)
    max_entries: int | None = Field(default=None, ge=1, le=10_000_000)
    daily_queries: int | None = Field(default=None, ge=1, le=100_000_000)
    rate_per_minute: int | None = Field(default=None, ge=1, le=1_000_000)
    daily_budget_usd: Decimal | None = Field(default=None, ge=0, le=1_000_000)


async def _read(request: Request, model):
    try:
        raw = await request.json()
    except (ValueError, RecursionError, UnicodeDecodeError):
        raise ApiError(400, "invalid_json", "the request body is not valid JSON") from None
    try:
        return model.model_validate(raw)
    except ValidationError as e:
        raise ApiError(422, "invalid_request", "the request is not valid") from _Detail(safe_details(e.errors()))
    except RecursionError:
        raise ApiError(400, "invalid_json", "the request body is not valid JSON") from None


class _Detail(Exception):
    def __init__(self, details):
        super().__init__("invalid")
        self.details = details


def _tenant_with_defaults(s: Settings, t: dict) -> dict:
    return {
        **t,
        "max_entries": t["max_entries"] or s.default_max_entries,
        "daily_queries": t["daily_queries"] or s.default_daily_queries,
        "rate_per_minute": t["rate_per_minute"] or s.default_rate_per_minute,
        "daily_budget_pico": t["daily_budget_pico"] if t["daily_budget_pico"] is not None else ledger.to_pico(s.default_daily_budget_usd),
    }


def create_app(
    settings: Settings | None = None, *, backend: Backend | None = None, embed_client=None, gen_client=None, sleep=time.sleep
) -> FastAPI:
    s = settings or load_settings()
    configure_logging(s.log_level)
    for v in s.secret_values():
        register_secret(v)
    configure_tracing(s.otel_exporter)
    init_database(s.database_path)
    if backend is None:
        if s.backend == "memory":
            backend = MemoryBackend()
        else:
            import redis

            from app.storage.redis_backend import RedisBackend

            client = redis.Redis.from_url(
                s.effective_redis_url, socket_timeout=s.redis_timeout_seconds, socket_connect_timeout=s.redis_timeout_seconds
            )
            backend = RedisBackend(client, s.vector_mode, s.embedding_dim)
    if embed_client is None and s.embedding_mode == "openai" and s.openai_api_key:
        embed_client = make_openai_client(s)
    if gen_client is None and (s.openai_api_key or s.generation_api_key):
        gen_client = make_generation_client(s)
    rt = service.Runtime(settings=s, backend=backend, embed_client=embed_client, gen_client=gen_client, sleep=sleep)

    app = FastAPI(title="semantic-cache", version=VERSION, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings, app.state.runtime = s, rt
    install_error_handlers(app)
    app.add_middleware(BodyLimitMiddleware, max_bytes=s.max_request_bytes)

    @app.middleware("http")
    async def _request_id(request: Request, call_next):
        rid = accept_request_id(request.headers.get("x-request-id")) or new_request_id()
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled", extra={"event": "unhandled", "path": request.url.path})
            response = internal_error_response()
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(_Detail)
    async def _detail(_: Request, exc: _Detail):
        return error_response(422, "invalid_request", "the request is not valid", details=exc.details)

    # ---- public --------------------------------------------------------------------------------------------------------------
    @app.get("/health")
    async def health():
        return {"status": "ok", "version": VERSION}

    @app.get("/ready")
    async def ready():
        checks = {"database": "ok", "cache_store": "ok"}
        try:
            await run_in_threadpool(check_database, s.database_path)
        except Exception:
            checks["database"] = "unavailable"
        try:
            await run_in_threadpool(rt.backend.ping)
        except BackendError:
            checks["cache_store"] = "unavailable"
        ok = all(v == "ok" for v in checks.values())
        return JSONResponse(
            {"status": "ready" if ok else "not_ready", "checks": checks, "vector_mode": rt.backend.vector_mode},
            status_code=200 if ok else 503,
        )

    @app.get("/metrics")
    async def prometheus(request: Request):
        if s.admin_token is not None:
            await auth.require_admin(request)
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    # ---- tenant API ----------------------------------------------------------------------------------------------------------
    @app.post("/v1/cache/query")
    async def query(request: Request, tenant: dict = Depends(auth.require_tenant)):
        req = await _read(request, QueryRequest)
        t = _tenant_with_defaults(s, tenant)
        rid = request_id_var.get()
        res = await run_in_threadpool(service.handle_query, rt, t, req, rid)
        return res

    @app.get("/v1/cache/entries")
    async def entries(limit: int = 100, tenant: dict = Depends(auth.require_tenant)):
        items = await _cache_call(lifecycle.list_metadata, rt.backend, tenant["id"], max(1, min(limit, 500)))
        return {"entries": items, "count": len(items)}

    @app.delete("/v1/cache/entries/{entry_id}")
    async def delete_entry(entry_id: str, tenant: dict = Depends(auth.require_tenant)):
        if len(entry_id) != 32 or not all(c in "0123456789abcdef" for c in entry_id):
            raise ApiError(404, "not_found", "request could not be served")
        ok = await _cache_call(lifecycle.invalidate_entry, rt.backend, tenant["id"], entry_id)
        if not ok:
            raise ApiError(404, "not_found", "request could not be served")  # same answer for "missing" and "someone else's"
        return {"deleted": 1}

    @app.delete("/v1/cache")
    async def invalidate(request: Request, tenant: dict = Depends(auth.require_tenant)):
        q = dict(request.query_params)
        wipe = q.pop("all", None) == "true"
        if wipe and q:
            raise ApiError(422, "invalid_request", "use either all=true or filters, not both")
        if not wipe and not q:
            raise ApiError(422, "invalid_request", "give at least one filter, or all=true to remove every entry")
        try:
            n = await _cache_call(
                lifecycle.wipe_tenant if wipe else lifecycle.invalidate_matching, rt.backend, tenant["id"], *([] if wipe else [q])
            )
        except ValueError:
            raise ApiError(422, "invalid_request", "unknown filter; allowed: " + ", ".join(lifecycle.FILTER_FIELDS)) from None
        return {"deleted": n}

    @app.get("/v1/cache/stats")
    async def stats(tenant: dict = Depends(auth.require_tenant)):
        def work():
            with db(s.database_path) as conn:
                return ledger.summary(conn, tenant["id"])

        out = await run_in_threadpool(work)
        out["cached_entries"] = await _cache_call(rt.backend.count, namespace(tenant["id"]))
        return out

    # ---- admin ---------------------------------------------------------------------------------------------------------------
    @app.post("/admin/tenants", dependencies=[Depends(auth.require_admin)])
    async def admin_create(request: Request):
        body = await _read(request, TenantCreate)

        def work():
            with db(s.database_path) as conn:
                tid = auth.create_tenant(conn, body.name, max_entries=body.max_entries, daily_queries=body.daily_queries,
                                         rate_per_minute=body.rate_per_minute, daily_budget_usd=body.daily_budget_usd)  # fmt: skip
                return tid, auth.create_api_key(conn, tid)

        tid, key = await run_in_threadpool(work)
        return JSONResponse({"tenant_id": tid, "api_key": key, "note": "store this key now; it cannot be shown again"}, status_code=201)

    @app.get("/admin/tenants", dependencies=[Depends(auth.require_admin)])
    async def admin_list():
        def work():
            with db(s.database_path) as conn:
                return auth.list_tenants(conn)

        return {"tenants": await run_in_threadpool(work)}

    def _tenant_op(fn):
        def work(tid: str):
            with db(s.database_path) as conn:
                return fn(conn, tid)

        return work

    @app.post("/admin/tenants/{tenant_id}/keys", dependencies=[Depends(auth.require_admin)])
    async def admin_new_key(tenant_id: str):
        def make(conn, tid):
            if not conn.execute("SELECT 1 FROM tenants WHERE id = ?", (tid,)).fetchone():
                return None
            return auth.create_api_key(conn, tid)

        key = await run_in_threadpool(_tenant_op(make), tenant_id)
        if key is None:
            raise ApiError(404, "not_found", "request could not be served")
        return JSONResponse({"tenant_id": tenant_id, "api_key": key}, status_code=201)

    @app.post("/admin/tenants/{tenant_id}/revoke-keys", dependencies=[Depends(auth.require_admin)])
    async def admin_revoke(tenant_id: str):
        return {"revoked": await run_in_threadpool(_tenant_op(auth.revoke_keys), tenant_id)}

    @app.post("/admin/tenants/{tenant_id}/{action}", dependencies=[Depends(auth.require_admin)])
    async def admin_status(tenant_id: str, action: str):
        if action not in ("suspend", "activate"):
            raise ApiError(404, "not_found", "request could not be served")
        ok = await run_in_threadpool(
            _tenant_op(lambda c, t: auth.set_status(c, t, "suspended" if action == "suspend" else "active")), tenant_id
        )
        if not ok:
            raise ApiError(404, "not_found", "request could not be served")
        return {"tenant_id": tenant_id, "status": "suspended" if action == "suspend" else "active"}

    @app.delete("/admin/tenants/{tenant_id}/cache", dependencies=[Depends(auth.require_admin)])
    async def admin_wipe(tenant_id: str):
        return {"deleted": await _cache_call(lifecycle.wipe_tenant, rt.backend, tenant_id)}

    @app.get("/admin/stats", dependencies=[Depends(auth.require_admin)])
    async def admin_stats():
        def work():
            with db(s.database_path) as conn:
                return ledger.summary(conn, None)

        return await run_in_threadpool(work)

    @app.get("/ui", response_class=HTMLResponse)
    async def ui():
        return HTMLResponse(
            UI_HTML,
            headers={
                "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'"
            },
        )

    return app


async def _cache_call(fn, *args):
    try:
        return await run_in_threadpool(fn, *args)
    except BackendError:
        metrics.inc("semcache_errors_total", component="cache_store")
        raise ApiError(503, "cache_unavailable", "the cache store is unavailable") from None


UI_HTML = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Semantic Cache</title>
<style>body{font:15px system-ui;max-width:760px;margin:2rem auto;padding:0 1rem}textarea,input{width:100%;box-sizing:border-box;margin:.3rem 0;padding:.5rem}
button{padding:.5rem 1rem}pre{background:#f4f4f4;padding:.7rem;white-space:pre-wrap;word-break:break-word}.tag{font-weight:600}</style>
<h2>Semantic Cache</h2>
<p>Paste a tenant API key (kept only in this page, never stored), ask a question, then ask a reworded version.</p>
<input id=k type=password placeholder="API key (sc_...)" autocomplete=off>
<textarea id=q rows=3 placeholder="Question"></textarea>
<button id=go>Ask</button> <button id=st>Stats</button>
<p>Status: <span class=tag id=s>-</span> &nbsp; similarity: <span id=sim>-</span></p><pre id=o></pre>
<script>
const $=i=>document.getElementById(i);
async function call(m,p,b){const r=await fetch(p,{method:m,headers:{'Authorization':'Bearer '+$('k').value,'Content-Type':'application/json'},body:b?JSON.stringify(b):undefined});return [r.status,await r.json()]}
$('go').onclick=async()=>{const [c,j]=await call('POST','/v1/cache/query',{query:$('q').value});
 if(c!=200){$('s').textContent='error '+c;$('o').textContent=j.error?j.error.message:'';return}
 $('s').textContent=j.cache_status;$('sim').textContent=j.similarity_score??'-';$('o').textContent=j.answer};
$('st').onclick=async()=>{const [c,j]=await call('GET','/v1/cache/stats');$('o').textContent=JSON.stringify(j,null,2)};
</script>"""
