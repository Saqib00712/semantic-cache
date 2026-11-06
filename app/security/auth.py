"""Who is calling. Two separate kinds of credential:

* A TENANT API key (`Authorization: Bearer sc_...`) identifies a tenant. The tenant id is derived on the server from the key and is
  never read from the request body, query string or any header.
* The ADMIN token (SEMCACHE_ADMIN_TOKEN) is a different secret that only opens /admin and /metrics. A tenant key never works there.
All failures look the same to the caller (401), so a response does not say whether a key exists."""

import hashlib
import hmac
import re
import secrets
import sqlite3
import uuid
from datetime import UTC, datetime
from decimal import Decimal

from fastapi import Request
from starlette.concurrency import run_in_threadpool

from app.api.errors import ApiError
from app.storage.database import db
from app.storage.ledger import to_pico

KEY_RE = re.compile(r"^sc_([a-f0-9]{8})_([A-Za-z0-9_\-]{32,64})$")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def hash_key(key: str) -> str:
    """SHA-256 is right for a 256-bit random token (a slow password hash only helps low-entropy human passwords)."""
    return hashlib.sha256(key.encode()).hexdigest()


def create_tenant(conn: sqlite3.Connection, name: str, *, max_entries: int | None = None, daily_queries: int | None = None,
                  rate_per_minute: int | None = None, daily_budget_usd: Decimal | None = None) -> str:  # fmt: skip
    tenant_id = "t_" + uuid.uuid4().hex[:12]
    conn.execute(
        "INSERT INTO tenants (id, name, max_entries, daily_queries, rate_per_minute, daily_budget_pico, created_at) VALUES (?,?,?,?,?,?,?)",
        (tenant_id, name, max_entries, daily_queries, rate_per_minute, to_pico(daily_budget_usd) if daily_budget_usd is not None else None,
         _now()),
    )  # fmt: skip
    return tenant_id


def create_api_key(conn: sqlite3.Connection, tenant_id: str) -> str:
    """The full key. This is the only moment it exists in readable form."""
    prefix, secret = secrets.token_hex(4), secrets.token_urlsafe(32)
    key = f"sc_{prefix}_{secret}"
    conn.execute("INSERT INTO api_keys (id, tenant_id, key_prefix, key_hash, created_at) VALUES (?,?,?,?,?)",
                 ("k_" + uuid.uuid4().hex[:12], tenant_id, prefix, hash_key(key), _now()))  # fmt: skip
    return key


def revoke_keys(conn: sqlite3.Connection, tenant_id: str) -> int:
    return conn.execute("UPDATE api_keys SET revoked_at = ? WHERE tenant_id = ? AND revoked_at IS NULL", (_now(), tenant_id)).rowcount


def set_status(conn: sqlite3.Connection, tenant_id: str, status: str) -> bool:
    return conn.execute("UPDATE tenants SET status = ? WHERE id = ?", (status, tenant_id)).rowcount == 1


def list_tenants(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute("SELECT id, name, status, max_entries, daily_queries, rate_per_minute, daily_budget_pico, created_at FROM tenants")
    return [{d[0]: v for d, v in zip(cur.description, r, strict=True)} for r in cur.fetchall()]


def authenticate(conn: sqlite3.Connection, presented: str) -> dict | None:
    m = KEY_RE.match(presented)
    if not m:
        return None
    row = conn.execute(
        "SELECT k.key_hash, k.revoked_at, t.id, t.name, t.status, t.max_entries, t.daily_queries, t.rate_per_minute, t.daily_budget_pico "
        "FROM api_keys k JOIN tenants t ON t.id = k.tenant_id WHERE k.key_prefix = ?", (m.group(1),)).fetchone()  # fmt: skip
    if row is None or not hmac.compare_digest(row[0], hash_key(presented)) or row[1] is not None:
        return None
    return {"id": row[2], "name": row[3], "status": row[4], "max_entries": row[5], "daily_queries": row[6], "rate_per_minute": row[7],
            "daily_budget_pico": row[8]}  # fmt: skip


def _bearer(request: Request) -> str | None:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


def _unauthorized() -> ApiError:
    return ApiError(401, "unauthorized", "a valid API key is required", {"WWW-Authenticate": "Bearer"})


def _lookup(path: str, presented: str) -> dict | None:
    with db(path) as conn:
        return authenticate(conn, presented)


async def require_tenant(request: Request) -> dict:
    presented = _bearer(request)
    if presented is None:
        raise _unauthorized()
    tenant = await run_in_threadpool(_lookup, request.app.state.settings.database_path, presented)
    if tenant is None:
        raise _unauthorized()
    if tenant["status"] != "active":
        raise ApiError(403, "tenant_suspended", "this tenant is suspended")
    return tenant


async def require_admin(request: Request) -> None:
    configured = request.app.state.settings.admin_token
    if configured is None:  # admin disabled: indistinguishable from an unknown path
        raise ApiError(404, "not_found", "request could not be served")
    presented = _bearer(request)
    if presented is None:
        raise _unauthorized()
    a, b = hashlib.sha256(presented.encode()).digest(), hashlib.sha256(configured.get_secret_value().encode()).digest()
    if not hmac.compare_digest(a, b):
        raise _unauthorized()
