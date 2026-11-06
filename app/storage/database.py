"""SQLite for what must survive a restart and be queryable: tenants, API key hashes and the cost ledger. NOT the cache itself
(entries live in Redis or the memory backend). One short-lived connection per use; WAL mode lets readers and one writer coexist."""

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE tenants (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','suspended')),
    max_entries INTEGER, daily_queries INTEGER, rate_per_minute INTEGER, daily_budget_pico INTEGER, created_at TEXT NOT NULL);
CREATE TABLE api_keys (
    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL REFERENCES tenants(id), key_prefix TEXT NOT NULL UNIQUE,
    key_hash TEXT NOT NULL, created_at TEXT NOT NULL, revoked_at TEXT);
CREATE TABLE usage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, day TEXT NOT NULL, tenant_id TEXT NOT NULL, cache_status TEXT NOT NULL,
    latency_ms INTEGER NOT NULL, lookup_ms INTEGER NOT NULL DEFAULT 0, gen_prompt_tokens INTEGER NOT NULL DEFAULT 0,
    gen_completion_tokens INTEGER NOT NULL DEFAULT 0, emb_tokens INTEGER NOT NULL DEFAULT 0, gen_cost_pico INTEGER NOT NULL DEFAULT 0,
    emb_cost_pico INTEGER NOT NULL DEFAULT 0, infra_cost_pico INTEGER NOT NULL DEFAULT 0, avoided_cost_pico INTEGER NOT NULL DEFAULT 0);
CREATE INDEX usage_tenant_day ON usage_events (tenant_id, day);
CREATE INDEX usage_day ON usage_events (day);
"""


class DatabaseError(Exception):
    """Safe text only: no paths, no SQL."""


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=5.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db(path: str) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        yield conn
    finally:
        conn.close()


def init_database(path: str) -> None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with db(path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS schema_meta (id INTEGER PRIMARY KEY CHECK (id = 1), version INTEGER NOT NULL)")
            row = conn.execute("SELECT version FROM schema_meta WHERE id = 1").fetchone()
            if row is not None and row[0] > SCHEMA_VERSION:
                raise DatabaseError("the database was created by a newer version of this application")
            if row is None:
                conn.execute("BEGIN IMMEDIATE")  # two processes starting together: one creates, the other waits
                try:
                    if conn.execute("SELECT version FROM schema_meta WHERE id = 1").fetchone() is None:
                        for statement in _SCHEMA.split(";"):
                            if statement.strip():
                                conn.execute(statement)
                        conn.execute("INSERT INTO schema_meta (id, version) VALUES (1, ?)", (SCHEMA_VERSION,))
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
    except DatabaseError:
        raise
    except (sqlite3.Error, OSError):
        raise DatabaseError("the database could not be opened or initialised") from None


def check_database(path: str) -> bool:
    try:
        with db(path) as conn:
            row = conn.execute("SELECT version FROM schema_meta WHERE id = 1").fetchone()
            return row is not None and row[0] == SCHEMA_VERSION
    except Exception:  # noqa: BLE001 - readiness answers yes or no
        return False
