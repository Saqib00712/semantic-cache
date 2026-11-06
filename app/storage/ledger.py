"""The cost ledger. Money is Decimal in code and an INTEGER number of pico-dollars (1e-12 USD) in SQLite, so sums are exact
(floats and SQLite's REAL would drift). Every query writes one event; analytics are computed from the events."""

import sqlite3
from datetime import UTC, datetime
from decimal import Decimal

PICO = Decimal("1e12")
STATUSES = ("exact_hit", "semantic_hit", "miss", "ineligible", "coalesced")


def to_pico(usd: Decimal) -> int:
    return int((usd * PICO).to_integral_value())


def from_pico(pico: int) -> Decimal:
    return Decimal(pico) / PICO


def _s(d: Decimal) -> str:
    return format(d, "f")


def today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def record_event(conn: sqlite3.Connection, *, tenant_id: str, cache_status: str, latency_ms: int, lookup_ms: int = 0,
                 gen_prompt_tokens: int = 0, gen_completion_tokens: int = 0, emb_tokens: int = 0, gen_cost: Decimal = Decimal(0),
                 emb_cost: Decimal = Decimal(0), infra_cost: Decimal = Decimal(0), avoided_cost: Decimal = Decimal(0)) -> None:  # fmt: skip
    now = datetime.now(UTC)
    conn.execute(
        "INSERT INTO usage_events (ts, day, tenant_id, cache_status, latency_ms, lookup_ms, gen_prompt_tokens, gen_completion_tokens, "
        "emb_tokens, gen_cost_pico, emb_cost_pico, infra_cost_pico, avoided_cost_pico) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (now.isoformat(timespec="milliseconds").replace("+00:00", "Z"), now.strftime("%Y-%m-%d"), tenant_id, cache_status, latency_ms,
         lookup_ms, gen_prompt_tokens, gen_completion_tokens, emb_tokens, to_pico(gen_cost), to_pico(emb_cost), to_pico(infra_cost),
         to_pico(avoided_cost)),
    )  # fmt: skip


def spend_today(conn: sqlite3.Connection, tenant_id: str) -> Decimal:
    """Money actually spent today (generation + embeddings). Used to enforce the tenant's daily budget."""
    row = conn.execute("SELECT COALESCE(SUM(gen_cost_pico + emb_cost_pico), 0) FROM usage_events WHERE tenant_id = ? AND day = ?",
                       (tenant_id, today())).fetchone()  # fmt: skip
    return from_pico(row[0])


def queries_today(conn: sqlite3.Connection, tenant_id: str) -> int:
    return conn.execute("SELECT COUNT(*) FROM usage_events WHERE tenant_id = ? AND day = ?", (tenant_id, today())).fetchone()[0]


def summary(conn: sqlite3.Connection, tenant_id: str | None = None) -> dict:
    """Totals and savings. Definitions (see docs/cost_model.md):
    gross_avoided   = original generation cost of every hit (an estimate: what the model call would have cost again)
    embedding_cost  = embedding spend on ALL requests (a hit still paid for its embedding; a miss paid for it and got nothing back)
    infra_cost      = configured per-request infrastructure cost (0 unless configured)
    net_savings     = gross_avoided - embedding_cost - infra_cost"""
    where, args = ("WHERE tenant_id = ?", [tenant_id]) if tenant_id else ("", [])
    rows = conn.execute(
        "SELECT cache_status, COUNT(*), COALESCE(SUM(latency_ms),0), COALESCE(SUM(lookup_ms),0), COALESCE(SUM(gen_prompt_tokens),0), "
        "COALESCE(SUM(gen_completion_tokens),0), COALESCE(SUM(emb_tokens),0), COALESCE(SUM(gen_cost_pico),0), "
        f"COALESCE(SUM(emb_cost_pico),0), COALESCE(SUM(infra_cost_pico),0), COALESCE(SUM(avoided_cost_pico),0) FROM usage_events {where} "
        "GROUP BY cache_status", args).fetchall()  # fmt: skip
    by_status = {s: 0 for s in STATUSES}
    total = {"requests": 0, "latency_ms": 0, "lookup_ms": 0, "gen_in": 0, "gen_out": 0, "emb_tokens": 0}
    gen = emb = infra = avoided = 0
    lat_by = {}
    for st, n, lat, look, gi, go, et, gc, ec, ic, ac in rows:
        by_status[st] = by_status.get(st, 0) + n
        total["requests"] += n
        total["latency_ms"] += lat
        total["lookup_ms"] += look
        total["gen_in"] += gi
        total["gen_out"] += go
        total["emb_tokens"] += et
        gen, emb, infra, avoided = gen + gc, emb + ec, infra + ic, avoided + ac
        lat_by[st] = (lat, n)
    n = total["requests"]
    hits = by_status["exact_hit"] + by_status["semantic_hit"] + by_status["coalesced"]
    avg = lambda st: (lat_by[st][0] / lat_by[st][1]) if st in lat_by else None  # noqa: E731
    return {
        "requests": n,
        "by_status": by_status,
        "hit_rate": (hits / n) if n else None,
        "input_tokens": total["gen_in"],
        "output_tokens": total["gen_out"],
        "embedding_tokens": total["emb_tokens"],
        "generation_cost_usd": _s(from_pico(gen)),
        "embedding_cost_usd": _s(from_pico(emb)),
        "infra_cost_usd": _s(from_pico(infra)),
        "gross_generation_cost_avoided_usd": _s(from_pico(avoided)),
        "estimated_net_savings_usd": _s(from_pico(avoided - emb - infra)),
        "avg_latency_ms": (total["latency_ms"] / n) if n else None,
        "avg_latency_ms_by_status": {st: avg(st) for st in lat_by},
        "avg_lookup_ms": (total["lookup_ms"] / n) if n else None,
    }
