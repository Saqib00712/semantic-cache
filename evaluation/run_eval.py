"""Offline evaluation of the safety pipeline and the cost model.

IMPORTANT: this uses the deterministic MOCK embedder and a fake generator. It tests whether the eligibility engine keeps a wrong answer
from being served; it does NOT measure how OpenAI embeddings behave, and its hit rates are not production hit rates. Run with
--live (needs OPENAI_API_KEY / SEMCACHE_OPENAI_API_KEY) to use real embeddings for the vector scores (generation stays fake, so
nothing is spent on completions).

Usage: python -m evaluation.run_eval [--threshold 0.8] [--live] [--out evaluation/report.md]"""

import argparse
import json
import os
import sys
import tempfile
import types
from decimal import Decimal
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.embeddings.service import embed
from app.main import create_app
from app.security import auth
from app.similarity.vector import cosine
from app.storage.database import db
from app.storage.memory import MemoryBackend

HERE = Path(__file__).parent
DIM = 256
KINDS_MUST_NOT_HIT = {
    "number_change",
    "negation",
    "identifier",
    "quoted",
    "context_version",
    "template_version",
    "param_change",
    "personal",
    "unrelated",
}


class _Gen:
    def __init__(self):
        self.calls = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls += 1
        msg = types.SimpleNamespace(content="ANSWER:" + kw["messages"][-1]["content"])
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=msg, finish_reason="stop")],
            usage=types.SimpleNamespace(prompt_tokens=200, completion_tokens=150),
        )


def load_cases():
    return [json.loads(line) for line in (HERE / "dataset.jsonl").read_text().splitlines() if line.strip()]


def run_pairs(threshold: float, live: bool):
    tmp = tempfile.mkdtemp()
    mode = "openai" if live else "mock"
    key = os.environ.get("SEMCACHE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    dim = 1536 if live else DIM
    s = Settings(_env_file=None, backend="memory", database_path=f"{tmp}/e.db", embedding_mode=mode, embedding_dim=dim, similarity_threshold=threshold, admin_token="e" * 40,
                 openai_api_key=key if live else None, log_level="ERROR", default_rate_per_minute=100000, default_daily_queries=1000000)  # fmt: skip
    gen = _Gen()
    app = create_app(s, backend=MemoryBackend(), gen_client=gen)
    client = TestClient(app)
    be, ec = app.state.runtime.backend, app.state.runtime.embed_client
    rows = []
    for case in load_cases():
        with db(s.database_path) as conn:  # a fresh tenant per case: cases cannot influence each other
            tid = auth.create_tenant(conn, case["id"])
            h = {"Authorization": f"Bearer {auth.create_api_key(conn, tid)}"}
        opts_seed = {k: v for k, v in case.get("seed_opts", {}).items()}
        client.post("/v1/cache/query", json={"query": case["seed"], **opts_seed}, headers=h)
        r = client.post("/v1/cache/query", json={"query": case["probe"], **case.get("probe_opts", {})}, headers=h).json()
        a, _, _ = embed(s, be, ec, case["seed"])
        b, _, _ = embed(s, be, ec, case["probe"])
        score = cosine(a, b)
        status = r["cache_status"]
        got = "hit" if status in ("exact_hit", "semantic_hit") else ("ineligible" if status == "ineligible" else "miss")
        rows.append({**case, "score": round(score, 4), "status": status, "got": got, "naive_hit": score >= threshold})
    return rows


def summarize(rows):
    should_hit = [r for r in rows if r["expect"] == "hit"]
    must_not = [r for r in rows if r["kind"] in KINDS_MUST_NOT_HIT]
    return {
        "cases": len(rows),
        "paraphrase_hit_rate": sum(r["got"] == "hit" for r in should_hit) / len(should_hit) if should_hit else None,
        "unsafe_hits_guarded": sum(r["got"] == "hit" for r in must_not),
        "unsafe_hits_similarity_only": sum(
            r["naive_hit"] and r["kind"] not in ("personal", "context_version", "template_version", "param_change") for r in must_not
        ),
        "must_not_cases": len(must_not),
        "correct_outcomes": sum(r["got"] == r["expect"] for r in rows),
    }


def cost_comparison(hit_rate: float, requests: int = 10_000, avg_in=200, avg_out=150):
    s = Settings(_env_file=None, backend="memory")
    per_gen = (Decimal(avg_in) * s.price_input_per_mtok + Decimal(avg_out) * s.price_output_per_mtok) / Decimal(1_000_000)
    per_emb = Decimal(30) * s.price_embedding_per_mtok / Decimal(1_000_000)
    baseline = per_gen * requests
    hits = int(requests * hit_rate)
    with_cache = per_gen * (requests - hits) + per_emb * requests
    return {"requests": requests, "assumed_hit_rate": hit_rate, "per_generation_usd": per_gen, "per_embedding_usd": per_emb, "baseline_usd": baseline,
            "with_cache_usd": with_cache, "net_savings_usd": baseline - with_cache, "break_even_hit_rate": float(per_emb / per_gen)}  # fmt: skip


def threshold_sweep(live: bool):
    out = []
    for t in (0.6, 0.7, 0.8, 0.9, 0.95):
        sm = summarize(run_pairs(t, live))
        out.append((t, sm))
    return out


def render(rows, sm, sweep, live, threshold):
    L = [
        "# Evaluation report",
        "",
        f"Embedder: **{'OpenAI (live)' if live else 'deterministic mock'}**. Threshold: **{threshold}**. Generation: fake (no completion spend).",
        "",
    ]
    if not live:
        L += [
            "> The mock embedder is a lexical hash with a tiny synonym list. These numbers test the **eligibility/safety pipeline**. They are **not** a",
            "> measurement of OpenAI embedding quality or of production hit rates. Re-run with `--live` for real embedding scores.",
            "",
        ]
    L += ["## Summary", "", f"- Cases: {sm['cases']}; outcomes matching expectation: {sm['correct_outcomes']}/{sm['cases']}",
          f"- Paraphrase hit rate (should hit): {sm['paraphrase_hit_rate']:.0%}",
          f"- Unsafe hits with the full pipeline: **{sm['unsafe_hits_guarded']}** of {sm['must_not_cases']} must-not-hit cases",
          f"- Unsafe hits a similarity-only cache would serve (same threshold, no signature/version/personal checks): **{sm['unsafe_hits_similarity_only']}**", "",
          "## Threshold sweep", "", "| threshold | paraphrase hit rate | unsafe hits (pipeline) | unsafe hits (similarity only) |", "|---|---|---|---|"]  # fmt: skip
    for t, x in sweep:
        L.append(f"| {t} | {x['paraphrase_hit_rate']:.0%} | {x['unsafe_hits_guarded']} | {x['unsafe_hits_similarity_only']} |")
    L += ["", "## Per case", "", "| id | kind | score | expected | got | status |", "|---|---|---|---|---|---|"]
    for r in rows:
        mark = "" if r["got"] == r["expect"] else " **!**"
        L.append(f"| {r['id']} | {r['kind']} | {r['score']} | {r['expect']} | {r['got']}{mark} | {r['status']} |")
    cc = cost_comparison(sm["paraphrase_hit_rate"] or 0.0)
    L += ["", "## Cost comparison (a model, not a measurement)", "",
          f"Assumptions: {cc['requests']} requests, 200 input + 150 output tokens per generation, 30 embedding tokens per request, prices from settings defaults (placeholders).",
          f"The hit rate used ({cc['assumed_hit_rate']:.0%}) is the paraphrase hit rate above applied to ALL traffic, which is optimistic: real traffic has a different mix.", "",
          "| | USD |", "|---|---|", f"| no cache | {cc['baseline_usd']:.4f} |", f"| with cache | {cc['with_cache_usd']:.4f} |", f"| estimated net savings | {cc['net_savings_usd']:.4f} |", "",
          f"Break-even hit rate (embedding cost per request / generation cost per request): {cc['break_even_hit_rate']:.2%}. Below this the cache costs more than it saves.", ""]  # fmt: skip
    L += ["## Known gaps", "", "- Small hand-written dataset; real traffic will differ.", "- Signature checks are heuristics: paraphrased numbers (\"thirty\" vs \"30\") or negation by antonym are not detected.",
          "- No measurement of answer quality for semantic hits beyond the rules above."]  # fmt: skip
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--out", default=str(HERE / "report.md"))
    a = ap.parse_args(argv)
    if a.live and not (os.environ.get("SEMCACHE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("--live needs SEMCACHE_OPENAI_API_KEY or OPENAI_API_KEY in the environment", file=sys.stderr)
        return 2
    rows = run_pairs(a.threshold, a.live)
    sm = summarize(rows)
    Path(a.out).write_text(render(rows, sm, threshold_sweep(a.live), a.live, a.threshold))
    print(json.dumps(sm, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
