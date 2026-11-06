"""Prometheus text metrics, hand-rendered (no extra dependency). Labels are low-cardinality fixed vocabularies: never tenant ids,
prompts, answers, keys or entry ids."""

import threading

_lock = threading.Lock()
_counters: dict[tuple[str, tuple], float] = {}
_hist: dict[tuple[str, tuple], list[float]] = {}
BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
HELP = {
    "semcache_requests_total": "Cache queries by outcome (cache_status).",
    "semcache_rejections_total": "Semantic candidates rejected, by reason.",
    "semcache_bypass_total": "Requests that skipped the cache, by reason.",
    "semcache_errors_total": "Errors by component.",
    "semcache_openai_calls_total": "OpenAI calls by kind and result.",
    "semcache_invalidations_total": "Invalidation operations by scope.",
    "semcache_degraded_total": "Requests served in degraded mode, by cause.",
    "semcache_request_seconds": "End-to-end request latency by cache_status.",
    "semcache_lookup_seconds": "Cache lookup latency (exact + embedding + vector search).",
}


def inc(name: str, amount: float = 1.0, **labels: str) -> None:
    key = (name, tuple(sorted(labels.items())))
    with _lock:
        _counters[key] = _counters.get(key, 0.0) + amount


def observe(name: str, seconds: float, **labels: str) -> None:
    key = (name, tuple(sorted(labels.items())))
    with _lock:
        h = _hist.setdefault(key, [0.0] * (len(BUCKETS) + 2))  # buckets..., +Inf count, sum
        for i, b in enumerate(BUCKETS):
            if seconds <= b:
                h[i] += 1
        h[len(BUCKETS)] += 1
        h[len(BUCKETS) + 1] += seconds


def reset() -> None:
    with _lock:
        _counters.clear()
        _hist.clear()


def get(name: str, **labels: str) -> float:
    with _lock:
        return _counters.get((name, tuple(sorted(labels.items()))), 0.0)


def _clean(v) -> str:
    return str(v).replace("\\", "").replace('"', "")


def _fmt(labels: tuple, extra: str = "") -> str:
    parts = ['%s="%s"' % (k, _clean(v)) for k, v in labels] + ([extra] if extra else [])
    return "{" + ",".join(parts) + "}" if parts else ""


def render() -> str:
    out, seen = [], set()
    with _lock:
        for (name, labels), v in sorted(_counters.items()):
            if name not in seen:
                out += [f"# HELP {name} {HELP.get(name, name)}", f"# TYPE {name} counter"]
                seen.add(name)
            out.append(f"{name}{_fmt(labels)} {v:g}")
        for (name, labels), h in sorted(_hist.items()):
            if name not in seen:
                out += [f"# HELP {name} {HELP.get(name, name)}", f"# TYPE {name} histogram"]
                seen.add(name)
            for i, b in enumerate(BUCKETS):
                out.append("%s_bucket%s %g" % (name, _fmt(labels, 'le="%s"' % b), h[i]))
            out.append("%s_bucket%s %g" % (name, _fmt(labels, 'le="+Inf"'), h[len(BUCKETS)]))
            out.append(f"{name}_sum{_fmt(labels)} {h[len(BUCKETS) + 1]:.6f}")
            out.append(f"{name}_count{_fmt(labels)} {h[len(BUCKETS)]:g}")
    return "\n".join(out) + "\n"