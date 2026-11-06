"""One JSON object per log line. Only allow-listed fields are written, so a prompt, answer, header or exception text cannot reach
the logs by accident. Redaction is a second line of defence: key-shaped strings, Authorization values and registered secrets."""

import json
import logging
import re
import sys
import time
from typing import IO

from app.observability.context import request_id_var

ALLOWED_FIELDS = (
    "event", "method", "path", "status", "duration_ms", "error_type", "detail", "cache_status", "decision", "reason",
    "tenant_id", "entry_id", "code", "attempt", "component",
)  # fmt: skip
_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{8,}"), "[REDACTED_KEY]"),
    (re.compile(r"sc_[a-f0-9]{8}_[A-Za-z0-9_\-]{16,}"), "[REDACTED_KEY]"),  # this service's own API keys
    (re.compile(r"(?i)(bearer|basic)\s+[A-Za-z0-9._~+/=\-]{6,}"), r"\1 [REDACTED]"),
    (re.compile(r"(?i)(authorization|x-api-key|api[_-]?key|password|secret|token)(\"?\s*[:=]\s*\"?)[^\s\",}]+"), r"\1\2[REDACTED]"),
    (re.compile(r"([a-z][a-z0-9+.\-]*://)[^/\s:@]*:[^/\s@]+@"), r"\1[REDACTED]@"),
]
_registered: set[str] = set()


def register_secret(value: str) -> None:
    if value and len(value) >= 8:
        _registered.add(value)


def redact(text: str) -> str:
    for secret in _registered:
        text = text.replace(secret, "[REDACTED]")
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z"
        entry: dict[str, object] = {"ts": ts, "level": record.levelname, "message": record.getMessage(), "request_id": request_id_var.get()}
        for name in ALLOWED_FIELDS:
            value = getattr(record, name, None)
            if value is not None:
                entry[name] = value
        return redact(json.dumps(entry, default=str))


def configure_logging(level: str = "INFO", stream: IO[str] | None = None) -> logging.Logger:
    logger = logging.getLogger("semcache")
    logger.setLevel(level)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("semcache")
