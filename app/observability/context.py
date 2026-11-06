"""The current request's correlation id (a contextvar, so every log line in a request carries it)."""

import re
import uuid
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
_SAFE = re.compile(r"^[A-Za-z0-9._\-]{8,64}$")


def accept_request_id(value: str | None) -> str | None:
    """A client-supplied id is kept only if it is short and harmless (no log injection through it)."""
    return value if value and _SAFE.match(value) else None


def new_request_id() -> str:
    return uuid.uuid4().hex
