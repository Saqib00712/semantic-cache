"""Test doubles. FakeGen imitates the OpenAI client surface the app uses (chat.completions.create)."""

import threading
import time
import types

import httpx
import openai


def _req():
    return httpx.Request("POST", "https://api.openai.test/v1/x")


def openai_error(kind: str):
    resp = lambda code: httpx.Response(code, request=_req())  # noqa: E731
    return {
        "rate_limit": lambda: openai.RateLimitError("slow down", response=resp(429), body=None),
        "timeout": lambda: openai.APITimeoutError(request=_req()),
        "connection": lambda: openai.APIConnectionError(request=_req()),
        "auth": lambda: openai.AuthenticationError("bad key", response=resp(401), body=None),
        "forbidden": lambda: openai.PermissionDeniedError("denied", response=resp(403), body=None),
        "bad_request": lambda: openai.BadRequestError("bad", response=resp(400), body=None),
        "server": lambda: openai.InternalServerError("boom", response=resp(500), body=None),
    }[kind]()


class FakeGen:
    """Counts calls; can delay, fail the first N calls with a given error, or return a fixed finish_reason / content."""

    def __init__(
        self, delay: float = 0.0, fail_first: int = 0, error: str = "server", finish_reason: str = "stop", content=None, prefix="ANSWER"
    ):
        self.calls, self.delay, self.fail_first, self.error, self.finish_reason, self.content, self.prefix = (
            0,
            delay,
            fail_first,
            error,
            finish_reason,
            content,
            prefix,
        )
        self.lock, self.last_kwargs, self.all_kwargs = threading.Lock(), None, []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **kw):
        with self.lock:
            self.calls += 1
            n = self.calls
            self.last_kwargs = kw
            self.all_kwargs.append(kw)
        if self.delay:
            time.sleep(self.delay)
        if n <= self.fail_first:
            raise openai_error(self.error)
        text = self.content if self.content is not None else f"{self.prefix}[{kw['messages'][-1]['content']}]"
        msg = types.SimpleNamespace(content=text)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason=self.finish_reason)],
                                     usage=types.SimpleNamespace(prompt_tokens=20, completion_tokens=10))  # fmt: skip


class FakeEmbed:
    def __init__(self, dim=8, vector=None, fail_first=0, error="server"):
        self.calls, self.dim, self.vector, self.fail_first, self.error = 0, dim, vector, fail_first, error
        self.embeddings = types.SimpleNamespace(create=self.create)
        self.last_kwargs = None

    def create(self, **kw):
        self.calls += 1
        self.last_kwargs = kw
        if self.calls <= self.fail_first:
            raise openai_error(self.error)
        v = self.vector if self.vector is not None else [1.0] + [0.0] * (self.dim - 1)
        return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=v)], usage=types.SimpleNamespace(prompt_tokens=5))
