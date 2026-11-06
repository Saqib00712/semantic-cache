import shutil
import socket
import subprocess
import time
import uuid

import pytest
import redis
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.observability import metrics
from app.security import auth
from app.storage.database import db
from app.storage.memory import MemoryBackend
from app.storage.redis_backend import RedisBackend
from tests.helpers import FakeGen

ADMIN = "a" * 40
DIM = 256


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def redis_url(tmp_path_factory):
    """A throwaway local redis-server (plain Redis, no RediSearch). Tests that need Redis skip when no server binary exists."""
    binary = shutil.which("redis-server")
    if not binary:
        pytest.skip("redis-server not installed")
    port = _free_port()
    proc = subprocess.Popen([binary, "--port", str(port), "--save", "", "--appendonly", "no", "--bind", "127.0.0.1"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=tmp_path_factory.mktemp("redis"))  # fmt: skip
    url = f"redis://127.0.0.1:{port}/0"
    for _ in range(100):
        try:
            redis.Redis.from_url(url).ping()
            break
        except redis.RedisError:
            time.sleep(0.05)
    else:
        proc.kill()
        pytest.skip("redis-server did not start")
    yield url
    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture(params=["memory", "redis"])
def backend(request):
    if request.param == "memory":
        yield MemoryBackend()
        return
    url = request.getfixturevalue("redis_url")
    client = redis.Redis.from_url(url)
    client.flushdb()
    be = RedisBackend(client, "scan", DIM)
    yield be
    client.flushdb()


@pytest.fixture
def make_settings(tmp_path):
    def make(**over):
        base = dict(database_path=str(tmp_path / f"{uuid.uuid4().hex}.db"), backend="memory", embedding_mode="mock", embedding_dim=DIM,
                    admin_token=ADMIN, similarity_threshold=0.8, retry_base_seconds=0.0, lock_poll_seconds=0.01, lock_wait_seconds=5.0,
                    log_level="ERROR")  # fmt: skip
        base.update(over)
        return Settings(_env_file=None, **base)

    return make


@pytest.fixture(autouse=True)
def _reset_metrics():
    metrics.reset()


class Env:
    def __init__(self, app, client, gen, settings, backend):
        self.app, self.client, self.gen, self.settings, self.backend = app, client, gen, settings, backend

    def tenant(self, name="t", **kw):
        with db(self.settings.database_path) as conn:
            tid = auth.create_tenant(conn, name, **kw)
            key = auth.create_api_key(conn, tid)
        return tid, {"Authorization": f"Bearer {key}"}

    def q(self, headers, query, **body):
        return self.client.post("/v1/cache/query", json={"query": query, **body}, headers=headers)

    @property
    def admin(self):
        return {"Authorization": f"Bearer {ADMIN}"}


@pytest.fixture
def env(backend, make_settings):
    gen = FakeGen()
    s = make_settings()
    app = create_app(s, backend=backend, gen_client=gen, sleep=lambda x: None)
    with TestClient(app) as c:
        yield Env(app, c, gen, s, backend)


@pytest.fixture
def make_env(make_settings):
    """Build an Env with custom settings / fakes (memory backend unless one is given)."""

    def build(*, backend=None, gen=None, embed_client=None, sleep=lambda x: None, **settings):
        gen = gen or FakeGen()
        s = make_settings(**settings)
        be = backend or MemoryBackend()
        app = create_app(s, backend=be, gen_client=gen, embed_client=embed_client, sleep=sleep)
        return Env(app, TestClient(app), gen, s, be)

    return build
