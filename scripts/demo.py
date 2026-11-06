"""Offline demo: mock embeddings + a fake model, in-process, no keys or Redis. Shows exact hit, semantic hit, and the guards refusing
to reuse an answer when the meaning changed. Run: python scripts/demo.py"""

import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.security import auth  # noqa: E402
from app.storage.database import db  # noqa: E402


class FakeModel:
    def __init__(self):
        self.calls = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls += 1
        msg = types.SimpleNamespace(content="(model answer to) " + kw["messages"][-1]["content"])
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=msg, finish_reason="stop")],
            usage=types.SimpleNamespace(prompt_tokens=60, completion_tokens=40),
        )


def main() -> None:
    s = Settings(
        _env_file=None,
        backend="memory",
        database_path=tempfile.mkdtemp() + "/demo.db",
        embedding_mode="mock",
        embedding_dim=256,
        similarity_threshold=0.8,
        log_level="ERROR",
    )
    model = FakeModel()
    client = TestClient(create_app(s, gen_client=model))
    with db(s.database_path) as conn:
        h = {"Authorization": "Bearer " + auth.create_api_key(conn, auth.create_tenant(conn, "demo"))}
    for q in ["How can I get a refund within 30 days?", "How can I get a refund within 30 days?", "How do I get my money back within 30 days",
              "How can I get a refund within 60 days?", "Where is my order 482913?"]:  # fmt: skip
        r = client.post("/v1/cache/query", json={"query": q}, headers=h).json()
        print(f"{r['cache_status']:13} score={r['similarity_score']}  <- {q}")
    print(f"\nmodel calls: {model.calls} for 5 requests")


if __name__ == "__main__":
    main()
