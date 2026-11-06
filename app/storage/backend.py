"""The storage interface the cache needs, and the entry record. Two implementations: Redis (production) and memory (single process:
tests, a Windows demo without Redis). A class hierarchy is justified here: two interchangeable stateful backends behind one contract.

Every method is scoped by a tenant namespace `ns` (see keys.namespace). A backend never searches or deletes across namespaces."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class StoredEntry:
    meta: dict[str, Any]  # raw (untrusted until parsed as EntryMeta)
    answer: str
    vec: np.ndarray | None
    hits: int = 0


class BackendError(Exception):
    """The storage layer failed (connection, timeout, protocol). Safe text only."""


class Backend:
    name = "base"
    vector_mode = "none"

    def ping(self) -> bool:
        raise NotImplementedError  # noqa: E701

    def get_exact(self, ns: str, exact_key: str) -> StoredEntry | None:
        raise NotImplementedError  # noqa: E701

    def get(self, ns: str, entry_id: str) -> StoredEntry | None:
        raise NotImplementedError  # noqa: E701

    def put(
        self,
        ns: str,
        entry_id: str,
        exact_key: str,
        meta: dict,
        answer: str,
        vec: np.ndarray | None,
        ttl: int,
        max_entries: int,
        expect_epoch: int,
    ) -> bool:
        raise NotImplementedError  # noqa: E701

    def search(self, ns: str, vec: np.ndarray, k: int, embedding_model: str) -> list[tuple[str, float]]:
        raise NotImplementedError  # noqa: E701

    def delete(self, ns: str, entry_id: str) -> bool:
        raise NotImplementedError  # noqa: E701

    def delete_matching(self, ns: str, filters: dict[str, str]) -> int:
        raise NotImplementedError  # noqa: E701

    def list_entries(self, ns: str, limit: int) -> list[dict]:
        raise NotImplementedError  # noqa: E701

    def count(self, ns: str) -> int:
        raise NotImplementedError  # noqa: E701

    def record_hit(self, ns: str, entry_id: str) -> None:
        raise NotImplementedError  # noqa: E701

    def bump_epoch(self, ns: str) -> int:
        raise NotImplementedError  # noqa: E701

    def get_epoch(self, ns: str) -> int:
        raise NotImplementedError  # noqa: E701

    def cleanup_expired(self, ns: str) -> int:
        raise NotImplementedError  # noqa: E701

    def emb_get(self, key: str) -> tuple[np.ndarray, int] | None:
        raise NotImplementedError  # noqa: E701

    def emb_put(self, key: str, vec: np.ndarray, tokens: int, ttl: int) -> None:
        raise NotImplementedError  # noqa: E701

    def lock_acquire(self, ns: str, key: str, token: str, ttl: float) -> bool:
        raise NotImplementedError  # noqa: E701

    def lock_release(self, ns: str, key: str, token: str) -> bool:
        raise NotImplementedError  # noqa: E701

    def flight_put(self, ns: str, key: str, payload: str, ttl: int) -> None:
        raise NotImplementedError  # noqa: E701

    def flight_get(self, ns: str, key: str) -> str | None:
        raise NotImplementedError  # noqa: E701

    def incr_window(self, ns: str, name: str, ttl: int) -> int:
        raise NotImplementedError  # noqa: E701

    def close(self) -> None:
        pass  # noqa: E701
