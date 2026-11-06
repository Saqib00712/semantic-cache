"""Vector helpers. Embeddings are stored unit-normalized float32, so cosine similarity is a plain dot product."""

import math

import numpy as np


class VectorError(ValueError):
    """A vector that must not be stored or searched (wrong size, NaN/inf, zero length)."""


def validate_vector(values, dim: int) -> np.ndarray:
    try:
        v = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError):
        raise VectorError("embedding is not numeric") from None
    if v.ndim != 1 or v.shape[0] != dim:
        raise VectorError(f"embedding has wrong dimension (got {v.shape}, expected {dim})")
    if not np.all(np.isfinite(v)):
        raise VectorError("embedding contains NaN or infinity")
    n = float(np.linalg.norm(v))
    if n < 1e-9 or not math.isfinite(n):
        raise VectorError("embedding has zero length")
    return (v / n).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
