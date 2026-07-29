"""Lloyd-Max / K-Means codebook for scalar quantization."""

from __future__ import annotations

import numpy as np

from .errors import QuantError


def lloyd_max(
    x: np.ndarray,
    k: int,
    max_iter: int = 50,
    tol: float = 1e-6,
    seed: int | None = 0,
) -> np.ndarray:
    """1D Lloyd-Max clustering; returns codebook shape (k,)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if k < 1:
        raise QuantError(f"codebook k must be >=1, got {k}")
    if x.size == 0:
        raise QuantError("lloyd_max on empty vector")
    if not np.isfinite(x).all():
        raise QuantError("lloyd_max input contains non-finite values")

    if np.unique(x).size == 1:
        return np.full(k, x[0], dtype=np.float64)

    rng = np.random.default_rng(seed)
    # init: linspace between percentiles for stability
    lo, hi = np.percentile(x, [2, 98])
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.full(k, lo, dtype=np.float64)
    codebook = np.linspace(lo, hi, k, dtype=np.float64)
    # small jitter
    codebook = codebook + rng.normal(0.0, 1e-8, size=k)

    prev = None
    for _ in range(max_iter):
        # assign
        dists = np.abs(x[:, None] - codebook[None, :])
        labels = dists.argmin(axis=1)
        new_cb = codebook.copy()
        for i in range(k):
            mask = labels == i
            if mask.any():
                new_cb[i] = x[mask].mean()
            else:
                # re-seed empty cluster
                new_cb[i] = x[rng.integers(0, x.size)]
        new_cb.sort()
        if prev is not None and np.max(np.abs(new_cb - prev)) < tol:
            codebook = new_cb
            break
        prev = new_cb
        codebook = new_cb
    return codebook.astype(np.float64)


def quantize_group(col: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Nearest-neighbor indices into codebook; uint8 length = col.size."""
    col = np.asarray(col, dtype=np.float64).ravel()
    codebook = np.asarray(codebook, dtype=np.float64).ravel()
    if codebook.size < 1:
        raise QuantError("empty codebook")
    dists = np.abs(col[:, None] - codebook[None, :])
    return dists.argmin(axis=1).astype(np.uint8)
