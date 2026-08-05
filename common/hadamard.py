"""Walsh–Hadamard rotation for weight preprocessing (FWHT, no full H matrix).

Core guarantee: ``hadamard_rotate`` always applies an orthogonal row transform
equivalent to ``W_rot = H @ S @ W`` (axis=0; ``S`` is optional random ``±1``).
Inverse is ``hadamard_unrotate`` = ``S @ H``, not a second forward pass.
Large matrices are processed in column chunks so streaming quantization never
skips the rotation.
"""

from __future__ import annotations

import numpy as np

from .errors import QuantError
from . import runtime


def next_pow2(n: int) -> int:
    if n < 1:
        raise QuantError(f"next_pow2 expects n>=1, got {n}")
    p = 1
    while p < n:
        p <<= 1
    return p


def hadamard_matrix(n: int, seed: int | None = None) -> np.ndarray:
    """Normalized Sylvester Hadamard of order n (power of two). H @ H.T = I.

    Only for small n (tests / diagnostics). Runtime rotation uses :func:`fwht_inplace`.
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise QuantError(f"hadamard_matrix requires power-of-two n, got {n}")
    if n > 4096:
        raise QuantError(f"hadamard_matrix refuses n>{4096} (use fwht); got {n}")
    h = np.array([[1.0]], dtype=np.float64)
    while h.shape[0] < n:
        top = np.concatenate([h, h], axis=1)
        bot = np.concatenate([h, -h], axis=1)
        h = np.concatenate([top, bot], axis=0)
    h = h / np.sqrt(n)
    if seed is not None:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=n)
        h = h * signs[np.newaxis, :]
    return h.astype(np.float64)


def fwht_inplace(a: np.ndarray) -> None:
    """In-place Fast Walsh–Hadamard along axis 0; leading dim must be power of two.

    Normalizes by 1/sqrt(n) so the transform is orthogonal (norm-preserving).
    """
    if a.ndim < 1:
        raise QuantError("fwht expects at least 1D")
    n = a.shape[0]
    if n < 1 or (n & (n - 1)) != 0:
        raise QuantError(f"fwht leading dim must be power-of-two, got {n}")
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            u = a[i : i + h].copy()
            v = a[i + h : i + 2 * h].copy()
            a[i : i + h] = u + v
            a[i + h : i + 2 * h] = u - v
        h *= 2
    a *= np.float32(1.0 / np.sqrt(n))


def hadamard_rotate(
    W: np.ndarray,
    axis: int = 0,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Apply orthogonal Hadamard rotation ``W_rot = H @ W`` (axis=0 only).

    Uses FWHT with zero-pad to the next power of two, then crops back. When the
    full padded buffer would exceed the RAM budget, columns are processed in
    chunks — mathematically identical to a single ``H @ W``, never skipped.

    ``axis`` must be 0 (core Spec). Returns ``meta["applied"]=True`` whenever
    ``K >= 2`` after padding target; for ``K==1`` rotation is identity.
    """
    if W.ndim != 2:
        raise QuantError(f"hadamard_rotate expects 2D weight, got shape {W.shape}")
    if axis != 0:
        raise QuantError(
            "core path requires axis=0 (W_rot = H @ W); "
            f"got axis={axis}"
        )

    W32 = np.asarray(W, dtype=np.float32)
    k, n = W32.shape
    meta: dict = {
        "seed": seed,
        "row_dim": k,
        "col_dim": n,
        "row_pad": 0,
        "col_pad": 0,
        "axis": 0,
        "applied": False,
        "chunked": False,
    }
    if k < 1 or n < 1:
        raise QuantError("empty weight")

    target = next_pow2(k)
    pad = target - k
    meta["row_pad"] = pad

    if target < 2:
        # 1xN: Hadamard of order 1 is [1]; identity.
        meta["applied"] = True
        meta["identity"] = True
        return W32.copy(), meta

    signs = None
    if seed is not None:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=target)

    max_elems = runtime.max_work_elems()
    # At least one column must fit: target * 1
    if target > max_elems:
        raise QuantError(
            f"Hadamard row pad length {target} exceeds work budget {max_elems}; "
            "increase ARIA_QUANT_MAX_ELEMS / host RAM"
        )
    chunk_w = max(1, int(max_elems // target))
    chunk_w = min(chunk_w, n)
    meta["chunked"] = chunk_w < n
    meta["chunk_width"] = chunk_w

    out = np.empty((k, n), dtype=np.float32)
    for start in range(0, n, chunk_w):
        end = min(n, start + chunk_w)
        cols = end - start
        work = np.zeros((target, cols), dtype=np.float32)
        work[:k, :] = W32[:, start:end]
        if signs is not None:
            work *= signs[:, None]
        fwht_inplace(work)
        out[:, start:end] = work[:k, :]

    meta["applied"] = True
    return out, meta


def hadamard_unrotate(
    W_rot: np.ndarray,
    axis: int = 0,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Inverse of :func:`hadamard_rotate` along axis=0.

    Forward is ``T = H @ S`` (row-scale by random ``±1``, then FWHT). Inverse is
    ``T^{-1} = S @ H`` (FWHT, then the same row-scale) — **not** applying ``T``
    twice. Zero-pad / crop mirrors the forward path; exact for power-of-two rows.
    """
    if W_rot.ndim != 2:
        raise QuantError(f"hadamard_unrotate expects 2D weight, got shape {W_rot.shape}")
    if axis != 0:
        raise QuantError(
            "core path requires axis=0; "
            f"got axis={axis}"
        )

    W32 = np.asarray(W_rot, dtype=np.float32)
    k, n = W32.shape
    meta: dict = {
        "seed": seed,
        "row_dim": k,
        "col_dim": n,
        "row_pad": 0,
        "col_pad": 0,
        "axis": 0,
        "applied": False,
        "chunked": False,
        "inverse": True,
    }
    if k < 1 or n < 1:
        raise QuantError("empty weight")

    target = next_pow2(k)
    pad = target - k
    meta["row_pad"] = pad

    if target < 2:
        meta["applied"] = True
        meta["identity"] = True
        return W32.copy(), meta

    signs = None
    if seed is not None:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=target)

    max_elems = runtime.max_work_elems()
    if target > max_elems:
        raise QuantError(
            f"Hadamard row pad length {target} exceeds work budget {max_elems}; "
            "increase ARIA_QUANT_MAX_ELEMS / host RAM"
        )
    chunk_w = max(1, int(max_elems // target))
    chunk_w = min(chunk_w, n)
    meta["chunked"] = chunk_w < n
    meta["chunk_width"] = chunk_w

    out = np.empty((k, n), dtype=np.float32)
    for start in range(0, n, chunk_w):
        end = min(n, start + chunk_w)
        cols = end - start
        work = np.zeros((target, cols), dtype=np.float32)
        work[:k, :] = W32[:, start:end]
        fwht_inplace(work)
        if signs is not None:
            work *= signs[:, None]
        out[:, start:end] = work[:k, :]

    meta["applied"] = True
    return out, meta
