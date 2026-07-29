"""Walsh–Hadamard rotation for weight preprocessing."""

from __future__ import annotations

import numpy as np

from .errors import QuantError


def next_pow2(n: int) -> int:
    if n < 1:
        raise QuantError(f"next_pow2 expects n>=1, got {n}")
    p = 1
    while p < n:
        p <<= 1
    return p


def hadamard_matrix(n: int, seed: int | None = None) -> np.ndarray:
    """Normalized Sylvester Hadamard of order n (power of two). H @ H.T = I."""
    if n < 1 or (n & (n - 1)) != 0:
        raise QuantError(f"hadamard_matrix requires power-of-two n, got {n}")
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
        # re-normalize columns already unit via diag ±1 on orthonormal H
    return h.astype(np.float64)


def hadamard_rotate(
    W: np.ndarray,
    axis: int = 0,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Left-multiply rows by Hadamard: W_rot = H @ W (axis=0).

    Non-power-of-two row dim is zero-padded to next_pow2 then cropped back.
    """
    if W.ndim != 2:
        raise QuantError(f"hadamard_rotate expects 2D weight, got shape {W.shape}")
    if axis != 0:
        raise QuantError("only axis=0 (row) Hadamard is supported in v1")

    W = np.asarray(W, dtype=np.float64)
    k, n = W.shape
    meta = {"seed": seed, "row_dim": k, "row_pad": 0, "applied": False}
    if k < 1:
        raise QuantError("empty weight rows")

    target = next_pow2(k)
    pad = target - k
    meta["row_pad"] = pad
    if pad:
        W_work = np.zeros((target, n), dtype=np.float64)
        W_work[:k, :] = W
    else:
        W_work = W

    if target < 2:
        meta["applied"] = False
        return W.astype(np.float64), meta

    H = hadamard_matrix(target, seed=seed)
    W_rot_full = H @ W_work
    W_rot = W_rot_full[:k, :]
    meta["applied"] = True
    return W_rot, meta
