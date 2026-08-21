"""Walsh–Hadamard rotation for weight preprocessing (FWHT, no full H matrix).

Live path (format_version≥2): **blocked** row transform — greedy largest
power-of-two tiles; per block ``W_rot = H_B @ S_B @ W``; inverse
``S_B @ H_B`` (not a second forward). Shape always ``(K, N)`` — no global pad/crop.

Legacy pad-crop helpers remain for auditing old bundles only.
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


def largest_pow2_le(n: int) -> int:
    if n < 1:
        raise QuantError(f"largest_pow2_le expects n>=1, got {n}")
    p = 1
    while (p << 1) <= n:
        p <<= 1
    return p


def pow2_tile_sizes(k: int) -> list[int]:
    """Greedy largest-pow2 tiling of row count ``k`` (e.g. 10 → [8, 2])."""
    if k < 1:
        raise QuantError(f"pow2_tile_sizes expects k>=1, got {k}")
    sizes: list[int] = []
    rem = k
    while rem > 0:
        b = largest_pow2_le(rem)
        sizes.append(b)
        rem -= b
    return sizes


def pow2_tile_blocks(k: int) -> list[dict]:
    """Return ``[{start, size}, ...]`` covering ``0..k``."""
    blocks: list[dict] = []
    start = 0
    for size in pow2_tile_sizes(k):
        blocks.append({"start": start, "size": size})
        start += size
    return blocks


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
    if n == 1:
        return
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            u = a[i : i + h].copy()
            v = a[i + h : i + 2 * h].copy()
            a[i : i + h] = u + v
            a[i + h : i + 2 * h] = u - v
        h *= 2
    a *= np.float32(1.0 / np.sqrt(n))


def fwht_torch(a: "object") -> "object":
    """Batched FWHT along dim 0 for a ``(n, ...)`` torch tensor (n power of two).

    Returns a new tensor; does not mutate ``a``. Normalizes by ``1/sqrt(n)``.
    """
    import torch

    if not isinstance(a, torch.Tensor):
        raise QuantError("fwht_torch expects a torch.Tensor")
    if a.ndim < 1:
        raise QuantError("fwht expects at least 1D")
    n = int(a.shape[0])
    if n < 1 or (n & (n - 1)) != 0:
        raise QuantError(f"fwht leading dim must be power-of-two, got {n}")
    if n == 1:
        return a.clone()
    out = a.to(dtype=torch.float32).contiguous().clone()
    shape = tuple(out.shape)
    cols = int(out.numel() // n)
    h = 1
    while h < n:
        x = out.view(n // (2 * h), 2, h, cols)
        u = x[:, 0].clone()
        v = x[:, 1].clone()
        x[:, 0].copy_(u + v)
        x[:, 1].copy_(u - v)
        h *= 2
    out.view(n, cols).mul_(1.0 / float(n) ** 0.5)
    return out.view(shape)


def hadamard_unrotate_torch(
    W_rot: "object",
    *,
    seed: int | None = None,
    device: str | object | None = None,
) -> tuple["object", dict]:
    """Torch blocked unrotate (same tiles/signs as :func:`hadamard_unrotate`)."""
    import torch

    if not isinstance(W_rot, torch.Tensor):
        W_rot = torch.as_tensor(np.asarray(W_rot, dtype=np.float32))
    if device is not None:
        W_rot = W_rot.to(device)
    W_rot = W_rot.to(dtype=torch.float32).contiguous()
    if W_rot.ndim != 2:
        raise QuantError(f"hadamard_unrotate_torch expects 2D, got {tuple(W_rot.shape)}")
    k, n = int(W_rot.shape[0]), int(W_rot.shape[1])
    blocks = pow2_tile_blocks(k)
    meta: dict = {
        "seed": seed,
        "row_dim": k,
        "col_dim": n,
        "row_pad": 0,
        "col_pad": 0,
        "axis": 0,
        "applied": True,
        "chunked": False,
        "mode": "blocked",
        "blocks": blocks,
        "inverse": True,
        "backend": "torch",
    }
    if k == 1:
        meta["identity"] = True
        return W_rot.clone(), meta

    signs_map = _block_signs(seed, blocks)
    out = torch.empty_like(W_rot)
    for b in blocks:
        rs = int(b["start"])
        sz = int(b["size"])
        re = rs + sz
        work = W_rot[rs:re].contiguous()
        work = fwht_torch(work)
        if signs_map is not None:
            signs = torch.as_tensor(signs_map[rs], device=work.device, dtype=work.dtype)
            work = work * signs[:, None]
        out[rs:re] = work
    return out, meta


def portable_block_signs(seed: int, start: int, size: int) -> np.ndarray:
    """Deterministic ±1 for a block; must match engine ``portable_block_signs``.

    SplitMix64-derived parity (not numpy PCG) so Rust can bit-match without
    reimplementing NumPy's Generator.
    """
    if size < 1:
        raise QuantError(f"portable_block_signs size must be >=1, got {size}")
    signs = np.empty(size, dtype=np.float32)
    state = (int(seed) ^ (int(start) * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
    for i in range(size):
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        z = z ^ (z >> 31)
        signs[i] = 1.0 if (z & 1) == 0 else -1.0
    return signs


def _block_signs(seed: int | None, blocks: list[dict]) -> dict[int, np.ndarray] | None:
    if seed is None:
        return None
    out: dict[int, np.ndarray] = {}
    for b in blocks:
        start = int(b["start"])
        size = int(b["size"])
        out[start] = portable_block_signs(int(seed), start, size)
    return out


def _apply_blocked(
    W32: np.ndarray,
    *,
    seed: int | None,
    inverse: bool,
) -> tuple[np.ndarray, dict]:
    k, n = W32.shape
    blocks = pow2_tile_blocks(k)
    meta: dict = {
        "seed": seed,
        "row_dim": k,
        "col_dim": n,
        "row_pad": 0,
        "col_pad": 0,
        "axis": 0,
        "applied": True,
        "chunked": False,
        "mode": "blocked",
        "blocks": blocks,
    }
    if inverse:
        meta["inverse"] = True

    if k == 1:
        meta["identity"] = True
        return W32.copy(), meta

    signs_map = _block_signs(seed, blocks)
    max_elems = runtime.max_work_elems()
    max_block = max(int(b["size"]) for b in blocks)
    if max_block > max_elems:
        raise QuantError(
            f"Hadamard block length {max_block} exceeds work budget {max_elems}; "
            "increase ARIA_QUANT_MAX_ELEMS / host RAM"
        )
    chunk_w = max(1, int(max_elems // max_block))
    chunk_w = min(chunk_w, n)
    meta["chunked"] = chunk_w < n
    meta["chunk_width"] = chunk_w

    out = np.empty((k, n), dtype=np.float32)
    for c0 in range(0, n, chunk_w):
        c1 = min(n, c0 + chunk_w)
        for b in blocks:
            rs = int(b["start"])
            sz = int(b["size"])
            re = rs + sz
            work = np.array(W32[rs:re, c0:c1], dtype=np.float32, copy=True)
            signs = None if signs_map is None else signs_map[rs]
            if not inverse:
                if signs is not None:
                    work *= signs[:, None]
                fwht_inplace(work)
            else:
                fwht_inplace(work)
                if signs is not None:
                    work *= signs[:, None]
            out[rs:re, c0:c1] = work
    return out, meta


def hadamard_rotate(
    W: np.ndarray,
    axis: int = 0,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Blocked orthogonal Hadamard ``W_rot = H@S@W`` on axis=0 (live path)."""
    if W.ndim != 2:
        raise QuantError(f"hadamard_rotate expects 2D weight, got shape {W.shape}")
    if axis != 0:
        raise QuantError(
            "core path requires axis=0 (W_rot = H @ W); "
            f"got axis={axis}"
        )
    W32 = np.asarray(W, dtype=np.float32)
    if W32.shape[0] < 1 or W32.shape[1] < 1:
        raise QuantError("empty weight")
    return _apply_blocked(W32, seed=seed, inverse=False)


def hadamard_unrotate(
    W_rot: np.ndarray,
    axis: int = 0,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Inverse of blocked :func:`hadamard_rotate` (``S@H`` per block)."""
    if W_rot.ndim != 2:
        raise QuantError(f"hadamard_unrotate expects 2D weight, got shape {W_rot.shape}")
    if axis != 0:
        raise QuantError(f"core path requires axis=0; got axis={axis}")
    W32 = np.asarray(W_rot, dtype=np.float32)
    if W32.shape[0] < 1 or W32.shape[1] < 1:
        raise QuantError("empty weight")
    return _apply_blocked(W32, seed=seed, inverse=True)


def hadamard_rotate_padded_legacy(
    W: np.ndarray,
    axis: int = 0,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Legacy global pad→FWHT→crop (format_version 1). Not used by quantize."""
    if W.ndim != 2:
        raise QuantError(f"hadamard_rotate expects 2D weight, got shape {W.shape}")
    if axis != 0:
        raise QuantError(f"core path requires axis=0; got axis={axis}")

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
        "mode": "padded_legacy",
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
        if signs is not None:
            work *= signs[:, None]
        fwht_inplace(work)
        out[:, start:end] = work[:k, :]

    meta["applied"] = True
    return out, meta


def hadamard_unrotate_with_ref(
    W_rot: np.ndarray,
    W_ref: np.ndarray,
    axis: int = 0,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Legacy pad-crop audit inverse (v1). Prefer :func:`hadamard_unrotate` for v2."""
    if W_rot.ndim != 2 or W_ref.ndim != 2:
        raise QuantError("hadamard_unrotate_with_ref expects 2D weights")
    if W_rot.shape != W_ref.shape:
        raise QuantError(
            f"shape mismatch W_rot {W_rot.shape} vs W_ref {W_ref.shape}"
        )
    if axis != 0:
        raise QuantError(f"core path requires axis=0; got axis={axis}")

    # For blocked (live) tensors, blocked unrotate is exact — use it.
    out, meta = hadamard_unrotate(W_rot, axis=0, seed=seed)
    meta["pad_mode"] = "none"
    return out, meta
