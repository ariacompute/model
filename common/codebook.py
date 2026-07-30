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
    lo, hi = np.percentile(x, [2, 98])
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.full(k, lo, dtype=np.float64)
    codebook = np.linspace(lo, hi, k, dtype=np.float64)
    codebook = codebook + rng.normal(0.0, 1e-8, size=k)

    prev = None
    for _ in range(max_iter):
        dists = np.abs(x[:, None] - codebook[None, :])
        labels = dists.argmin(axis=1)
        new_cb = codebook.copy()
        for i in range(k):
            mask = labels == i
            if mask.any():
                new_cb[i] = x[mask].mean()
            else:
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


def lloyd_max_columns(
    block: np.ndarray,
    k: int,
    max_iter: int = 20,
    tol: float = 1e-6,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lloyd-Max for every column of ``block`` (group_size, n).

    Returns ``(codebooks, scales, norms, indices)`` with shapes
    ``(n, k)``, ``(n,)``, ``(n,)``, ``(group_size, n)``.
    """
    block = np.asarray(block, dtype=np.float32)
    if block.ndim != 2:
        raise QuantError(f"lloyd_max_columns expects 2D block, got {block.shape}")
    gs, n = block.shape
    if k < 1:
        raise QuantError(f"codebook k must be >=1, got {k}")

    codebooks = np.zeros((n, k), dtype=np.float32)
    scales = np.zeros(n, dtype=np.float32)
    norms = np.zeros(n, dtype=np.float32)
    indices = np.zeros((gs, n), dtype=np.uint8)
    base = 0 if seed is None else int(seed)

    for j in range(n):
        col = block[:, j]
        scale = float(np.max(np.abs(col))) if col.size else 0.0
        if scale < 1e-12:
            scale = 1.0
        scales[j] = scale
        norms[j] = float(np.linalg.norm(col))
        cb = lloyd_max(col, k, max_iter=max_iter, tol=tol, seed=base + j)
        codebooks[j, :] = cb.astype(np.float32)
        indices[:, j] = quantize_group(col, cb)
    return codebooks, scales, norms, indices


def lloyd_max_columns_torch(
    block: np.ndarray,
    k: int,
    max_iter: int = 20,
    tol: float = 1e-6,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batched Lloyd-Max on CUDA for all columns of a group block."""
    import torch

    block = np.asarray(block, dtype=np.float32)
    gs, n = block.shape
    device = torch.device("cuda")
    x = torch.from_numpy(block.T.copy()).to(device)  # (n, gs)
    scales = x.abs().amax(dim=1)
    scales = torch.where(scales < 1e-12, torch.ones_like(scales), scales)
    norms = torch.linalg.vector_norm(x, dim=1)

    # percentile-ish init via sorted samples
    xs, _ = torch.sort(x, dim=1)
    idx_lo = max(0, int(0.02 * (gs - 1)))
    idx_hi = min(gs - 1, int(0.98 * (gs - 1)))
    lo = xs[:, idx_lo]
    hi = xs[:, idx_hi]
    t = torch.linspace(0, 1, k, device=device, dtype=x.dtype)
    codebook = lo[:, None] + (hi - lo)[:, None] * t[None, :]  # (n, k)
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed) & 0xFFFFFFFF)
        codebook = codebook + torch.randn(codebook.shape, generator=g, device=device) * 1e-8

    prev = None
    for _ in range(max_iter):
        dist = (x[:, :, None] - codebook[:, None, :]).abs()  # (n, gs, k)
        labels = dist.argmin(dim=-1)  # (n, gs)
        new_cb = codebook.clone()
        for ci in range(k):
            mask = labels == ci  # (n, gs)
            counts = mask.sum(dim=1).clamp_min(1)
            summed = torch.where(mask, x, torch.zeros_like(x)).sum(dim=1)
            means = summed / counts
            empty = mask.sum(dim=1) == 0
            if empty.any():
                n_empty = int(empty.sum().item())
                rows = torch.nonzero(empty, as_tuple=False).squeeze(1)
                pick = torch.randint(0, gs, (n_empty,), device=device)
                means = means.clone()
                means[rows] = x[rows, pick]
            new_cb[:, ci] = means
        new_cb, _ = torch.sort(new_cb, dim=1)
        if prev is not None:
            if (new_cb - prev).abs().max().item() < tol:
                codebook = new_cb
                break
        prev = new_cb
        codebook = new_cb

    dist = (x[:, :, None] - codebook[:, None, :]).abs()
    indices = dist.argmin(dim=-1).to(torch.uint8)  # (n, gs)
    return (
        codebook.detach().cpu().numpy().astype(np.float32),
        scales.detach().cpu().numpy().astype(np.float32),
        norms.detach().cpu().numpy().astype(np.float32),
        indices.detach().cpu().numpy().T.astype(np.uint8),  # (gs, n)
    )
