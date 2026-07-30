"""Hadamard + Lloyd-Max codebook weight quantization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import codebook as cb
from . import pack
from . import hadamard
from .errors import QuantError, ShapeMismatchError

SENSITIVE_SUBSTR = (
    "embed",
    "embd",
    "lm_head",
    "output",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)

VALID_BIT_VALUES = {1, 2, 3, 4, 2.54, 3.26}


def parse_bits(value: Any) -> float:
    try:
        bits = float(value)
    except (TypeError, ValueError) as e:
        raise QuantError(f"invalid bits value: {value!r}") from e
    if bits in (1.0, 2.0, 3.0, 4.0):
        bits = float(int(bits))
    if bits not in VALID_BIT_VALUES:
        raise QuantError(f"bits must be one of {sorted(VALID_BIT_VALUES)}, got {bits}")
    return bits


def quantization_label(bits: float) -> str:
    bits = parse_bits(bits)
    if bits in (1.0, 2.0, 3.0, 4.0):
        return f"q{int(bits)}"
    if bits == 2.54:
        return "q2.54"
    if bits == 3.26:
        return "q3.26"
    raise QuantError(f"unhandled bits {bits}")


def _is_sensitive(name: str) -> bool:
    low = name.lower()
    return any(s in low for s in SENSITIVE_SUBSTR)


def allocate_mixed_bits(layer_names: list[str], target: float) -> dict[str, int]:
    """Assign per-layer integer bits for mixed-precision targets 2.54 / 3.26.

    Sensitive layers are preferred for the higher bit budget; counts are chosen so
    the mean bit-width lands in the Spec bands ([2.45,2.65] / [3.15,3.40]).
    """
    target = parse_bits(target)
    if target not in (2.54, 3.26):
        raise QuantError(f"allocate_mixed_bits expects 2.54 or 3.26, got {target}")

    names = list(layer_names)
    if not names:
        return {}

    sensitive = [n for n in names if _is_sensitive(n)]
    others = sorted(n for n in names if not _is_sensitive(n))
    ordered = sensitive + others
    n = len(ordered)

    if target == 2.54:
        n_hi = int(round(0.54 * n))
        n_hi = min(max(n_hi, 0), n)
        lo, hi = 2, 3
        band = (2.45, 2.65)
    else:
        n_hi = int(round(0.26 * n))
        n_hi = min(max(n_hi, 0), n)
        lo, hi = 3, 4
        band = (3.15, 3.40)

    assign = {name: (hi if i < n_hi else lo) for i, name in enumerate(ordered)}

    def avg() -> float:
        return sum(assign.values()) / len(assign)

    guard = 0
    while not (band[0] <= avg() <= band[1]) and guard < n + 2:
        a = avg()
        if a < band[0] and n_hi < n:
            n_hi += 1
        elif a > band[1] and n_hi > 0:
            n_hi -= 1
        else:
            break
        assign = {name: (hi if i < n_hi else lo) for i, name in enumerate(ordered)}
        guard += 1

    a = avg()
    if n >= 4 and not (band[0] <= a <= band[1]):
        raise QuantError(f"cannot meet {target} average bit band, got {a:.4f}")
    return assign


@dataclass
class QuantTensor:
    bits: int
    group_size: int
    shape: tuple[int, int]
    packed_indices: bytes
    codebook: np.ndarray  # fp16 (G, N, Kc)
    input_scale: np.ndarray  # fp16 (G, N)
    input_scale_recip: np.ndarray  # fp16 (G, N)
    norms: np.ndarray  # fp16 (G, N)
    hadamard_meta: dict = field(default_factory=dict)
    row_pad: int = 0


def quantize_weight(
    W: np.ndarray,
    bits: int,
    group_size: int = 32,
    seed: int | None = None,
    max_iter: int = 50,
    workers: int | None = None,
) -> QuantTensor:
    """Core path: Hadamard rotate (axis=0) then per-group Lloyd-Max codebook.

    Streaming callers must use this function for every 2D weight so algorithm
    semantics stay identical to an in-memory full-model pass.
    """
    if bits not in (1, 2, 3, 4):
        raise QuantError(f"quantize_weight bits must be 1..4, got {bits}")
    if group_size < 1:
        raise QuantError(f"group_size must be >=1, got {group_size}")

    from . import runtime

    W = np.asarray(W, dtype=np.float32)
    if W.ndim != 2:
        raise QuantError(f"expected 2D weight, got {W.shape}")
    if not np.isfinite(W).all():
        raise QuantError("weight contains non-finite values")

    k0, n = W.shape
    # Core feature 1: Hadamard rotation (never skipped; chunked if needed).
    W_rot, hmeta = hadamard.hadamard_rotate(W, axis=0, seed=seed)
    if not hmeta.get("applied"):
        raise QuantError(
            f"Hadamard rotation failed to apply for shape {(k0, n)}: {hmeta}"
        )
    del W
    k = W_rot.shape[0]

    gpad = (-k) % group_size
    if gpad:
        W_work = np.zeros((k + gpad, n), dtype=np.float32)
        W_work[:k, :] = W_rot
        del W_rot
    else:
        W_work = W_rot
    k_work = W_work.shape[0]
    if k_work % group_size != 0:
        raise QuantError(f"internal: rows {k_work} not divisible by group_size {group_size}")

    num_groups = k_work // group_size
    kc = 1 << bits
    codebooks = np.zeros((num_groups, n, kc), dtype=np.float32)
    scales = np.zeros((num_groups, n), dtype=np.float32)
    norms = np.zeros((num_groups, n), dtype=np.float32)
    all_idx = np.zeros((num_groups, group_size, n), dtype=np.uint8)

    rng_seed = 0 if seed is None else int(seed)
    use_cuda = runtime.cuda_available() and n >= 64 and group_size * n * kc < (1 << 28)
    n_workers = workers if workers is not None else runtime.default_workers()
    parallel = (not use_cuda) and n_workers > 1 and num_groups > 2 and (num_groups * n) >= 8192

    def _one_group(g: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Core feature 2: Lloyd-Max codebook per (group, channel).
        sl = slice(g * group_size, (g + 1) * group_size)
        block = W_work[sl, :]
        if use_cuda:
            cbs, sc, nm, idx = cb.lloyd_max_columns_torch(
                block, kc, max_iter=max_iter, seed=rng_seed + g * 10007
            )
        else:
            cbs, sc, nm, idx = cb.lloyd_max_columns(
                block, kc, max_iter=max_iter, seed=rng_seed + g * 10007
            )
        if cbs.shape != (n, kc) or idx.shape != (group_size, n):
            raise QuantError(
                f"Lloyd-Max output shape mismatch: codebook {cbs.shape} indices {idx.shape}"
            )
        return g, cbs, sc, nm, idx

    if not parallel:
        for g in range(num_groups):
            g_i, cbs, sc, nm, idx = _one_group(g)
            codebooks[g_i] = cbs
            scales[g_i] = sc
            norms[g_i] = nm
            all_idx[g_i] = idx
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for g_i, cbs, sc, nm, idx in pool.map(_one_group, range(num_groups)):
                codebooks[g_i] = cbs
                scales[g_i] = sc
                norms[g_i] = nm
                all_idx[g_i] = idx
    del W_work

    indices_mat = all_idx.reshape(num_groups * group_size, n)
    indices_flat = indices_mat.ravel(order="C")
    packed = pack.pack_indices(indices_flat, bits)

    recip = np.where(scales > 0, 1.0 / scales, 0.0)
    return QuantTensor(
        bits=bits,
        group_size=group_size,
        shape=(k0, n),
        packed_indices=packed,
        codebook=codebooks.astype(np.float16),
        input_scale=scales.astype(np.float16),
        input_scale_recip=recip.astype(np.float16),
        norms=norms.astype(np.float16),
        hadamard_meta=hmeta,
        row_pad=int(hmeta.get("row_pad", 0)) + gpad,
    )


def dequantize(t: QuantTensor) -> np.ndarray:
    """Reconstruct weight in rotated space, shape = original (K, N)."""
    k0, n = t.shape
    gs = t.group_size
    num_groups = t.codebook.shape[0]
    k_work = num_groups * gs
    expected = k_work * n
    indices = pack.unpack_indices(t.packed_indices, expected, t.bits)
    if indices.size != expected:
        raise ShapeMismatchError(
            f"unpacked {indices.size} indices, expected {expected} for shape work ({k_work},{n})"
        )
    idx_mat = indices.reshape(k_work, n)
    out = np.zeros((k_work, n), dtype=np.float32)
    kc = 1 << t.bits
    cb_arr = t.codebook.astype(np.float32)
    if cb_arr.shape != (num_groups, n, kc):
        raise ShapeMismatchError(f"codebook shape {cb_arr.shape} != {(num_groups, n, kc)}")

    for g in range(num_groups):
        rows = slice(g * gs, (g + 1) * gs)
        for j in range(n):
            out[rows, j] = cb_arr[g, j, idx_mat[rows, j]]
    return out[:k0, :]
