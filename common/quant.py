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

# Compute-sensitive for q1.5 (excludes large embedding / PLE tables).
HI_SUBSTR = (
    "lm_head",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "attn_o",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
)

PLE_NAME_SUBSTR = (
    "per_layer",
    "per-layer",
    "embed",
    "embd",
    "embedding",
)

VALID_BIT_VALUES = {1, 2, 3, 4, 8, 1.5, 2.54, 3.26}
INTEGER_BITS = (1, 2, 3, 4, 8)
Q15_BAND = (1.35, 1.55)
PLE_NUMEL_FLOOR = 50_000_000
PLE_ROW_FLOOR = 32_000


def parse_bits(value: Any) -> float:
    try:
        bits = float(value)
    except (TypeError, ValueError) as e:
        raise QuantError(f"invalid bits value: {value!r}") from e
    if bits in (1.0, 2.0, 3.0, 4.0, 8.0):
        bits = float(int(bits))
    if bits not in VALID_BIT_VALUES:
        raise QuantError(f"bits must be one of {sorted(VALID_BIT_VALUES)}, got {bits}")
    return bits


def quantization_label(bits: float) -> str:
    bits = parse_bits(bits)
    if bits in (1.0, 2.0, 3.0, 4.0, 8.0):
        return f"q{int(bits)}"
    if bits == 1.5:
        return "q1.5"
    if bits == 2.54:
        return "q2.54"
    if bits == 3.26:
        return "q3.26"
    raise QuantError(f"unhandled bits {bits}")


def _is_sensitive(name: str) -> bool:
    low = name.lower()
    return any(s in low for s in SENSITIVE_SUBSTR)


def _is_ple_name(name: str) -> bool:
    low = name.lower()
    if "lm_head" in low:
        return False
    if any(s in low for s in PLE_NAME_SUBSTR):
        return True
    if low.startswith("ple_") or "_ple_" in low or ".ple." in low or low.endswith("_ple"):
        return True
    return False


def _is_hi_name(name: str) -> bool:
    low = name.lower()
    if any(s in low for s in HI_SUBSTR):
        return True
    if low.endswith("output.weight") or ".output.weight" in low:
        return True
    return False


def classify_tensor_role(
    name: str,
    shape: tuple[int, ...] | None = None,
    numel: int | None = None,
) -> str:
    """Return ``ple`` | ``hi`` | ``compute`` for q1.5 policy."""
    if numel is None and shape is not None:
        n = 1
        for d in shape:
            n *= int(d)
        numel = n
    if _is_ple_name(name):
        return "ple"
    # Size heuristic only with a vocab-like row dim (avoids tagging large FFN as PLE).
    if shape is not None and len(shape) == 2:
        rows = int(shape[0])
        n = int(numel) if numel is not None else rows * int(shape[1])
        if n >= PLE_NUMEL_FLOOR and rows >= PLE_ROW_FLOOR and not _is_hi_name(name):
            return "ple"
    if _is_hi_name(name):
        return "hi"
    return "compute"


def weighted_avg_bits(assign: dict[str, int], numels: dict[str, int]) -> float:
    total_n = 0
    total_b = 0.0
    for name, bits in assign.items():
        n = int(numels.get(name, 1))
        if n < 0:
            raise QuantError(f"numel must be >=0 for {name}")
        total_n += n
        total_b += float(bits) * n
    if total_n <= 0:
        raise QuantError("weighted_avg_bits: total numel is 0")
    return total_b / total_n


def estimate_index_bytes(assign: dict[str, int], numels: dict[str, int]) -> int:
    """Packed index payload size (no codebooks) for a bit map."""
    total = 0
    for name, bits in assign.items():
        total += pack.packed_size(int(numels.get(name, 0)), int(bits))
    return total


def _validate_q15_overrides(ple_bits: int, compute_bits: int, hi_bits: int) -> None:
    if ple_bits not in (1, 2):
        raise QuantError(f"ple_bits must be 1 or 2, got {ple_bits}")
    if compute_bits not in (1, 2, 3):
        raise QuantError(f"compute_bits must be 1, 2, or 3, got {compute_bits}")
    if hi_bits not in (2, 3, 4):
        raise QuantError(f"hi_bits must be 2, 3, or 4, got {hi_bits}")
    if not (ple_bits <= compute_bits <= hi_bits):
        raise QuantError(
            f"require ple_bits <= compute_bits <= hi_bits, got "
            f"{ple_bits} <= {compute_bits} <= {hi_bits}"
        )


def allocate_mixed_bits_weighted(
    layers: list[tuple[str, int]],
    target: float = 1.5,
    *,
    ple_bits: int = 1,
    compute_bits: int = 2,
    hi_bits: int = 3,
    shapes: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, int]:
    """PLE-default-1 + param-weighted mix for ``--bits 1.5``.

    ``layers`` is a list of ``(name, numel)``. Optional ``shapes`` improves PLE detection.
    Never raises PLE above ``ple_bits`` while adjusting the average into ``Q15_BAND``.
    """
    target = parse_bits(target)
    if target != 1.5:
        raise QuantError(f"allocate_mixed_bits_weighted expects 1.5, got {target}")
    _validate_q15_overrides(ple_bits, compute_bits, hi_bits)

    if not layers:
        return {}

    numels = {name: int(numel) for name, numel in layers}
    for name, n in numels.items():
        if n < 0:
            raise QuantError(f"numel must be >=0 for {name}")

    roles: dict[str, str] = {}
    assign: dict[str, int] = {}
    for name, numel in numels.items():
        shape = None if shapes is None else shapes.get(name)
        role = classify_tensor_role(name, shape=shape, numel=numel)
        roles[name] = role
        if role == "ple":
            assign[name] = int(ple_bits)
        elif role == "hi":
            assign[name] = int(hi_bits)
        else:
            assign[name] = int(compute_bits)

    band_lo, band_hi = Q15_BAND
    max_steps = max(8, 4 * len(assign))

    def avg() -> float:
        return weighted_avg_bits(assign, numels)

    def demote_once() -> bool:
        # Prefer demoting hi toward compute_bits, then other non-PLE toward ple_bits.
        hi_cand = [
            n
            for n, r in roles.items()
            if r == "hi" and assign[n] > compute_bits
        ]
        hi_cand.sort(key=lambda n: (-numels[n], n))
        if hi_cand:
            assign[hi_cand[0]] -= 1
            return True
        other = [
            n
            for n, r in roles.items()
            if r != "ple" and assign[n] > ple_bits
        ]
        other.sort(key=lambda n: (-numels[n], n))
        if other:
            assign[other[0]] -= 1
            return True
        return False

    def promote_once() -> bool:
        # Prefer promoting hi toward hi_bits, then compute; never touch PLE.
        hi_cand = [
            n
            for n, r in roles.items()
            if r == "hi" and assign[n] < hi_bits
        ]
        hi_cand.sort(key=lambda n: (-numels[n], n))
        if hi_cand:
            assign[hi_cand[0]] += 1
            return True
        other = [
            n
            for n, r in roles.items()
            if r == "compute" and assign[n] < hi_bits
        ]
        other.sort(key=lambda n: (-numels[n], n))
        if other:
            assign[other[0]] += 1
            return True
        return False

    for _ in range(max_steps):
        a = avg()
        if band_lo <= a <= band_hi:
            break
        if a > band_hi:
            if not demote_once():
                break
        elif not promote_once():
            break

    a = avg()
    if len(assign) >= 2 and not (band_lo <= a <= band_hi):
        raise QuantError(
            f"cannot meet q1.5 weighted bit band {band_lo}-{band_hi}, got {a:.4f}"
        )
    # Lock: PLE must stay at ple_bits (never raised by adjustment).
    for name, role in roles.items():
        if role == "ple" and assign[name] != ple_bits:
            raise QuantError(
                f"internal: PLE layer {name} has bits {assign[name]} != ple_bits {ple_bits}"
            )
    return assign


def bit_policy_meta(
    assign: dict[str, int],
    numels: dict[str, int],
    *,
    ple_bits: int = 1,
    compute_bits: int = 2,
    hi_bits: int = 3,
) -> dict[str, Any]:
    return {
        "bit_policy": "ple_weighted",
        "avg_bits_weighted": round(weighted_avg_bits(assign, numels), 6),
        "ple_bits": int(ple_bits),
        "compute_bits": int(compute_bits),
        "hi_bits": int(hi_bits),
        "estimate_index_bytes": estimate_index_bytes(assign, numels),
    }


def allocate_mixed_bits(layer_names: list[str], target: float) -> dict[str, int]:
    """Assign per-layer integer bits for mixed-precision targets 2.54 / 3.26."""
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
    codebook: np.ndarray  # fp16 (G, Kc) or (G, N, Kc)
    input_scale: np.ndarray
    input_scale_recip: np.ndarray
    norms: np.ndarray
    hadamard_meta: dict = field(default_factory=dict)
    row_pad: int = 0
    codebook_share: str = "group"  # "group" | "channel"


def quantize_weight(
    W: np.ndarray,
    bits: int,
    group_size: int = 32,
    seed: int | None = None,
    max_iter: int = 50,
    workers: int | None = None,
    codebook_share: str = "group",
) -> QuantTensor:
    """Hadamard (axis=0) + Lloyd-Max codebook.

    ``codebook_share=group`` (default): one codebook per row-group — small on disk.
    ``codebook_share=channel``: per-(group, channel) — higher fidelity, ~3× larger.
    """
    if bits not in INTEGER_BITS:
        raise QuantError(f"quantize_weight bits must be one of {INTEGER_BITS}, got {bits}")
    if group_size < 1:
        raise QuantError(f"group_size must be >=1, got {group_size}")
    if codebook_share not in ("group", "channel"):
        raise QuantError(f"codebook_share must be 'group' or 'channel', got {codebook_share!r}")

    from . import runtime

    W = np.asarray(W, dtype=np.float32)
    if W.ndim != 2:
        raise QuantError(f"expected 2D weight, got {W.shape}")
    if not np.isfinite(W).all():
        raise QuantError("weight contains non-finite values")

    k0, n = W.shape
    W_rot, hmeta = hadamard.hadamard_rotate(W, axis=0, seed=seed)
    if not hmeta.get("applied"):
        raise QuantError(f"Hadamard rotation failed to apply for shape {(k0, n)}: {hmeta}")
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
    all_idx = np.zeros((num_groups, group_size, n), dtype=np.uint8)
    rng_seed = 0 if seed is None else int(seed)
    empty = np.zeros((0,), dtype=np.float16)

    if codebook_share == "group":
        codebooks = np.zeros((num_groups, kc), dtype=np.float32)
        length = group_size * n
        # Batched GPU Lloyd-Max when CUDA is up and groups are large enough to amortize H2D.
        use_cuda_group = (
            runtime.cuda_available()
            and length >= 256
            and num_groups >= 1
        )
        if use_cuda_group:
            # Cap batch by GPU VRAM-aware distance buffer B*L*K (float32 elems).
            max_batch = max(
                1,
                min(num_groups, runtime.cuda_batch_elem_budget() // max(length * kc, 1)),
            )
            for start in range(0, num_groups, max_batch):
                end = min(num_groups, start + max_batch)
                bsz = end - start
                batch = np.empty((bsz, length), dtype=np.float32)
                for i, g in enumerate(range(start, end)):
                    sl = slice(g * group_size, (g + 1) * group_size)
                    batch[i] = W_work[sl, :].ravel()
                cbs, idx = cb.lloyd_max_batched_torch(
                    batch,
                    kc,
                    max_iter=max_iter,
                    seed=rng_seed + start * 10007,
                    device="cuda",
                )
                if cbs.shape != (bsz, kc) or idx.shape != (bsz, length):
                    raise QuantError(
                        f"batched Lloyd-Max shape mismatch: codebook {cbs.shape} "
                        f"indices {idx.shape} (expected B={bsz}, L={length}, K={kc})"
                    )
                codebooks[start:end] = cbs
                all_idx[start:end] = idx.reshape(bsz, group_size, n)
        else:
            n_workers = workers if workers is not None else runtime.default_workers()
            parallel = n_workers > 1 and num_groups > 2 and length >= 1024

            def _one_group_cpu(g: int) -> tuple[int, np.ndarray, np.ndarray]:
                sl = slice(g * group_size, (g + 1) * group_size)
                flat = W_work[sl, :].ravel()
                cb_vec = cb.lloyd_max(
                    flat, kc, max_iter=max_iter, seed=rng_seed + g * 10007
                )
                idx = cb.quantize_group(flat, cb_vec).reshape(group_size, n)
                return g, cb_vec.astype(np.float32), idx

            if not parallel:
                for g in range(num_groups):
                    g_i, cbs, idx = _one_group_cpu(g)
                    codebooks[g_i] = cbs
                    all_idx[g_i] = idx
            else:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    for g_i, cbs, idx in pool.map(_one_group_cpu, range(num_groups)):
                        codebooks[g_i] = cbs
                        all_idx[g_i] = idx
        del W_work
        return QuantTensor(
            bits=bits,
            group_size=group_size,
            shape=(k0, n),
            packed_indices=pack.pack_indices(all_idx.reshape(-1), bits),
            codebook=codebooks.astype(np.float16),
            input_scale=empty,
            input_scale_recip=empty,
            norms=empty,
            hadamard_meta=hmeta,
            row_pad=gpad,
            codebook_share="group",
        )

    codebooks = np.zeros((num_groups, n, kc), dtype=np.float32)
    use_cuda = runtime.cuda_available() and n >= 64 and group_size * n * kc < (1 << 28)
    n_workers = workers if workers is not None else runtime.default_workers()
    parallel = (not use_cuda) and n_workers > 1 and num_groups > 2 and (num_groups * n) >= 8192

    def _one_group(g: int) -> tuple[int, np.ndarray, np.ndarray]:
        sl = slice(g * group_size, (g + 1) * group_size)
        block = W_work[sl, :]
        if use_cuda:
            cbs, _sc, _nm, idx = cb.lloyd_max_columns_torch(
                block, kc, max_iter=max_iter, seed=rng_seed + g * 10007
            )
        else:
            cbs, _sc, _nm, idx = cb.lloyd_max_columns(
                block, kc, max_iter=max_iter, seed=rng_seed + g * 10007
            )
        if cbs.shape != (n, kc) or idx.shape != (group_size, n):
            raise QuantError(
                f"Lloyd-Max output shape mismatch: codebook {cbs.shape} indices {idx.shape}"
            )
        return g, cbs, idx

    if not parallel:
        for g in range(num_groups):
            g_i, cbs, idx = _one_group(g)
            codebooks[g_i] = cbs
            all_idx[g_i] = idx
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for g_i, cbs, idx in pool.map(_one_group, range(num_groups)):
                codebooks[g_i] = cbs
                all_idx[g_i] = idx
    del W_work

    return QuantTensor(
        bits=bits,
        group_size=group_size,
        shape=(k0, n),
        packed_indices=pack.pack_indices(all_idx.reshape(-1), bits),
        codebook=codebooks.astype(np.float16),
        input_scale=empty,
        input_scale_recip=empty,
        norms=empty,
        hadamard_meta=hmeta,
        row_pad=gpad,
        codebook_share="channel",
    )


def dequantize(t: QuantTensor) -> np.ndarray:
    """Reconstruct weight in rotated space, shape = original (K, N)."""
    k0, n = t.shape
    gs = t.group_size
    cb_arr = t.codebook.astype(np.float32)
    share = getattr(t, "codebook_share", None) or (
        "group" if cb_arr.ndim == 2 else "channel"
    )
    if share == "group":
        if cb_arr.ndim != 2:
            raise ShapeMismatchError(f"group codebook must be 2D, got {cb_arr.shape}")
        num_groups = cb_arr.shape[0]
    else:
        if cb_arr.ndim != 3:
            raise ShapeMismatchError(f"channel codebook must be 3D, got {cb_arr.shape}")
        num_groups, n_cb, kc = cb_arr.shape
        if n_cb != n or kc != (1 << t.bits):
            raise ShapeMismatchError(f"codebook shape {cb_arr.shape} incompatible with N={n}")

    k_work = num_groups * gs
    expected = k_work * n
    indices = pack.unpack_indices(t.packed_indices, expected, t.bits)
    if indices.size != expected:
        raise ShapeMismatchError(
            f"unpacked {indices.size} indices, expected {expected} for shape work ({k_work},{n})"
        )
    idx_mat = indices.reshape(k_work, n)
    out = np.zeros((k_work, n), dtype=np.float32)

    if share == "group":
        for g in range(num_groups):
            rows = slice(g * gs, (g + 1) * gs)
            out[rows, :] = cb_arr[g, idx_mat[rows, :]]
    else:
        for g in range(num_groups):
            rows = slice(g * gs, (g + 1) * gs)
            for j in range(n):
                out[rows, j] = cb_arr[g, j, idx_mat[rows, j]]
    return out[:k0, :]


def dequantize_torch(t: QuantTensor, *, device: str | object = "cuda"):
    """Torch codebook gather → rotated-space ``(K, N)`` on ``device``."""
    import torch

    k0, n = t.shape
    gs = t.group_size
    cb_arr = t.codebook.astype(np.float32)
    share = getattr(t, "codebook_share", None) or (
        "group" if cb_arr.ndim == 2 else "channel"
    )
    if share == "group":
        if cb_arr.ndim != 2:
            raise ShapeMismatchError(f"group codebook must be 2D, got {cb_arr.shape}")
        num_groups = int(cb_arr.shape[0])
    else:
        if cb_arr.ndim != 3:
            raise ShapeMismatchError(f"channel codebook must be 3D, got {cb_arr.shape}")
        num_groups, n_cb, kc = (int(x) for x in cb_arr.shape)
        if n_cb != n or kc != (1 << t.bits):
            raise ShapeMismatchError(f"codebook shape {cb_arr.shape} incompatible with N={n}")

    k_work = num_groups * gs
    expected = k_work * n
    indices = pack.unpack_indices(t.packed_indices, expected, t.bits)
    if indices.size != expected:
        raise ShapeMismatchError(
            f"unpacked {indices.size} indices, expected {expected} for shape work ({k_work},{n})"
        )
    dev = torch.device(device)
    idx = torch.as_tensor(indices.reshape(k_work, n), device=dev, dtype=torch.long)
    cb = torch.as_tensor(cb_arr, device=dev, dtype=torch.float32)
    if share == "group":
        g_ids = torch.arange(k_work, device=dev) // gs
        out = cb[g_ids[:, None].expand_as(idx), idx]
    else:
        out = torch.empty((k_work, n), device=dev, dtype=torch.float32)
        for g in range(num_groups):
            rows = slice(g * gs, (g + 1) * gs)
            # out[r, j] = cb[g, j, idx[r, j]]
            jj = torch.arange(n, device=dev)[None, :].expand(gs, n)
            out[rows] = cb[g, jj, idx[rows]]
    return out[:k0]


def reconstruct_weight(t: QuantTensor, seed: int | None = None) -> np.ndarray:
    """Dequant rotated codebook then blocked unrotate → original-space ``(K, N)``."""
    from . import hadamard

    recon_rot = dequantize(t)
    if seed is None:
        seed = t.hadamard_meta.get("seed")
    out, meta = hadamard.hadamard_unrotate(recon_rot, axis=0, seed=seed)
    if not meta.get("applied"):
        raise QuantError("Hadamard unrotate failed during reconstruct_weight")
    return out


def reconstruct_weight_torch(
    t: QuantTensor,
    seed: int | None = None,
    *,
    device: str | object = "cuda",
):
    """CUDA/CPU-torch reconstruct (same math as :func:`reconstruct_weight`).

    Returns a float32 tensor on ``device``. Prefer this in diag inject when the
    HF model already lives on GPU to avoid host FWHT.
    """
    from . import hadamard

    if seed is None:
        seed = t.hadamard_meta.get("seed")
    recon_rot = dequantize_torch(t, device=device)
    out, meta = hadamard.hadamard_unrotate_torch(recon_rot, seed=seed, device=device)
    if not meta.get("applied"):
        raise QuantError("Hadamard unrotate failed during reconstruct_weight_torch")
    return out