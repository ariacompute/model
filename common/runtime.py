"""Host resource helpers for large-machine quantization (e.g. H200 / RTX PRO 6000)."""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def total_ram_bytes() -> int:
    env = os.environ.get("ARIA_QUANT_RAM_BYTES")
    if env:
        return int(env)
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page = os.sysconf("SC_PAGE_SIZE")
        if isinstance(pages, int) and isinstance(page, int) and pages > 0 and page > 0:
            return int(pages) * int(page)
    except (ValueError, OSError, AttributeError):
        pass
    return 16 * (1 << 30)  # conservative fallback 16 GiB


@lru_cache(maxsize=1)
def max_work_elems() -> int:
    """Max float32 elements for one Hadamard work buffer.

    Uses ``ARIA_QUANT_MAX_ELEMS`` if set; else ~25% of detected RAM
    (capped), floored at 128 Mi elements (~512 MiB).
    """
    env = os.environ.get("ARIA_QUANT_MAX_ELEMS")
    if env:
        return max(1, int(env))
    budget = int(total_ram_bytes() * 0.25 / 4)  # float32
    return max(128 * (1 << 20), min(budget, 32 * (1 << 30)))  # cap ~128 GiB elems


@lru_cache(maxsize=1)
def default_workers() -> int:
    env = os.environ.get("ARIA_QUANT_WORKERS")
    if env:
        return max(1, int(env))
    cpu = os.cpu_count() or 4
    return max(1, min(cpu, 32))


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    if os.environ.get("ARIA_QUANT_FORCE_CPU", "").lower() in ("1", "true", "yes"):
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@lru_cache(maxsize=1)
def cuda_mem_bytes() -> int:
    """Total CUDA device 0 memory in bytes, or 0 if unavailable."""
    if not cuda_available():
        return 0
    try:
        import torch

        return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        return 0


@lru_cache(maxsize=1)
def cuda_batch_elem_budget() -> int:
    """Max float32 elements for one GPU Lloyd distance buffer (B×L×K).

    Uses ``ARIA_QUANT_CUDA_BATCH_ELEMS`` if set; else ~1/16 of device VRAM
    (floored at 64 Mi elems, capped), suited to large GPUs such as H200 141 GiB
    or RTX PRO 6000 96 GiB.
    """
    env = os.environ.get("ARIA_QUANT_CUDA_BATCH_ELEMS")
    if env:
        return max(1, int(env))
    mem = cuda_mem_bytes()
    if mem <= 0:
        return 1 << 26
    # float32 → /4; use ~1/16 of VRAM for the dominant dist tensor.
    budget = int(mem / 16 / 4)
    return max(64 * (1 << 20), min(budget, 1 << 30))


def runtime_summary() -> str:
    ram_gib = total_ram_bytes() / (1 << 30)
    parts = [
        f"ram={ram_gib:.1f}GiB",
        f"workers={default_workers()}",
        f"max_work_elems={max_work_elems()}",
        f"cuda={cuda_available()}",
    ]
    if cuda_available():
        try:
            import torch

            name = torch.cuda.get_device_name(0).replace(" ", "_")
            vram_gib = cuda_mem_bytes() / (1 << 30)
            parts.append(f"gpu={name}")
            parts.append(f"vram={vram_gib:.0f}GiB")
            parts.append(f"cuda_batch_elems={cuda_batch_elem_budget()}")
        except Exception:
            pass
    return " ".join(parts)
