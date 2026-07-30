"""Shared CLI for family quantize scripts."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np

from . import bundle, quant, hf_utils
from .errors import ConfigError, QuantError


def build_parser(default_bits: float = 4) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hadamard + codebook weight quantization")
    p.add_argument("--model", type=str, default=None, help="HF repo id")
    p.add_argument("--bits", type=float, default=None, help="1|2|3|4|2.54|3.26")
    p.add_argument("--group-size", type=int, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--tiny", action="store_true", help="synthetic checkpoint, no network")
    p.add_argument("--config", type=str, default=None, help="path to config.yaml")
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="CPU group workers (default: min(32, cpu_count); H100 box ~16)",
    )
    p.add_argument(
        "--codebook-share",
        choices=("group", "channel"),
        default=None,
        help="group=shared codebook per row-group (small, default); "
        "channel=per-channel codebook (larger, higher fidelity)",
    )
    return p


def read_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    try:
        import yaml
    except ImportError as e:
        raise ConfigError(f"pyyaml required to read config: {e}") from e
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError("config.yaml must be a mapping")
    return data


def _is_2d_shape(shape: tuple[int, ...]) -> bool:
    return len(shape) == 2 and shape[0] >= 1 and shape[1] >= 1


def _is_2d_weight(name: str, arr: np.ndarray) -> bool:
    return _is_2d_shape(arr.shape)


def _bit_map_for_names(
    names_2d: list[str],
    bits: float,
) -> dict[str, int]:
    bits = quant.parse_bits(bits)
    if bits in (2.54, 3.26):
        return quant.allocate_mixed_bits(names_2d, bits)
    b = int(bits)
    return {n: b for n in names_2d}


def quantize_state_dict(
    weights: dict[str, np.ndarray],
    bits: float,
    group_size: int = 32,
    seed: int | None = None,
    codebook_share: str = "group",
) -> dict[str, quant.QuantTensor | np.ndarray]:
    bits = quant.parse_bits(bits)
    names_2d = [n for n, w in weights.items() if _is_2d_weight(n, w)]
    bit_map = _bit_map_for_names(names_2d, bits)

    out: dict[str, quant.QuantTensor | np.ndarray] = {}
    for name, arr in weights.items():
        if name in bit_map:
            out[name] = quant.quantize_weight(
                arr,
                bits=bit_map[name],
                group_size=group_size,
                seed=seed,
                codebook_share=codebook_share,
            )
        else:
            out[name] = np.asarray(arr)
    return out


def _process_one(
    name: str,
    arr: np.ndarray,
    bit_map: dict[str, int],
    group_size: int,
    seed: int | None,
    workers: int | None = None,
    codebook_share: str = "group",
) -> quant.QuantTensor | np.ndarray:
    """Quantize one tensor. 2D weights always go through Hadamard + Lloyd-Max."""
    if name in bit_map:
        obj = quant.quantize_weight(
            arr,
            bits=bit_map[name],
            group_size=group_size,
            seed=seed,
            workers=workers,
            codebook_share=codebook_share,
        )
        if not obj.hadamard_meta.get("applied"):
            raise QuantError(
                f"{name}: Hadamard rotation was not applied (streaming invariant broken)"
            )
        if obj.codebook.size == 0 or obj.bits not in (1, 2, 3, 4):
            raise QuantError(f"{name}: Lloyd-Max codebook missing after quantize")
        return obj
    return np.asarray(arr)


def run_quantize(args: argparse.Namespace, family_dir: str, label: str) -> Path:
    from . import runtime

    cfg_path = args.config or str(Path(family_dir) / "config.yaml")
    cfg = read_config(cfg_path) if Path(cfg_path).is_file() else {}
    bits = args.bits if args.bits is not None else cfg.get("default_bits", 4)
    bits = quant.parse_bits(bits)
    group_size = args.group_size or cfg.get("group_size", 32)
    seed = args.seed if args.seed is not None else cfg.get("hadamard_seed", 0)
    workers = args.workers if getattr(args, "workers", None) is not None else runtime.default_workers()
    codebook_share = (
        args.codebook_share
        if getattr(args, "codebook_share", None) is not None
        else cfg.get("codebook_share", "group")
    )
    repo = args.model or cfg.get("base_model")
    qlabel = quant.quantization_label(bits)
    out = Path(args.out or str(Path(family_dir) / "weights" / qlabel.replace(".", "_")))

    print(f"[{label}] host: {runtime.runtime_summary()}", flush=True)
    print(f"[{label}] codebook_share={codebook_share}", flush=True)

    if args.tiny:
        weights = hf_utils.make_tiny_state_dict(seed=seed or 0)
        model_cfg = hf_utils.tiny_model_config()
        names_2d = [n for n, w in weights.items() if _is_2d_weight(n, w)]
        bit_map = _bit_map_for_names(names_2d, bits)
        print(f"[{label}] quantizing ({qlabel}) group_size={group_size} ...", flush=True)
        print(f"[{label}] writing bundle -> {out} ...", flush=True)
        with bundle.BundleWriter(
            out,
            model_cfg,
            qlabel,
            group_size_default=group_size,
            hadamard_seed=seed,
        ) as writer:
            for i, (name, arr) in enumerate(weights.items(), 1):
                obj = _process_one(
                    name, arr, bit_map, group_size, seed, workers=workers, codebook_share=codebook_share
                )
                writer.add(name, obj)
                del obj
                if i % 8 == 0:
                    gc.collect()
            writer.close()
        print(f"[{label}] done -> {out}", flush=True)
        return out

    if not repo:
        raise ConfigError("base_model / --model required when not using --tiny")

    print(f"[{label}] loading config from {repo} ...", flush=True)
    model_cfg = hf_utils.load_model_config(repo)

    print(f"[{label}] scanning tensor shapes (no weights yet) ...", flush=True)
    infos = hf_utils.load_model_info(repo)
    names_2d = [n for n, shape in infos if _is_2d_shape(shape)]
    bit_map = _bit_map_for_names(names_2d, bits)
    total = len(infos)
    print(
        f"[{label}] {total} tensors, {len(names_2d)} 2D to quantize ({qlabel}) "
        f"group_size={group_size} workers={workers}",
        flush=True,
    )
    print(f"[{label}] streaming + quantizing + writing -> {out} ...", flush=True)

    with bundle.BundleWriter(
        out,
        model_cfg,
        qlabel,
        group_size_default=group_size,
        hadamard_seed=seed,
    ) as writer:
        for i, (name, arr) in enumerate(hf_utils.stream_weights(repo), 1):
            shape = tuple(arr.shape)
            print(f"[{label}] [{i}/{total}] {name} {shape}", flush=True)
            obj = _process_one(
                    name, arr, bit_map, group_size, seed, workers=workers, codebook_share=codebook_share
                )
            del arr
            writer.add(name, obj)
            del obj
            gc.collect()
        writer.close()

    try:
        hf_utils.copy_tokenizer(repo, out)
    except Exception as e:
        print(f"[{label}] tokenizer copy skipped: {e}", flush=True)

    print(f"[{label}] done -> {out}", flush=True)
    return out
