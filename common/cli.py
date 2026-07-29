"""Shared CLI for family quantize scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import bundle, quant, hf_utils
from .quant import QuantTensor
from .errors import ConfigError


def build_parser(default_bits: float = 4) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hadamard + codebook weight quantization")
    p.add_argument("--model", type=str, default=None, help="HF repo id")
    p.add_argument("--bits", type=float, default=None, help="1|2|3|4|2.54|3.26")
    p.add_argument("--group-size", type=int, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--tiny", action="store_true", help="synthetic checkpoint, no network")
    p.add_argument("--config", type=str, default=None, help="path to config.yaml")
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


def _is_2d_weight(name: str, arr: np.ndarray) -> bool:
    return arr.ndim == 2 and arr.shape[0] >= 1 and arr.shape[1] >= 1


def quantize_state_dict(
    weights: dict[str, np.ndarray],
    bits: float,
    group_size: int = 32,
    seed: int | None = None,
) -> dict[str, QuantTensor | np.ndarray]:
    bits = quant.parse_bits(bits)
    names_2d = [n for n, w in weights.items() if _is_2d_weight(n, w)]
    if bits in (2.54, 3.26):
        bit_map = quant.allocate_mixed_bits(names_2d, bits)
    else:
        b = int(bits)
        bit_map = {n: b for n in names_2d}

    out: dict[str, QuantTensor | np.ndarray] = {}
    for name, arr in weights.items():
        if name in bit_map:
            out[name] = quant.quantize_weight(
                arr, bits=bit_map[name], group_size=group_size, seed=seed
            )
        else:
            out[name] = np.asarray(arr)
    return out


def run_quantize(args: argparse.Namespace, family_dir: str, label: str) -> Path:
    cfg_path = args.config or str(Path(family_dir) / "config.yaml")
    cfg = read_config(cfg_path) if Path(cfg_path).is_file() else {}
    bits = args.bits if args.bits is not None else cfg.get("default_bits", 4)
    bits = quant.parse_bits(bits)
    group_size = args.group_size or cfg.get("group_size", 32)
    seed = args.seed if args.seed is not None else cfg.get("hadamard_seed", 0)
    repo = args.model or cfg.get("base_model")
    qlabel = quant.quantization_label(bits)

    if args.tiny:
        weights = hf_utils.make_tiny_state_dict(seed=seed or 0)
        model_cfg = hf_utils.tiny_model_config()
        tok_src = None
    else:
        if not repo:
            raise ConfigError("base_model / --model required when not using --tiny")
        print(f"[{label}] loading config from {repo} ...", flush=True)
        model_cfg = hf_utils.load_model_config(repo)
        print(f"[{label}] streaming weights ...", flush=True)
        weights = {name: arr for name, arr in hf_utils.stream_weights(repo)}
        tok_src = repo

    print(f"[{label}] quantizing ({qlabel}) group_size={group_size} ...", flush=True)
    q_tensors = quantize_state_dict(weights, bits=bits, group_size=group_size, seed=seed)

    out = args.out or str(Path(family_dir) / "weights" / qlabel.replace(".", "_"))
    print(f"[{label}] writing bundle -> {out} ...", flush=True)
    bundle.write_bundle(
        out,
        q_tensors,
        model_cfg,
        quantization=qlabel,
        tokenizer_src=None if args.tiny else None,
        group_size_default=group_size,
        hadamard_seed=seed,
    )
    if not args.tiny and tok_src:
        try:
            hf_utils.copy_tokenizer(tok_src, out)
        except Exception as e:
            print(f"[{label}] tokenizer copy skipped: {e}", flush=True)
    print(f"[{label}] done -> {out}", flush=True)
    return Path(out)
