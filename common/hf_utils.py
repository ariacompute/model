"""HuggingFace helpers + synthetic tiny checkpoint for the quant pipeline."""

from __future__ import annotations

import json
import mmap
import struct
import shutil
from pathlib import Path
from typing import Generator

import numpy as np

from .errors import ModelFetchError, ConfigError, UnsupportedError
from .bundle import TOKENIZER_FILES


def _bf16_raw_to_f32(raw: bytes | memoryview, shape: tuple[int, ...]) -> np.ndarray:
    """Reinterpret IEEE bfloat16 bytes as float32 (shift mantissa/exponent)."""
    u16 = np.frombuffer(raw, dtype=np.uint16)
    expected = 1
    for d in shape:
        expected *= int(d)
    if u16.size != expected:
        raise UnsupportedError(f"BF16 size mismatch: got {u16.size} for shape {shape}")
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape).copy()


def _safetensors_bytes_to_f32(
    raw: bytes | memoryview, dtype: str, shape: tuple[int, ...]
) -> np.ndarray:
    """Decode one safetensors payload to contiguous float32."""
    if dtype == "BF16":
        return _bf16_raw_to_f32(raw, shape)
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
    if dtype == "F64":
        return np.frombuffer(raw, dtype=np.float64).reshape(shape).astype(np.float32)
    # Rare integer / bool tensors in weight files — promote for the quant path.
    np_dtype = {
        "I8": np.int8,
        "I16": np.int16,
        "I32": np.int32,
        "I64": np.int64,
        "U8": np.uint8,
        "U16": np.uint16,
        "U32": np.uint32,
        "U64": np.uint64,
        "BOOL": np.bool_,
    }.get(dtype)
    if np_dtype is None:
        raise UnsupportedError(f"unsupported safetensors dtype: {dtype}")
    return np.frombuffer(raw, dtype=np_dtype).reshape(shape).astype(np.float32)


def iter_safetensors_f32(path: str | Path) -> Generator[tuple[str, np.ndarray], None, None]:
    """Stream tensors from a .safetensors file as float32 (handles BF16 without torch)."""
    path = Path(path)
    try:
        with path.open("rb") as fo:
            with mmap.mmap(fo.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                if len(mm) < 8:
                    raise UnsupportedError(f"safetensors too short: {path}")
                header_len = struct.unpack_from("<Q", mm, 0)[0]
                header_end = 8 + header_len
                if header_end > len(mm):
                    raise UnsupportedError(f"invalid safetensors header length: {path}")
                try:
                    header = json.loads(bytes(mm[8:header_end]))
                except json.JSONDecodeError as e:
                    raise UnsupportedError(f"invalid safetensors header JSON: {path}") from e
                if not isinstance(header, dict):
                    raise UnsupportedError(f"safetensors header must be an object: {path}")
                data_base = header_end
                for name, info in header.items():
                    if name == "__metadata__":
                        continue
                    if not isinstance(info, dict):
                        raise UnsupportedError(f"bad tensor info for {name!r} in {path}")
                    dtype = info.get("dtype")
                    shape = tuple(info.get("shape") or ())
                    offsets = info.get("data_offsets")
                    if not isinstance(dtype, str) or not isinstance(offsets, (list, tuple)) or len(offsets) != 2:
                        raise UnsupportedError(f"incomplete tensor info for {name!r} in {path}")
                    start, end = int(offsets[0]), int(offsets[1])
                    raw = mm[data_base + start : data_base + end]
                    yield name, _safetensors_bytes_to_f32(raw, dtype, shape)
    except OSError as e:
        raise ModelFetchError(str(path), str(e), kind="missing") from e


def make_tiny_state_dict(
    vocab: int = 128,
    hidden: int = 64,
    layers: int = 2,
    inter: int = 128,
    heads: int = 4,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Synthetic checkpoint; hidden=64 so rows divisible by group_size=32."""
    rng = np.random.default_rng(seed)
    tensors: dict[str, np.ndarray] = {}
    tensors["token_embd.weight"] = rng.normal(0, 0.02, size=(vocab, hidden)).astype(np.float32)
    for L in range(layers):
        tensors[f"blk.{L}.attn_norm.weight"] = rng.normal(0, 0.02, size=(hidden,)).astype(np.float32)
        tensors[f"blk.{L}.ffn_norm.weight"] = rng.normal(0, 0.02, size=(hidden,)).astype(np.float32)
        tensors[f"blk.{L}.attn_q.weight"] = rng.normal(0, 0.02, size=(hidden, hidden)).astype(np.float32)
        tensors[f"blk.{L}.attn_k.weight"] = rng.normal(0, 0.02, size=(hidden, hidden)).astype(np.float32)
        tensors[f"blk.{L}.attn_v.weight"] = rng.normal(0, 0.02, size=(hidden, hidden)).astype(np.float32)
        tensors[f"blk.{L}.attn_output.weight"] = rng.normal(0, 0.02, size=(hidden, hidden)).astype(np.float32)
        tensors[f"blk.{L}.ffn_gate.weight"] = rng.normal(0, 0.02, size=(inter, hidden)).astype(np.float32)
        tensors[f"blk.{L}.ffn_up.weight"] = rng.normal(0, 0.02, size=(inter, hidden)).astype(np.float32)
        tensors[f"blk.{L}.ffn_down.weight"] = rng.normal(0, 0.02, size=(hidden, inter)).astype(np.float32)
    tensors["output_norm.weight"] = rng.normal(0, 0.02, size=(hidden,)).astype(np.float32)
    tensors["output.weight"] = rng.normal(0, 0.02, size=(vocab, hidden)).astype(np.float32)
    return tensors


def tiny_model_config(
    vocab: int = 128,
    hidden: int = 64,
    layers: int = 2,
    inter: int = 128,
    heads: int = 4,
) -> dict:
    return {
        "hidden_size": hidden,
        "num_layers": layers,
        "num_attention_heads": heads,
        "num_kv_heads": heads,
        "intermediate_size": inter,
        "vocab_size": vocab,
        "context_length": 64,
        "rope_theta": 10000.0,
    }


def load_model_config(repo: str) -> dict:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ModelFetchError(repo, f"huggingface_hub not installed: {e}", kind="missing")
    try:
        path = hf_hub_download(repo_id=repo, filename="config.json")
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:
        msg = str(e)
        kind = "network"
        if "401" in msg or "403" in msg or "auth" in msg.lower():
            kind = "auth"
        elif "404" in msg or "not found" in msg.lower():
            kind = "missing"
        raise ModelFetchError(repo, msg, kind=kind) from e
    return config_from_hf(raw)


def config_from_hf(model_config: dict) -> dict:
    """Flatten common HF config (+ nested text_config) into aria model fields."""
    cfg = dict(model_config)
    text = cfg.get("text_config")
    if isinstance(text, dict):
        for k, v in text.items():
            cfg.setdefault(k, v)

    def pick(*keys, default=None):
        for k in keys:
            if k in cfg and cfg[k] is not None:
                return cfg[k]
        return default

    hidden = pick("hidden_size", "d_model", "n_embd")
    if hidden is None:
        raise ConfigError("model config missing hidden_size")
    layers = pick("num_hidden_layers", "n_layer", "num_layers")
    if layers is None:
        raise ConfigError("model config missing num_hidden_layers")
    heads = pick("num_attention_heads", "n_head") or 1
    kv = pick("num_key_value_heads", "num_kv_heads") or heads
    inter = pick("intermediate_size", "ffn_dim", "n_inner") or (4 * int(hidden))
    vocab = pick("vocab_size") or 0
    ctx = pick("max_position_embeddings", "context_length") or 2048
    rope = pick("rope_theta") or 10000.0
    return {
        "hidden_size": int(hidden),
        "num_layers": int(layers),
        "num_attention_heads": int(heads),
        "num_kv_heads": int(kv),
        "intermediate_size": int(inter),
        "vocab_size": int(vocab),
        "context_length": int(ctx),
        "rope_theta": float(rope),
    }


def stream_weights(repo: str) -> Generator[tuple[str, np.ndarray], None, None]:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as e:
        raise ModelFetchError(repo, f"deps missing for stream_weights: {e}", kind="missing")

    try:
        files = list_repo_files(repo)
    except Exception as e:
        raise ModelFetchError(repo, str(e), kind="network") from e

    st_files = [f for f in files if f.endswith(".safetensors") and not f.endswith(".safetensors.index.json")]
    # prefer sharded model-*.safetensors; skip openvino etc.
    st_files = [f for f in st_files if "model" in Path(f).name or f.endswith("model.safetensors")]
    if not st_files:
        st_files = [f for f in files if f.endswith(".safetensors")]
    if not st_files:
        raise ModelFetchError(repo, "no safetensors weights found", kind="missing")

    for fname in sorted(st_files):
        try:
            local = hf_hub_download(repo_id=repo, filename=fname)
        except Exception as e:
            raise ModelFetchError(repo, str(e), kind="network") from e
        # mmap reader: numpy's safetensors path cannot decode BF16 (Gemma/Qwen).
        yield from iter_safetensors_f32(local)


def copy_tokenizer(repo_or_path: str, dest_dir: str | Path) -> None:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    src = Path(repo_or_path)
    if src.is_dir():
        for fname in TOKENIZER_FILES:
            p = src / fname
            if p.is_file():
                shutil.copy2(p, dest / fname)
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ModelFetchError(repo_or_path, f"huggingface_hub not installed: {e}", kind="missing")
    for fname in TOKENIZER_FILES:
        try:
            path = hf_hub_download(repo_id=repo_or_path, filename=fname)
            shutil.copy2(path, dest / fname)
        except Exception:
            continue
