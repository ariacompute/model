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


def _scalar_int(value, *, name: str, reduce_seq=max) -> int:
    """Coerce HF config ints; MatFormer may store per-layer sequences (e.g. Gemma-3n)."""
    if isinstance(value, (list, tuple)):
        if not value:
            raise ConfigError(f"model config {name} is an empty list")
        try:
            return int(reduce_seq(int(x) for x in value))
        except (TypeError, ValueError) as e:
            raise ConfigError(f"model config {name} has non-integer entries: {value!r}") from e
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"model config {name} must be int-like, got {type(value).__name__}") from e


def _rope_theta_from_cfg(cfg: dict, pick) -> float:
    """Prefer top-level rope_theta; else nested rope_parameters.*.rope_theta / theta."""
    top = pick("rope_theta")
    if top is not None:
        return float(top)
    rp = cfg.get("rope_parameters")
    if isinstance(rp, dict):
        # Prefer full_attention / global, then any nested dict, then flat theta.
        for key in ("full_attention", "global", "default", "rope"):
            sub = rp.get(key)
            if isinstance(sub, dict):
                for tk in ("rope_theta", "theta", "base"):
                    if sub.get(tk) is not None:
                        return float(sub[tk])
        for tk in ("rope_theta", "theta", "base"):
            if rp.get(tk) is not None:
                return float(rp[tk])
        for sub in rp.values():
            if isinstance(sub, dict):
                for tk in ("rope_theta", "theta", "base"):
                    if sub.get(tk) is not None:
                        return float(sub[tk])
    return 10000.0


def _optional_int(value, *, name: str):
    if value is None:
        return None
    return _scalar_int(value, name=name)


def _optional_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("1", "true", "yes"):
            return True
        if low in ("0", "false", "no"):
            return False
    return bool(value)


def config_from_hf(model_config: dict) -> dict:
    """Flatten common HF config (+ nested text_config) into aria model fields."""
    cfg = dict(model_config)
    # Prefer nested text_config for LLM geometry when present (VL/VLA wrappers).
    text = cfg.get("text_config")
    if isinstance(text, dict):
        for k, v in text.items():
            # Nested text wins over unrelated top-level vision/audio scalars.
            if k not in cfg or cfg[k] is None:
                cfg[k] = v
            elif k in (
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "intermediate_size",
                "vocab_size",
                "head_dim",
                "rope_theta",
                "rope_parameters",
                "layer_types",
                "hidden_act",
                "hidden_activation",
            ):
                cfg[k] = v

    # OpenVLA / Prismatic: language_model or llm_config may hold decoder dims.
    for nest_key in ("llm_config", "language_model_config", "language_config"):
        nest = cfg.get(nest_key)
        if isinstance(nest, dict):
            for k, v in nest.items():
                cfg.setdefault(k, v)

    # OpenPI / LeRobot: paligemma / vlm nested configs.
    for nest_key in ("paligemma_config", "vlm_config", "policy_config"):
        nest = cfg.get(nest_key)
        if isinstance(nest, dict):
            inner = nest.get("text_config") if isinstance(nest.get("text_config"), dict) else nest
            if isinstance(inner, dict):
                for k, v in inner.items():
                    cfg.setdefault(k, v)

    # LingBot: may only declare vlm_family — try common qwen3_vl-sized defaults via nest.
    if cfg.get("vlm_family") and cfg.get("hidden_size") is None:
        # Cannot invent geometry; surface a clear error below.
        pass

    def pick(*keys, default=None):
        for k in keys:
            if k in cfg and cfg[k] is not None:
                return cfg[k]
        return default

    hidden = pick("hidden_size", "d_model", "n_embd")
    if hidden is None:
        raise ConfigError(
            "model config missing hidden_size "
            "(need top-level, text_config, or VLA nested llm/paligemma config)"
        )
    layers = pick("num_hidden_layers", "n_layer", "num_layers")
    if layers is None:
        raise ConfigError("model config missing num_hidden_layers")
    heads = pick("num_attention_heads", "n_head") or 1
    kv = pick("num_key_value_heads", "num_kv_heads") or heads
    inter = pick("intermediate_size", "ffn_dim", "n_inner", "block_ff_dim")
    if inter is None:
        inter = 4 * _scalar_int(hidden, name="hidden_size")
    vocab = pick("vocab_size") or 0
    ctx = pick("max_position_embeddings", "context_length", "model_max_length") or 2048
    rope = _rope_theta_from_cfg(cfg, pick)

    out = {
        "hidden_size": _scalar_int(hidden, name="hidden_size"),
        "num_layers": _scalar_int(layers, name="num_hidden_layers"),
        "num_attention_heads": _scalar_int(heads, name="num_attention_heads"),
        "num_kv_heads": _scalar_int(kv, name="num_key_value_heads"),
        # Gemma-3n MatFormer: per-layer intermediate_size list → keep max for metadata.
        "intermediate_size": _scalar_int(inter, name="intermediate_size", reduce_seq=max),
        "vocab_size": _scalar_int(vocab, name="vocab_size"),
        "context_length": _scalar_int(ctx, name="context_length"),
        "rope_theta": float(rope),
    }

    head_dim = pick("head_dim")
    if head_dim is not None:
        out["head_dim"] = _scalar_int(head_dim, name="head_dim")

    layer_types = pick("layer_types")
    if isinstance(layer_types, list) and layer_types:
        out["layer_types"] = [str(x) for x in layer_types]

    n_kv_shared = pick("num_kv_shared_layers")
    if n_kv_shared is not None:
        out["num_kv_shared_layers"] = _scalar_int(n_kv_shared, name="num_kv_shared_layers")

    dbl = pick("use_double_wide_mlp")
    if dbl is not None:
        out["use_double_wide_mlp"] = _optional_bool(dbl)

    act = pick("hidden_act", "hidden_activation")
    if act is not None:
        out["hidden_act"] = str(act)

    n_experts = pick("num_experts", "num_local_experts")
    if n_experts is not None:
        out["num_experts"] = _scalar_int(n_experts, name="num_experts")
    top_k = pick("num_experts_per_tok", "num_experts_per_token", "moe_top_k")
    if top_k is not None:
        out["num_experts_per_tok"] = _scalar_int(top_k, name="num_experts_per_tok")

    tie = pick("tie_word_embeddings")
    if tie is not None:
        out["tie_word_embeddings"] = _optional_bool(tie)

    return out



def load_model_info(repo: str) -> list[tuple[str, tuple[int, ...]]]:
    """List (name, shape) from safetensors headers only — no weight payloads."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as e:
        raise ModelFetchError(repo, f"deps missing for load_model_info: {e}", kind="missing")

    try:
        files = list_repo_files(repo)
    except Exception as e:
        raise ModelFetchError(repo, str(e), kind="network") from e

    st_files = _select_safetensors(files, repo)
    infos: list[tuple[str, tuple[int, ...]]] = []
    for fname in st_files:
        try:
            local = hf_hub_download(repo_id=repo, filename=fname)
        except Exception as e:
            raise ModelFetchError(repo, str(e), kind="network") from e
        infos.extend(_safetensors_header_shapes(local))
    return infos


def _select_safetensors(files: list[str], repo: str) -> list[str]:
    st_files = [f for f in files if f.endswith(".safetensors") and not f.endswith(".safetensors.index.json")]
    st_files = [f for f in st_files if "model" in Path(f).name or f.endswith("model.safetensors")]
    if not st_files:
        st_files = [f for f in files if f.endswith(".safetensors")]
    if not st_files:
        raise ModelFetchError(repo, "no safetensors weights found", kind="missing")
    return sorted(st_files)


def _safetensors_header_shapes(path: str | Path) -> list[tuple[str, tuple[int, ...]]]:
    path = Path(path)
    with path.open("rb") as fo:
        with mmap.mmap(fo.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            if len(mm) < 8:
                raise UnsupportedError(f"safetensors too short: {path}")
            header_len = struct.unpack_from("<Q", mm, 0)[0]
            header_end = 8 + header_len
            header = json.loads(bytes(mm[8:header_end]))
            out: list[tuple[str, tuple[int, ...]]] = []
            for name, info in header.items():
                if name == "__metadata__" or not isinstance(info, dict):
                    continue
                shape = tuple(int(d) for d in (info.get("shape") or ()))
                out.append((name, shape))
            return out


def stream_weights(repo: str) -> Generator[tuple[str, np.ndarray], None, None]:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as e:
        raise ModelFetchError(repo, f"deps missing for stream_weights: {e}", kind="missing")

    try:
        files = list_repo_files(repo)
    except Exception as e:
        raise ModelFetchError(repo, str(e), kind="network") from e

    st_files = _select_safetensors(files, repo)
    for fname in st_files:
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
