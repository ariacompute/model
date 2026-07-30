"""Model bundle writer/reader: weight.bin + config.json (+ tokenizer)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from .quant import QuantTensor
from .errors import FormatError, ShapeMismatchError

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)

BUNDLE_FORMAT = "aria-quant-bundle"


def _f16_bytes(arr: np.ndarray) -> bytes:
    return np.asarray(arr, dtype=np.float16).tobytes(order="C")


def _f32_bytes(arr: np.ndarray) -> bytes:
    return np.asarray(arr, dtype=np.float32).tobytes(order="C")


class BundleWriter:
    """Stream tensors to ``weight.bin`` one at a time (OOM-safe)."""

    def __init__(
        self,
        out_dir: str | Path,
        model_config: dict,
        quantization: str,
        group_size_default: int = 32,
        hadamard_seed: int | None = None,
    ):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.bin_path = self.out / "weight.bin"
        self._fh: BinaryIO = self.bin_path.open("wb")
        self._offset = 0
        self.tensor_meta: dict[str, Any] = {}
        self.model_config = model_config
        self.quantization = quantization
        self.group_size_default = group_size_default
        self.hadamard_seed = hadamard_seed

    def _write(self, data: bytes) -> tuple[int, int]:
        start = self._offset
        self._fh.write(data)
        length = len(data)
        self._offset += length
        return start, length

    def add(self, name: str, obj: QuantTensor | np.ndarray) -> None:
        if isinstance(obj, QuantTensor):
            off: dict[str, list[int]] = {}
            off["packed_indices"] = list(self._write(obj.packed_indices))
            off["codebook"] = list(self._write(_f16_bytes(obj.codebook)))
            off["input_scale"] = list(self._write(_f16_bytes(obj.input_scale)))
            off["input_scale_recip"] = list(self._write(_f16_bytes(obj.input_scale_recip)))
            off["norms"] = list(self._write(_f16_bytes(obj.norms)))
            self.tensor_meta[name] = {
                "kind": "codebook",
                "bits": obj.bits,
                "group_size": obj.group_size,
                "shape": [int(obj.shape[0]), int(obj.shape[1])],
                "row_pad": int(obj.row_pad),
                "hadamard": dict(obj.hadamard_meta),
                "offsets": off,
            }
        else:
            arr = np.asarray(obj)
            if arr.dtype == np.float32:
                dtype = "f32"
                raw = _f32_bytes(arr)
            else:
                dtype = "f16"
                raw = _f16_bytes(arr.astype(np.float16, copy=False))
            off = {"data": list(self._write(raw))}
            self.tensor_meta[name] = {
                "kind": "raw",
                "dtype": dtype,
                "shape": [int(d) for d in arr.shape],
                "offsets": off,
            }

    def close(self) -> Path:
        if not self._fh.closed:
            self._fh.close()
        config = {
            "format": BUNDLE_FORMAT,
            "format_version": 1,
            "quantization": self.quantization,
            "group_size_default": self.group_size_default,
            "hadamard_seed": self.hadamard_seed,
            "model": self.model_config,
            "tensors": self.tensor_meta,
        }
        (self.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        return self.out

    def __enter__(self) -> "BundleWriter":
        return self

    def __exit__(self, *exc) -> None:
        if not self._fh.closed:
            self.close()


def write_bundle(
    out_dir: str | Path,
    tensors: dict[str, QuantTensor | np.ndarray],
    model_config: dict,
    quantization: str,
    tokenizer_src: str | None = None,
    group_size_default: int = 32,
    hadamard_seed: int | None = None,
) -> Path:
    with BundleWriter(
        out_dir,
        model_config,
        quantization,
        group_size_default=group_size_default,
        hadamard_seed=hadamard_seed,
    ) as writer:
        for name, obj in tensors.items():
            writer.add(name, obj)
        out = writer.close()

    if tokenizer_src:
        src = Path(tokenizer_src)
        if src.is_dir():
            for fname in TOKENIZER_FILES:
                p = src / fname
                if p.is_file():
                    shutil.copy2(p, out / fname)
    return out


def _read_slice(blob: bytes, start: int, length: int) -> bytes:
    end = start + length
    if start < 0 or end > len(blob):
        raise FormatError(f"offset [{start},{length}] out of range (bin size {len(blob)})")
    return blob[start:end]


def load_bundle(out_dir: str | Path) -> tuple[dict, dict[str, QuantTensor | np.ndarray]]:
    out = Path(out_dir)
    cfg_path = out / "config.json"
    bin_path = out / "weight.bin"
    if not cfg_path.is_file():
        raise FormatError(f"missing config.json in {out}")
    if not bin_path.is_file():
        raise FormatError(f"missing weight.bin in {out}")

    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    if config.get("format") != BUNDLE_FORMAT:
        raise FormatError(f"unsupported format {config.get('format')!r}")
    blob = bin_path.read_bytes()
    tensors: dict[str, QuantTensor | np.ndarray] = {}

    for name, meta in config.get("tensors", {}).items():
        kind = meta.get("kind")
        offsets = meta.get("offsets") or {}
        if kind == "codebook":
            def seg(key: str) -> bytes:
                if key not in offsets:
                    raise FormatError(f"tensor {name} missing offset {key}")
                s, L = offsets[key]
                return _read_slice(blob, int(s), int(L))

            shape = tuple(meta["shape"])
            bits = int(meta["bits"])
            gs = int(meta["group_size"])
            k, n = shape
            cb_raw = seg("codebook")
            kc = 1 << bits
            if len(cb_raw) % 2 != 0:
                raise FormatError(f"codebook byte length odd for {name}")
            n_elem = len(cb_raw) // 2
            if n * kc == 0 or n_elem % (n * kc) != 0:
                raise ShapeMismatchError(f"cannot infer groups for {name}")
            num_groups = n_elem // (n * kc)
            codebook = np.frombuffer(cb_raw, dtype=np.float16).reshape(num_groups, n, kc).copy()
            scales = np.frombuffer(seg("input_scale"), dtype=np.float16).reshape(num_groups, n).copy()
            recip = np.frombuffer(seg("input_scale_recip"), dtype=np.float16).reshape(num_groups, n).copy()
            norms = np.frombuffer(seg("norms"), dtype=np.float16).reshape(num_groups, n).copy()
            packed = seg("packed_indices")
            tensors[name] = QuantTensor(
                bits=bits,
                group_size=gs,
                shape=(k, n),
                packed_indices=packed,
                codebook=codebook,
                input_scale=scales,
                input_scale_recip=recip,
                norms=norms,
                hadamard_meta=dict(meta.get("hadamard") or {}),
                row_pad=int(meta.get("row_pad", 0)),
            )
        elif kind == "raw":
            s, L = offsets["data"]
            raw = _read_slice(blob, int(s), int(L))
            dtype = meta.get("dtype", "f16")
            shape = tuple(meta["shape"])
            if dtype == "f32":
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
            else:
                arr = np.frombuffer(raw, dtype=np.float16).reshape(shape).copy()
            tensors[name] = arr
        else:
            raise FormatError(f"unknown tensor kind {kind!r} for {name}")

    return config, tensors
