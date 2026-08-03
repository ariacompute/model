"""Layer-wise dequant audit against reference weights (report-only thresholds)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import bundle, hadamard, hf_utils, quant
from .errors import ConfigError, QuantError

TEXT_KIND = "text"
VLA_KIND = "vla"

VLA_SLUG_PREFIXES = ("openvla", "openpi", "lingbot")
TEXT_SLUG_PREFIXES = ("qwen", "gemma", "lfm", "inkling", "nanbeige", "bonsai")


def infer_kind(bundle_dir: str | Path | None = None, family: str | None = None) -> str:
    """Infer ``text`` vs ``vla`` from family slug or bundle path."""
    tokens: list[str] = []
    if family:
        tokens.append(family.lower())
    if bundle_dir is not None:
        tokens.append(Path(bundle_dir).name.lower())
        tokens.append(str(Path(bundle_dir)).lower())
    blob = " ".join(tokens)
    for p in VLA_SLUG_PREFIXES:
        if p in blob:
            return VLA_KIND
    for p in TEXT_SLUG_PREFIXES:
        if p in blob:
            return TEXT_KIND
    return TEXT_KIND


def classify_layer_role(name: str) -> str:
    low = name.lower()
    if any(s in low for s in ("vision", "visual", "image", "siglip", "dino", "patch")):
        return "vision"
    if any(s in low for s in ("action", "policy", "expert", "flow")):
        return "action"
    if any(s in low for s in ("embed", "embd", "embedding", "per_layer", "ple")):
        return "embed"
    if any(
        s in low
        for s in (
            "attn",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "self_attn",
            "lm_head",
        )
    ):
        return "attn"
    if any(s in low for s in ("ffn", "mlp", "gate_proj", "up_proj", "down_proj", "moe")):
        return "ffn"
    return "other"


def threshold_orig_rmse(bits: int, name: str = "") -> float:
    """Report-only upper bound for original-space relative RMSE."""
    low = name.lower()
    if bits >= 8:
        return 0.15
    if bits == 4:
        return 0.35
    if bits <= 2 or any(s in low for s in ("embed", "embd", "embedding", "per_layer")):
        return 0.80
    return 0.50


def rel_rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise QuantError(f"shape mismatch for RMSE: {a.shape} vs {b.shape}")
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    rms = float(np.sqrt(np.mean(a ** 2))) + 1e-12
    return rmse / rms


def stratified_sample(names: list[str], k: int, seed: int = 0) -> list[str]:
    """Pick up to ``k`` names covering roles when possible."""
    if k <= 0 or not names:
        return []
    rng = np.random.default_rng(seed)
    by_role: dict[str, list[str]] = {}
    for n in names:
        by_role.setdefault(classify_layer_role(n), []).append(n)
    for role in by_role:
        ordered = sorted(by_role[role])
        by_role[role] = [ordered[i] for i in rng.permutation(len(ordered))]

    order = ["embed", "attn", "ffn", "vision", "action", "other"]
    picked: list[str] = []
    # Round-robin across roles.
    while len(picked) < k:
        progressed = False
        for role in order:
            bucket = by_role.get(role) or []
            if not bucket:
                continue
            name = bucket.pop()
            if name not in picked:
                picked.append(name)
                progressed = True
            if len(picked) >= k:
                break
        if not progressed:
            break
    return picked


def inverse_hadamard(W_rot: np.ndarray, seed: int | None) -> np.ndarray:
    """Hadamard is involutory under the same seed/signs — apply rotate again."""
    out, meta = hadamard.hadamard_rotate(W_rot, axis=0, seed=seed)
    if not meta.get("applied"):
        raise QuantError("inverse Hadamard failed to apply")
    return out


def audit_one_tensor(
    name: str,
    qt: quant.QuantTensor,
    W: np.ndarray,
    seed: int | None,
) -> dict[str, Any]:
    W = np.asarray(W, dtype=np.float32)
    if W.shape != qt.shape:
        raise QuantError(f"{name}: ref shape {W.shape} != bundle {qt.shape}")
    W_rot, hmeta = hadamard.hadamard_rotate(W, axis=0, seed=seed)
    recon_rot = quant.dequantize(qt)
    err_rot = rel_rmse(W_rot, recon_rot)
    recon_orig = inverse_hadamard(recon_rot, seed=seed)
    err_orig = rel_rmse(W, recon_orig)
    thr = threshold_orig_rmse(int(qt.bits), name)
    return {
        "name": name,
        "role": classify_layer_role(name),
        "bits": int(qt.bits),
        "shape": [int(qt.shape[0]), int(qt.shape[1])],
        "codebook_share": getattr(qt, "codebook_share", "group"),
        "rel_rmse_rot": round(err_rot, 6),
        "rel_rmse_orig": round(err_orig, 6),
        "threshold_orig": thr,
        "pass": bool(err_orig <= thr),
        "hadamard_applied": bool(hmeta.get("applied")),
    }


def load_ref_weights(
    names: Iterable[str],
    *,
    model: str | None = None,
    ref_tiny: bool = False,
    tiny_seed: int = 0,
) -> dict[str, np.ndarray]:
    want = set(names)
    if ref_tiny:
        sd = hf_utils.make_tiny_state_dict(seed=tiny_seed)
        missing = sorted(want - set(sd))
        if missing:
            raise ConfigError(f"tiny ref missing tensors: {missing[:5]}")
        return {n: sd[n] for n in want}
    if not model:
        raise ConfigError("--model is required unless --ref tiny")
    found: dict[str, np.ndarray] = {}
    for name, arr in hf_utils.stream_weights(model):
        if name in want:
            found[name] = arr
            if len(found) == len(want):
                break
    missing = sorted(want - set(found))
    if missing:
        raise ConfigError(f"HF model missing tensors: {missing[:8]}")
    return found


def run_layer_audit(
    bundle_dir: str | Path,
    *,
    model: str | None = None,
    sample: int = 8,
    seed: int = 0,
    ref_tiny: bool = False,
    family: str | None = None,
) -> dict[str, Any]:
    cfg, tensors = bundle.load_bundle(bundle_dir)
    codebook_names = sorted(
        n for n, t in tensors.items() if isinstance(t, quant.QuantTensor)
    )
    picked = stratified_sample(codebook_names, sample, seed=seed)
    hadamard_seed = cfg.get("hadamard_seed")
    if hadamard_seed is None:
        hadamard_seed = 0
    refs = load_ref_weights(
        picked, model=model, ref_tiny=ref_tiny, tiny_seed=int(hadamard_seed)
    )
    rows = [
        audit_one_tensor(n, tensors[n], refs[n], int(hadamard_seed))
        for n in picked
    ]
    n_fail = sum(1 for r in rows if not r["pass"])
    kind = infer_kind(bundle_dir, family=family)
    report = {
        "mode": "layer",
        "kind": kind,
        "bundle": str(Path(bundle_dir).resolve()),
        "model": model,
        "ref": "tiny" if ref_tiny else "hf",
        "quantization": cfg.get("quantization"),
        "sample": len(rows),
        "fail_count": n_fail,
        "ci_fail": False,
        "note": "thresholds are report-only; exit code stays 0 for CI",
        "layers": rows,
    }
    return report


def write_report(report: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
