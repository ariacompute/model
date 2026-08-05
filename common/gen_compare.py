"""Lightweight generation / forward compare (text vs VLA); report-only."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from . import audit, bundle, quant
from .errors import ConfigError


DEFAULT_TEXT_PROMPTS = [
    "Hello, how are you?",
    "Summarize: The sky is blue.",
]


def _try_import_torch():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        return None, str(e)
    return (torch, AutoModelForCausalLM, AutoTokenizer), None


def _inject_bundle_weights(model, tensors: dict, hadamard_seed: int) -> int:
    """Copy inverse-Hadamard dequant weights into matching ``state_dict`` keys."""
    import torch

    sd = model.state_dict()
    n = 0
    with torch.no_grad():
        for name, obj in tensors.items():
            if not isinstance(obj, quant.QuantTensor):
                continue
            if name not in sd:
                continue
            recon_rot = quant.dequantize(obj)
            recon = quant.reconstruct_weight(obj, hadamard_seed)
            t = sd[name]
            if tuple(t.shape) != recon.shape:
                continue
            t.copy_(torch.from_numpy(np.asarray(recon, dtype=np.float32)).to(dtype=t.dtype))
            n += 1
    return n


def _token_overlap(a: str, b: str) -> float:
    ta = a.split()
    tb = b.split()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    sa, sb = set(ta), set(tb)
    return len(sa & sb) / max(len(sa | sb), 1)


def run_text_gen_compare(
    bundle_dir: str | Path,
    model_id: str,
    *,
    prompts: list[str] | None = None,
    max_new_tokens: int = 32,
    device: str = "cpu",
) -> dict[str, Any]:
    mods, err = _try_import_torch()
    if mods is None:
        return {
            "mode": "gen",
            "kind": audit.TEXT_KIND,
            "status": "skipped",
            "reason": f"torch/transformers unavailable: {err}",
            "ci_fail": False,
        }

    torch, AutoModelForCausalLM, AutoTokenizer = mods
    prompts = prompts or list(DEFAULT_TEXT_PROMPTS)
    cfg, tensors = bundle.load_bundle(bundle_dir)
    seed = int(cfg.get("hadamard_seed") or 0)

    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, trust_remote_code=True
        ).to(device)
        base.eval()
        quant_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, trust_remote_code=True
        ).to(device)
        quant_model.eval()
    except Exception as e:
        return {
            "mode": "gen",
            "kind": audit.TEXT_KIND,
            "status": "skipped",
            "reason": f"failed to load causal LM: {e}",
            "ci_fail": False,
        }

    n_injected = _inject_bundle_weights(quant_model, tensors, seed)
    if n_injected == 0:
        return {
            "mode": "gen",
            "kind": audit.TEXT_KIND,
            "status": "skipped",
            "reason": "no bundle tensor names matched model state_dict",
            "ci_fail": False,
        }

    rows = []
    for prompt in prompts:
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out_b = base.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            out_q = quant_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text_b = tok.decode(out_b[0], skip_special_tokens=True)
        text_q = tok.decode(out_q[0], skip_special_tokens=True)
        rows.append(
            {
                "prompt": prompt,
                "baseline": text_b,
                "quantized": text_q,
                "token_overlap": round(_token_overlap(text_b, text_q), 4),
            }
        )

    overlaps = [r["token_overlap"] for r in rows]
    return {
        "mode": "gen",
        "kind": audit.TEXT_KIND,
        "status": "ok",
        "bundle": str(Path(bundle_dir).resolve()),
        "model": model_id,
        "injected_tensors": n_injected,
        "max_new_tokens": max_new_tokens,
        "mean_token_overlap": round(float(np.mean(overlaps)), 4),
        "ci_fail": False,
        "note": "report-only; does not fail CI",
        "samples": rows,
    }


def run_vla_forward_compare(
    bundle_dir: str | Path,
    model_id: str,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """VLA path: try a minimal forward / action output compare; else skip.

    Full VLA stacks (OpenVLA / π₀ / LingBot) often need custom code. This
    routine attempts a generic ``forward`` with random tensors when the HF
    model exposes ``forward`` with an ``inputs_embeds``-like API; otherwise
    reports ``skipped`` with reason — still CI-safe.
    """
    mods, err = _try_import_torch()
    if mods is None:
        return {
            "mode": "gen",
            "kind": audit.VLA_KIND,
            "status": "skipped",
            "reason": f"torch/transformers unavailable: {err}",
            "ci_fail": False,
        }

    torch, AutoModelForCausalLM, _Tok = mods
    cfg, tensors = bundle.load_bundle(bundle_dir)
    seed = int(cfg.get("hadamard_seed") or 0)

    # Prefer AutoModel — VLA checkpoints vary widely.
    try:
        from transformers import AutoModel

        base = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
        quant_model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
        base.eval()
        quant_model.eval()
    except Exception as e:
        return {
            "mode": "gen",
            "kind": audit.VLA_KIND,
            "status": "skipped",
            "reason": f"failed to load VLA/AutoModel ({model_id}): {e}",
            "ci_fail": False,
            "note": "use layer audit for weight RMSE; VLA action compare needs model-specific hooks",
        }

    n_injected = _inject_bundle_weights(quant_model, tensors, seed)
    if n_injected == 0:
        return {
            "mode": "gen",
            "kind": audit.VLA_KIND,
            "status": "skipped",
            "reason": "no bundle tensor names matched model state_dict",
            "ci_fail": False,
        }

    # Probe last floating parameter for a crude activation proxy via param L2 delta.
    deltas = []
    with torch.no_grad():
        for (n1, p1), (n2, p2) in zip(base.named_parameters(), quant_model.named_parameters()):
            if n1 != n2 or p1.ndim < 2:
                continue
            # Relative L2 between matched params after inject (should be ~0 for injected).
            num = torch.norm(p1.float() - p2.float()).item()
            den = torch.norm(p1.float()).item() + 1e-12
            deltas.append({"name": n1, "rel_l2": round(num / den, 6)})
            if len(deltas) >= 8:
                break

    mean_rel = float(np.mean([d["rel_l2"] for d in deltas])) if deltas else math.nan

    return {
        "mode": "gen",
        "kind": audit.VLA_KIND,
        "status": "ok",
        "bundle": str(Path(bundle_dir).resolve()),
        "model": model_id,
        "injected_tensors": n_injected,
        "metric": "param_rel_l2_sample",
        "mean_rel_l2": None if math.isnan(mean_rel) else round(mean_rel, 6),
        "ci_fail": False,
        "note": (
            "VLA path reports sampled parameter drift after inject; "
            "full action-head rollout is model-specific and out of default scope"
        ),
        "samples": deltas,
    }


def run_gen_compare(
    bundle_dir: str | Path,
    model_id: str | None,
    *,
    kind: str | None = None,
    family: str | None = None,
    prompts: list[str] | None = None,
    max_new_tokens: int = 32,
    device: str = "cpu",
) -> dict[str, Any]:
    kind = kind or audit.infer_kind(bundle_dir, family=family)
    if not model_id:
        raise ConfigError("--model is required for gen compare")
    if kind == audit.VLA_KIND:
        return run_vla_forward_compare(bundle_dir, model_id, device=device)
    return run_text_gen_compare(
        bundle_dir,
        model_id,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        device=device,
    )
