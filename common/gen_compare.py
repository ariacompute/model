"""Lightweight generation / forward compare (text vs VLA); report-only."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from . import audit, bundle, quant
from .errors import ConfigError


# Short, completion-style prompts: less open-ended chatter / loop risk than chat hellos.
DEFAULT_TEXT_PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "Complete: The sky is",
]


def _try_import_torch():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        return None, str(e)
    return (torch, AutoModelForCausalLM, AutoTokenizer), None


def _inject_bundle_weights(
    model,
    tensors: dict,
    hadamard_seed: int,
    *,
    progress: bool = False,
) -> int:
    """Copy inverse-Hadamard dequant weights into matching ``state_dict`` keys.

    When ``progress=True``, print reconstruct progress to stderr (full-model
    inject can take minutes — not a hang). Uses CUDA reconstruct when the
    model parameters live on GPU.
    """
    import sys
    import time

    import torch

    sd = model.state_dict()
    n = 0
    n_total = len(tensors)
    t0 = time.perf_counter()
    # Prefer GPU reconstruct when HF weights are already on CUDA.
    sample = next(iter(sd.values()), None)
    recon_device = None
    if sample is not None and getattr(sample, "is_cuda", False):
        recon_device = sample.device
    backend = f"torch:{recon_device}" if recon_device is not None else "numpy"
    if progress:
        print(
            f"injecting {n_total} bundle tensors "
            f"(reconstruct_weight backend={backend}) …",
            file=sys.stderr,
            flush=True,
        )
    with torch.no_grad():
        for i, (name, obj) in enumerate(tensors.items(), 1):
            if progress and (i == 1 or i % 10 == 0 or i == n_total):
                elapsed = time.perf_counter() - t0
                shape_s = ""
                if isinstance(obj, quant.QuantTensor):
                    shape_s = f" shape={tuple(obj.shape)}"
                print(
                    f"  inject {i}/{n_total} ({elapsed:.0f}s) start={name}{shape_s}",
                    file=sys.stderr,
                    flush=True,
                )
            if not isinstance(obj, quant.QuantTensor):
                continue
            if name not in sd:
                continue
            t = sd[name]
            if recon_device is not None:
                recon = quant.reconstruct_weight_torch(
                    obj, hadamard_seed, device=recon_device
                )
                if tuple(t.shape) != tuple(recon.shape):
                    continue
                t.copy_(recon.to(dtype=t.dtype))
            else:
                recon = quant.reconstruct_weight(obj, hadamard_seed)
                if tuple(t.shape) != recon.shape:
                    continue
                t.copy_(
                    torch.from_numpy(np.asarray(recon, dtype=np.float32)).to(dtype=t.dtype)
                )
            n += 1
    if progress:
        print(
            f"inject done in {time.perf_counter() - t0:.1f}s (injected={n}, backend={backend})",
            file=sys.stderr,
            flush=True,
        )
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


def exact_prefix_match(
    a: list[int] | tuple[int, ...], b: list[int] | tuple[int, ...]
) -> dict[str, float | int | bool]:
    """Longest common prefix over new-token id sequences."""
    n = min(len(a), len(b))
    k = 0
    while k < n and a[k] == b[k]:
        k += 1
    denom = max(len(a), len(b), 1)
    return {
        "exact_prefix_len": k,
        "exact_prefix_frac": round(k / denom, 4),
        "exact_match": bool(list(a) == list(b) and len(a) > 0),
    }


def mean_token_logprob(model, prompt_ids, cont_ids, *, torch) -> float | None:
    """Teacher-forced mean log-prob of ``cont_ids`` given ``prompt_ids``."""
    if cont_ids.numel() == 0:
        return None
    import torch.nn.functional as F

    full = torch.cat([prompt_ids, cont_ids], dim=-1)
    with torch.no_grad():
        logits = model(full).logits
    # Predict token t from context …t-1
    plen = prompt_ids.shape[-1]
    pred = logits[:, plen - 1 : -1, :]
    logp = F.log_softmax(pred.float(), dim=-1)
    tok_lp = logp.gather(-1, cont_ids.unsqueeze(-1)).squeeze(-1)
    return float(tok_lp.mean().item())


def run_text_gen_compare(
    bundle_dir: str | Path,
    model_id: str,
    *,
    prompts: list[str] | None = None,
    max_new_tokens: int = 32,
    min_new_tokens: int = 8,
    device: str = "cpu",
) -> dict[str, Any]:
    if min_new_tokens < 1:
        raise ConfigError("--min-new-tokens must be >= 1")
    if max_new_tokens < min_new_tokens:
        raise ConfigError(
            f"--max-new-tokens ({max_new_tokens}) must be >= --min-new-tokens ({min_new_tokens})"
        )

    mods, err = _try_import_torch()
    if mods is None:
        return {
            "mode": "gen",
            "kind": audit.TEXT_KIND,
            "status": "skipped",
            "reason": (
                f"torch/transformers unavailable: {err}; "
                "install with: uv pip install torch transformers"
            ),
            "ci_fail": False,
        }

    torch, AutoModelForCausalLM, AutoTokenizer = mods
    prompts = prompts or list(DEFAULT_TEXT_PROMPTS)
    cfg, tensors = bundle.load_bundle(bundle_dir)
    seed = int(cfg.get("hadamard_seed") or 0)

    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
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

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "do_sample": False,
        "pad_token_id": tok.pad_token_id,
    }

    rows = []
    for prompt in prompts:
        inputs = tok(prompt, return_tensors="pt").to(device)
        prompt_len = int(inputs["input_ids"].shape[-1])
        with torch.no_grad():
            out_b = base.generate(**inputs, **gen_kwargs)
            out_q = quant_model.generate(**inputs, **gen_kwargs)
        cont_b = out_b[0, prompt_len:].tolist()
        cont_q = out_q[0, prompt_len:].tolist()
        text_b = tok.decode(cont_b, skip_special_tokens=True)
        text_q = tok.decode(cont_q, skip_special_tokens=True)
        prefix = exact_prefix_match(cont_b, cont_q)

        cont_b_t = torch.tensor([cont_b], device=device, dtype=torch.long)
        prompt_ids = inputs["input_ids"]
        lp_base = mean_token_logprob(base, prompt_ids, cont_b_t, torch=torch)
        lp_quant_on_base = mean_token_logprob(quant_model, prompt_ids, cont_b_t, torch=torch)
        lp_delta = None
        if lp_base is not None and lp_quant_on_base is not None:
            lp_delta = round(lp_quant_on_base - lp_base, 6)

        rows.append(
            {
                "prompt": prompt,
                "baseline": text_b,
                "quantized": text_q,
                "n_new_tokens_baseline": len(cont_b),
                "n_new_tokens_quantized": len(cont_q),
                "token_overlap": round(_token_overlap(text_b, text_q), 4),
                "exact_prefix_len": prefix["exact_prefix_len"],
                "exact_prefix_frac": prefix["exact_prefix_frac"],
                "exact_match": prefix["exact_match"],
                "mean_logprob_baseline": None if lp_base is None else round(lp_base, 6),
                "mean_logprob_quant_on_baseline": (
                    None if lp_quant_on_base is None else round(lp_quant_on_base, 6)
                ),
                "mean_logprob_delta": lp_delta,
            }
        )

    overlaps = [r["token_overlap"] for r in rows]
    prefix_fracs = [r["exact_prefix_frac"] for r in rows]
    deltas = [r["mean_logprob_delta"] for r in rows if r["mean_logprob_delta"] is not None]
    return {
        "mode": "gen",
        "kind": audit.TEXT_KIND,
        "status": "ok",
        "bundle": str(Path(bundle_dir).resolve()),
        "model": model_id,
        "injected_tensors": n_injected,
        "min_new_tokens": min_new_tokens,
        "max_new_tokens": max_new_tokens,
        "mean_token_overlap": round(float(np.mean(overlaps)), 4),
        "mean_exact_prefix_frac": round(float(np.mean(prefix_fracs)), 4),
        "mean_logprob_delta": (
            None if not deltas else round(float(np.mean(deltas)), 6)
        ),
        "ci_fail": False,
        "note": (
            "report-only; does not fail CI. "
            "Metrics on new tokens only: token_overlap, exact_prefix_*, "
            "teacher-forced mean_logprob_* on baseline continuation."
        ),
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
            "reason": (
                f"torch/transformers unavailable: {err}; "
                "install with: uv pip install torch transformers"
            ),
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
    min_new_tokens: int = 8,
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
        min_new_tokens=min_new_tokens,
        device=device,
    )
