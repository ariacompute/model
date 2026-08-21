#!/usr/bin/env python3
"""GPU diagnostic: HF fp32 vs bundle reconstruct_weight on the same Gemma-4 chat prompt.

Isolates *quant* vs *chat-template* (engine.log Hello garbage, e.g. "uhnyaчь…",
prompt_tokens=28 from the old Gemma-3 `<start_of_turn>` markers). Uses official
Gemma apply_chat_template plus the engine gemma4_it string (`<|turn>` / `<turn|>`)
so prompt ids can be compared.

Gemma-4-E2B-it is a VL wrapper (Gemma4ForConditionalGeneration). Chat diag
injects **text LM weights only** by default (skips audio_tower / vision_*);
use --inject-all to reconstruct every tower (much slower). Tokenizer is loaded
from --hf (not the Aria bundle).

H200 example (from model repo root):

  pip install torch transformers
  python scripts/diag_gemma4_chat.py \\
    --bundle ~/.ariacompute/models/gemma-4-e2b-it_q4 \\
    --hf google/gemma-4-E2B-it \\
    --device cuda \\
    --report ./out/model_diag_gemma4.json

Attention uses eager kernels (no JIT CUDA / Python.h). If a compile error still
appears: sudo apt install python3-dev. Two fp32 copies of E2B need GPU RAM.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Avoid HuggingFace hub JIT CUDA kernels (cuda_utils.c needs Python.h / python3-dev).
os.environ.setdefault("TRANSFORMERS_USE_HUB_KERNELS", "0")
os.environ.setdefault("DISABLE_KERNEL_MAPPING", "1")
os.environ.setdefault("FLASH_ATTENTION_SKIP_CUDA_BUILD", "1")


def _pick_device(name: str) -> str:
    if name != "auto":
        return name
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_tokenizer(AutoTokenizer, name: str):
    """Load from HF repo (not the Aria bundle)."""
    kwargs = {"trust_remote_code": True}
    try:
        return AutoTokenizer.from_pretrained(name, fix_mistral_regex=True, **kwargs)
    except TypeError:
        return AutoTokenizer.from_pretrained(name, **kwargs)


def _from_pretrained(cls, name: str, torch, kwargs: dict):
    try:
        return cls.from_pretrained(name, dtype=torch.float32, **kwargs)
    except TypeError:
        try:
            return cls.from_pretrained(name, torch_dtype=torch.float32, **kwargs)
        except (TypeError, ValueError):
            kw = dict(kwargs)
            kw.pop("attn_implementation", None)
            return cls.from_pretrained(name, torch_dtype=torch.float32, **kw)
    except ValueError:
        kw = dict(kwargs)
        kw.pop("attn_implementation", None)
        try:
            return cls.from_pretrained(name, dtype=torch.float32, **kw)
        except TypeError:
            return cls.from_pretrained(name, torch_dtype=torch.float32, **kw)


def _model_classes():
    """Gemma-4-it is Gemma4ForConditionalGeneration; CausalLM may still work."""
    classes = []
    try:
        from transformers import AutoModelForCausalLM

        classes.append(AutoModelForCausalLM)
    except ImportError:
        pass
    try:
        from transformers import AutoModelForImageTextToText

        classes.append(AutoModelForImageTextToText)
    except ImportError:
        pass
    try:
        from transformers import AutoModelForConditionalGeneration

        classes.append(AutoModelForConditionalGeneration)
    except ImportError:
        pass
    return classes


def _load_gemma(torch, name: str, device: str):
    kwargs = {"trust_remote_code": True, "attn_implementation": "eager"}
    last = None
    for cls in _model_classes():
        try:
            model = _from_pretrained(cls, name, torch, kwargs)
            return model.to(device).eval()
        except (ValueError, OSError, TypeError) as e:
            last = e
            continue
    if last is not None:
        raise last
    raise ImportError("transformers has no AutoModel* class to load Gemma-4")


def _encode_ids(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def engine_gemma_it_template(user: str) -> str:
    """Must match inference/src/chat.rs gemma4_it."""
    return f"<bos><|turn>user\n{user}<turn|>\n<|turn>model\n"


def _as_id_list(ids) -> list[int]:
    """Unwrap apply_chat_template(tokenize=True): list, tensor, or BatchEncoding."""
    if ids is None:
        return []
    # Gemma-4 returns a mapping {'input_ids': [...], 'attention_mask': ...}, not a list.
    if not isinstance(ids, (list, tuple, str, bytes)) and hasattr(ids, "get"):
        got = ids.get("input_ids")
        if got is None:
            got = ids.get("ids")
        if got is not None:
            ids = got
    elif hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if hasattr(ids, "detach"):
        ids = ids.detach()
    if hasattr(ids, "cpu"):
        ids = ids.cpu()
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(x) for x in ids]


def _gen(model, tok, prompt_ids, *, max_new: int, device: str):
    import torch

    ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    attn = torch.ones_like(ids)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gen_kw = dict(
        max_new_tokens=max_new,
        min_new_tokens=1,
        do_sample=False,
        pad_token_id=pad,
        eos_token_id=tok.eos_token_id,
    )
    t0 = time.perf_counter()
    with torch.no_grad():
        try:
            out = model.generate(ids, attention_mask=attn, **gen_kw)
        except TypeError:
            out = model.generate(ids, **gen_kw)
    ms = (time.perf_counter() - t0) * 1000
    seq = out.sequences if hasattr(out, "sequences") else out
    new_ids = seq[0, ids.shape[-1] :].tolist()
    return {
        "new_ids": new_ids,
        "n_new": len(new_ids),
        "text_skip_special": tok.decode(new_ids, skip_special_tokens=True),
        "text_raw": tok.decode(new_ids, skip_special_tokens=False),
        "latency_ms": round(ms, 1),
    }


def _alias_keys(name: str) -> list[str]:
    alts = [name]
    swaps = (
        ("model.language_model.", "model."),
        ("model.", "model.language_model."),
        ("language_model.model.", "model.language_model."),
        ("model.language_model.", "language_model.model."),
    )
    for src, dst in swaps:
        if name.startswith(src):
            alts.append(dst + name[len(src) :])
    return list(dict.fromkeys(alts))


# Chat diag only needs the text decoder. Full E2B Aria bundles also ship
# vision/audio towers (~thousands of tensors); reconstructing them dominates wall time.
_NON_TEXT_MARKERS = (
    "audio_tower",
    "vision_tower",
    "vision_model",
    "multi_modal_projector",
    "vision_projector",
    "audio_projector",
    "embed_audio",
    "embed_vision",
)


def _is_text_weight(name: str) -> bool:
    lower = name.lower()
    return not any(m in lower for m in _NON_TEXT_MARKERS)


def _inject_bundle(
    model,
    tensors: dict,
    hadamard_seed: int,
    *,
    text_only: bool = True,
) -> dict:
    """Copy reconstruct + raw 1D tensors into matching HF keys (with VL aliases).

    By default skip vision/audio towers — chat compare only needs the LM. Full
    E2B reconstruct of every tower is CPU-heavy and can take tens of minutes.
    """
    import numpy as np
    import torch

    from common import quant as quant_mod

    if text_only:
        skipped = [n for n in tensors if not _is_text_weight(n)]
        work = {n: t for n, t in tensors.items() if _is_text_weight(n)}
    else:
        skipped = []
        work = tensors

    sd = model.state_dict()
    n_quant = n_raw = n_shape = 0
    unmatched: list[str] = []
    n_total = len(work)
    t0 = time.perf_counter()
    print(
        f"injecting {n_total}/{len(tensors)} tensors "
        f"(text_only={text_only}, skipped_non_text={len(skipped)}; "
        "CPU reconstruct_weight) …",
        file=sys.stderr,
        flush=True,
    )
    with torch.no_grad():
        for i, (name, obj) in enumerate(work.items(), 1):
            if i == 1 or i % 25 == 0 or i == n_total:
                elapsed = time.perf_counter() - t0
                print(
                    f"  inject {i}/{n_total} ({elapsed:.0f}s) last={name}",
                    file=sys.stderr,
                    flush=True,
                )
            key = next((k for k in _alias_keys(name) if k in sd), None)
            if key is None:
                unmatched.append(name)
                continue
            t = sd[key]
            if isinstance(obj, quant_mod.QuantTensor):
                recon = quant_mod.reconstruct_weight(obj, hadamard_seed)
                if tuple(t.shape) != recon.shape:
                    n_shape += 1
                    unmatched.append(f"{name} shape {recon.shape} vs sd {tuple(t.shape)}")
                    continue
                t.copy_(torch.from_numpy(np.asarray(recon, dtype=np.float32)).to(dtype=t.dtype))
                n_quant += 1
            else:
                arr = np.asarray(obj)
                if tuple(t.shape) != arr.shape:
                    n_shape += 1
                    unmatched.append(f"{name} shape {arr.shape} vs sd {tuple(t.shape)}")
                    continue
                t.copy_(torch.from_numpy(arr.astype(np.float32, copy=False)).to(dtype=t.dtype))
                n_raw += 1
    print(
        f"inject done in {time.perf_counter() - t0:.1f}s "
        f"(quant={n_quant} raw={n_raw} unmatched={len(unmatched)})",
        file=sys.stderr,
        flush=True,
    )
    n_bundle_quant = sum(1 for v in tensors.values() if isinstance(v, quant_mod.QuantTensor))
    n_bundle_raw = len(tensors) - n_bundle_quant
    return {
        "n_injected_quant": n_quant,
        "n_injected_raw": n_raw,
        "n_injected": n_quant + n_raw,
        "n_bundle_quant": n_bundle_quant,
        "n_bundle_raw": n_bundle_raw,
        "n_shape_mismatch": n_shape,
        "n_unmatched": len(unmatched),
        "n_skipped_non_text": len(skipped),
        "text_only": text_only,
        "unmatched_sample": unmatched[:20],
        "skipped_non_text_sample": skipped[:12],
        "sd_n_tensors": len(sd),
    }


def _bundle_model_fields(model_cfg: dict) -> dict:
    return {
        "rope_theta": model_cfg.get("rope_theta"),
        "hidden_size": model_cfg.get("hidden_size"),
        "num_layers": model_cfg.get("num_layers"),
        "num_attention_heads": model_cfg.get("num_attention_heads"),
        "num_kv_heads": model_cfg.get("num_kv_heads"),
        "head_dim": model_cfg.get("head_dim"),
        "hidden_act": model_cfg.get("hidden_act"),
        "tie_word_embeddings": model_cfg.get("tie_word_embeddings"),
        "vocab_size": model_cfg.get("vocab_size"),
        "layer_types": model_cfg.get("layer_types"),
        "sliding_window": model_cfg.get("sliding_window"),
        "num_kv_shared_layers": model_cfg.get("num_kv_shared_layers"),
        "use_double_wide_mlp": model_cfg.get("use_double_wide_mlp"),
        "global_head_dim": model_cfg.get("global_head_dim"),
        "partial_rotary_factor": model_cfg.get("partial_rotary_factor"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True, help="Aria bundle dir")
    p.add_argument("--hf", default="google/gemma-4-E2B-it")
    p.add_argument("--device", default="auto", help="cuda | cpu | auto")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--user", default="Hello")
    p.add_argument("--report", default=None)
    p.add_argument(
        "--inject-all",
        action="store_true",
        help="Also reconstruct vision/audio towers (slow; default is text LM only)",
    )
    args = p.parse_args()

    device = _pick_device(args.device)
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as e:
        print(json.dumps({"status": "skipped", "reason": f"torch/transformers: {e}"}))
        return 0

    from common import bundle as bundle_mod
    from common.gen_compare import exact_prefix_match

    bundle_dir = Path(args.bundle).expanduser().resolve()
    cfg, tensors = bundle_mod.load_bundle(bundle_dir)
    model_cfg = cfg.get("model") or {}
    seed = int(cfg.get("hadamard_seed") or 0)

    tok = _load_tokenizer(AutoTokenizer, args.hf)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    messages = [{"role": "user", "content": args.user}]
    hf_prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    hf_ids_tokenize_true = _as_id_list(
        tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    )

    engine_text = engine_gemma_it_template(args.user)
    engine_ids = _encode_ids(tok, engine_text)
    engine_ids_no_bos = _encode_ids(tok, engine_text.removeprefix("<bos>"))
    hf_ids = _encode_ids(tok, hf_prompt)

    print(
        f"loading HF {args.hf} on {device} (eager attn, no hub CUDA kernels; "
        "loads the model twice — fp32 teacher + inject target) …",
        file=sys.stderr,
        flush=True,
    )
    print("  [1/2] fp32 teacher …", file=sys.stderr, flush=True)
    base = _load_gemma(torch, args.hf, device)
    print("  [2/2] fp32 shell for reconstruct inject …", file=sys.stderr, flush=True)
    quant_m = _load_gemma(torch, args.hf, device)
    inject = _inject_bundle(quant_m, tensors, seed, text_only=not args.inject_all)
    n_inj = int(inject["n_injected_quant"])

    print("generate: HF chat template on fp32 …", file=sys.stderr, flush=True)
    chat_base = _gen(base, tok, hf_ids, max_new=args.max_new_tokens, device=device)
    if n_inj > 0:
        print("generate: HF chat template on reconstruct …", file=sys.stderr, flush=True)
        chat_q = _gen(quant_m, tok, hf_ids, max_new=args.max_new_tokens, device=device)
        print("generate: engine gemma4_it template on reconstruct …", file=sys.stderr, flush=True)
        eng_q = _gen(quant_m, tok, engine_ids, max_new=args.max_new_tokens, device=device)
    else:
        chat_q = None
        eng_q = None

    france = "The capital of France is"
    france_ids = _encode_ids(tok, france)
    print("generate: France completion fp32 …", file=sys.stderr, flush=True)
    france_base = _gen(base, tok, france_ids, max_new=args.max_new_tokens, device=device)
    if n_inj > 0:
        print("generate: France completion reconstruct …", file=sys.stderr, flush=True)
        france_q = _gen(quant_m, tok, france_ids, max_new=args.max_new_tokens, device=device)
    else:
        france_q = None

    prefix_chat = (
        exact_prefix_match(chat_base["new_ids"], chat_q["new_ids"])
        if chat_q is not None
        else {"exact_prefix_len": 0, "exact_prefix_frac": 0.0, "exact_match": False}
    )
    prefix_fr = (
        exact_prefix_match(france_base["new_ids"], france_q["new_ids"])
        if france_q is not None
        else {"exact_prefix_len": 0, "exact_prefix_frac": 0.0, "exact_match": False}
    )

    template_match = hf_prompt == engine_text
    ids_match = hf_ids == engine_ids

    hints = []
    if n_inj == 0:
        hints.append(
            "INJECT: no QuantTensor names matched HF state_dict; reconstruct is not a teacher"
        )
    if not template_match or not ids_match:
        hints.append(
            "TEMPLATE: HF apply_chat_template != engine gemma_it; prompt_ids differ "
            "(check literal <bos> vs tokenizer bos_token)"
        )
    if hf_ids_tokenize_true and hf_ids_tokenize_true != hf_ids:
        hints.append(
            "TEMPLATE: apply_chat_template(tokenize=True) ids != encode(template, add_special=False)"
        )
    if engine_ids and len(engine_ids) != 10 and args.user == "Hello":
        hints.append(
            f"PROMPT_LEN: engine-template encode len={len(engine_ids)} "
            "(HF Gemma-4 Hello template is 10 ids; old <start_of_turn> encode was 28)"
        )
    if int(prefix_fr.get("exact_prefix_len") or 0) >= 4 and int(
        prefix_chat.get("exact_prefix_len") or 0
    ) == 0:
        hints.append("CHAT vs COMPLETION: recon is ok on France prompt; chat path diverges")
    if n_inj > 0 and int(prefix_chat.get("exact_prefix_len") or 0) >= 4:
        hints.append("QUANT+HF chat look ok; if aria-engine is still garbage → ENGINE_GRAPH")
    elif n_inj > 0 and int(prefix_fr.get("exact_prefix_len") or 0) < 2:
        hints.append("QUANT: reconstruct_weight greedy already diverges from fp32")

    report = {
        "side": "model",
        "status": "ok",
        "device": device,
        "cuda": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bundle": str(bundle_dir),
        "hf": args.hf,
        "tokenizer_source": args.hf,
        "hf_arch": type(base).__name__,
        "inject": inject,
        "n_injected": n_inj,
        "bundle_model": _bundle_model_fields(model_cfg),
        "format_version": cfg.get("format_version"),
        "hadamard_seed": seed,
        "user": args.user,
        "bos_token": getattr(tok, "bos_token", None),
        "bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id,
        "hf_chat_prompt": hf_prompt,
        "engine_chat_prompt": engine_text,
        "template_string_match": template_match,
        "prompt_ids_hf": hf_ids,
        "prompt_ids_hf_tokenize_true": hf_ids_tokenize_true,
        "prompt_ids_engine_template": engine_ids,
        "prompt_ids_engine_without_bos_prefix": engine_ids_no_bos,
        "prompt_ids_match": ids_match,
        "prompt_ids_len_hf": len(hf_ids),
        "prompt_ids_len_engine_template": len(engine_ids),
        "chat": {
            "fp32": chat_base,
            "reconstruct": chat_q,
            "reconstruct_on_engine_template": eng_q,
            **prefix_chat,
        },
        "completion_france": {
            "prompt": france,
            "prompt_ids": france_ids,
            "fp32": france_base,
            "reconstruct": france_q,
            **prefix_fr,
        },
        "hints": hints,
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        outp = Path(args.report).expanduser()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(text, encoding="utf-8")
        print(f"wrote {outp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
