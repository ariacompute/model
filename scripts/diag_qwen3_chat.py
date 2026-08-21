#!/usr/bin/env python3
"""GPU diagnostic: HF fp32 vs bundle reconstruct_weight on the same Qwen3 chat prompt.

Isolates *quant* vs *chat-template* (engine.log Hello garbage). Uses official
Qwen3 apply_chat_template (enable_thinking=False) plus the engine hardcoded
ChatML string so prompt ids can be compared.

H200 example (from model repo root):

  pip install torch transformers
  python scripts/diag_qwen3_chat.py \\
    --bundle ~/.ariacompute/models/qwen3-0.6b_q4 \\
    --hf Qwen/Qwen3-0.6B \\
    --device cuda \\
    --report ./out/model_diag_qwen3.json

Tokenizer is loaded from --hf (not the Aria bundle) so transformers does not
trip the Mistral regex warning. Attention uses eager kernels (no JIT CUDA /
Python.h). If a compile error still appears: sudo apt install python3-dev.
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
    """Load from HF repo (not the Aria bundle). Bundle tokenizer.json can trip
    transformers' Mistral regex warning and tokenize incorrectly."""
    kwargs = {"trust_remote_code": True}
    try:
        return AutoTokenizer.from_pretrained(name, fix_mistral_regex=True, **kwargs)
    except TypeError:
        return AutoTokenizer.from_pretrained(name, **kwargs)


def _load_causal_lm(AutoModelForCausalLM, torch, name: str, device: str):
    kwargs = {"trust_remote_code": True, "attn_implementation": "eager"}
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, **kwargs)
    except TypeError:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch.float32, **kwargs
            )
        except (TypeError, ValueError):
            kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch.float32, **kwargs
            )
    except ValueError:
        kwargs.pop("attn_implementation", None)
        try:
            model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, **kwargs)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch.float32, **kwargs
            )
    return model.to(device).eval()


def _encode_ids(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def _engine_template(user: str) -> str:
    return (
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _gen(model, tok, prompt_ids, *, max_new: int, device: str):
    import torch

    ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new,
            min_new_tokens=1,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    ms = (time.perf_counter() - t0) * 1000
    new_ids = out[0, ids.shape[-1] :].tolist()
    return {
        "new_ids": new_ids,
        "n_new": len(new_ids),
        "text_skip_special": tok.decode(new_ids, skip_special_tokens=True),
        "text_raw": tok.decode(new_ids, skip_special_tokens=False),
        "latency_ms": round(ms, 1),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True, help="Aria bundle dir")
    p.add_argument("--hf", default="Qwen/Qwen3-0.6B")
    p.add_argument("--device", default="auto", help="cuda | cpu | auto")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--user", default="Hello")
    p.add_argument("--report", default=None)
    args = p.parse_args()

    device = _pick_device(args.device)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(json.dumps({"status": "skipped", "reason": f"torch/transformers: {e}"}))
        return 0

    from common import bundle as bundle_mod
    from common.gen_compare import _inject_bundle_weights, exact_prefix_match

    bundle_dir = Path(args.bundle).expanduser().resolve()
    cfg, tensors = bundle_mod.load_bundle(bundle_dir)
    model_cfg = cfg.get("model") or {}
    seed = int(cfg.get("hadamard_seed") or 0)

    tok = _load_tokenizer(AutoTokenizer, args.hf)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    messages = [{"role": "user", "content": args.user}]
    try:
        hf_prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        hf_prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    think_on = None
    try:
        think_on = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        pass

    engine_text = _engine_template(args.user)
    engine_ids = _encode_ids(tok, engine_text)
    hf_ids = _encode_ids(tok, hf_prompt)

    print(
        f"loading HF {args.hf} on {device} (eager attn, no hub CUDA kernels; "
        "loads the model twice — fp32 teacher + inject target) …",
        file=sys.stderr,
        flush=True,
    )
    print("  [1/2] fp32 teacher …", file=sys.stderr, flush=True)
    base = _load_causal_lm(AutoModelForCausalLM, torch, args.hf, device)
    print("  [2/2] fp32 shell for reconstruct inject …", file=sys.stderr, flush=True)
    quant_m = _load_causal_lm(AutoModelForCausalLM, torch, args.hf, device)
    n_inj = _inject_bundle_weights(quant_m, tensors, seed, progress=True)

    print("generate: HF chat template on fp32 …", file=sys.stderr, flush=True)
    chat_base = _gen(base, tok, hf_ids, max_new=args.max_new_tokens, device=device)
    print("generate: HF chat template on reconstruct …", file=sys.stderr, flush=True)
    chat_q = _gen(quant_m, tok, hf_ids, max_new=args.max_new_tokens, device=device)
    print("generate: engine ChatML template on reconstruct …", file=sys.stderr, flush=True)
    eng_q = _gen(
        quant_m, tok, engine_ids, max_new=args.max_new_tokens, device=device
    )

    france = "The capital of France is"
    france_ids = _encode_ids(tok, france)
    print("generate: France completion fp32 …", file=sys.stderr, flush=True)
    france_base = _gen(base, tok, france_ids, max_new=args.max_new_tokens, device=device)
    print("generate: France completion reconstruct …", file=sys.stderr, flush=True)
    france_q = _gen(quant_m, tok, france_ids, max_new=args.max_new_tokens, device=device)

    prefix_chat = exact_prefix_match(chat_base["new_ids"], chat_q["new_ids"])
    prefix_fr = exact_prefix_match(france_base["new_ids"], france_q["new_ids"])

    template_match = hf_prompt == engine_text
    ids_match = hf_ids == engine_ids

    hints = []
    if not template_match or not ids_match:
        hints.append("TEMPLATE: HF apply_chat_template != engine ChatML; prompt_ids differ")
    if int(prefix_fr.get("exact_prefix_len") or 0) >= 4 and int(
        prefix_chat.get("exact_prefix_len") or 0
    ) == 0:
        hints.append("CHAT vs COMPLETION: recon is ok on France prompt; chat path diverges")
    if int(prefix_chat.get("exact_prefix_len") or 0) >= 4:
        hints.append("QUANT+HF chat look ok; if aria-engine is still garbage → ENGINE_GRAPH")
    elif int(prefix_fr.get("exact_prefix_len") or 0) < 2:
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
        "n_injected": n_inj,
        "bundle_model": {
            "rope_theta": model_cfg.get("rope_theta"),
            "hidden_size": model_cfg.get("hidden_size"),
            "num_layers": model_cfg.get("num_layers"),
            "num_attention_heads": model_cfg.get("num_attention_heads"),
            "num_kv_heads": model_cfg.get("num_kv_heads"),
            "head_dim": model_cfg.get("head_dim"),
            "hidden_act": model_cfg.get("hidden_act"),
            "tie_word_embeddings": model_cfg.get("tie_word_embeddings"),
            "vocab_size": model_cfg.get("vocab_size"),
        },
        "format_version": cfg.get("format_version"),
        "hadamard_seed": seed,
        "user": args.user,
        "hf_chat_prompt": hf_prompt,
        "engine_chat_prompt": engine_text,
        "hf_thinking_on_prompt": think_on,
        "template_string_match": template_match,
        "prompt_ids_hf": hf_ids,
        "prompt_ids_engine_template": engine_ids,
        "prompt_ids_match": ids_match,
        "eos_token_id": tok.eos_token_id,
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
