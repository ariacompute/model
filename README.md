# aria model

[English](README.md) | [中文](README_cn.md)

Python toolkit for **Hadamard rotation + Lloyd-Max codebook quantization** for on-device LLM weights.

Supported families (Qwen / Gemma / LFM / Nanbeige / Bonsai / Inkling / OpenVLA / OpenPI / LingBot):
full table in [`requirements.md` §1.1](requirements.md). Each folder has `quantize.py` + `config.yaml`.

Output is an Aria-style bundle: `weight.bin` + `config.json` (+ tokenizer). Spec:
[`requirements.md`](requirements.md). Agent index: [`AGENTS.md`](AGENTS.md).

**int4** = `--bits 4` (codebook K=16). **int8** = `--bits 8` (codebook K=256, Hadamard+Lloyd-Max only).
VL models quantize vision towers by default. Weights must be **safetensors** (GGUF not parsed).

## Setup

```bash
# from repo root
uv venv .venv && source .venv/bin/activate   # or: python3 -m venv .venv
uv pip install -r requirements.txt           # or: pip install -r requirements.txt
# optional HF extras (CPU torch is enough for transformers helpers):
# uv pip install transformers

export HF_TOKEN=...   # higher Hub rate limits (recommended for full downloads)
```

### CUDA PyTorch (GPU Lloyd-Max)

`codebook_share=group` / `channel` use GPU when `torch.cuda.is_available()` **and** the wheel
includes your GPU arch. Pick the index by GPU:

| Host GPU | Arch | Install |
|----------|------|---------|
| **H200** (Hopper) | `sm_90` | `cu124` or `cu128` |
| **RTX PRO 6000** (Blackwell Server Edition) | **`sm_120`** | **`cu128` only** (PyTorch ≥ 2.7) — **not** `cu124` |

```bash
# H200 (Hopper) — either is fine
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
# or: uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# RTX PRO 6000 Blackwell — must use CUDA 12.8+ wheels (sm_120)
uv pip uninstall torch torchvision torchaudio   # remove any cu124 install first
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
# if needed: uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

Verify before quantizing (especially on Blackwell):

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("capability", torch.cuda.get_device_capability(0))
print("arch", torch.cuda.get_arch_list())
print("ok", torch.randn(4, device="cuda").sum().item())
PY
```

On RTX PRO 6000, `arch` must list **`sm_120`**. If you see
`no kernel image is available for execution on the device`, you still have a `cu124` (or older)
build — reinstall with `cu128`.

Core quant path needs **numpy** (+ **pyyaml** for `config.yaml`). Real HF downloads also need **safetensors** and **huggingface_hub**.

### Recommended hosts (reference)

Validated targets for full-model runs (either box is fine; VRAM drives GPU batch size):

| | **H200** | **RTX PRO 6000** |
|--|----------|------------------|
| CPU | **16 vCPUs** | **24 vCPUs** |
| Host RAM | **200 GiB** | **218 GiB** |
| GPU | **1× NVIDIA H200 NVLink** (Hopper) | **1× NVIDIA RTX PRO 6000** (Blackwell, `sm_120`) |
| GPU memory | **141 GiB** | **96 GiB** |
| Suggested `--workers` | 16 | 24 |
| PyTorch CUDA wheel | `cu124` or `cu128` | **`cu128`** (not `cu124`) |

With CUDA torch installed, `codebook_share=group` uses batched GPU Lloyd-Max sized from device VRAM; CPU fallback uses `--workers` (default `min(32, cpu_count)`).

## CLI flags

| Flag | Description |
|------|-------------|
| `--bits` | `1` / `2` / `3` / `4` / `8` (int8 codebook), or mixed `1.5` / `2.54` / `3.26` |
| `--model` | Override HF repo id (default from `config.yaml`) |
| `--group-size` | Codebook group size (default `32`) |
| `--seed` | Hadamard randomization seed (default `0`) |
| `--out` | **Required in examples below** — output bundle directory |
| `--codebook-share` | `group` (default, small) or `channel` (larger, higher fidelity) |
| `--ple-bits` / `--compute-bits` / `--hi-bits` | Overrides for `--bits 1.5` only (defaults 1 / 2 / 3) |
| `--workers` | Parallel group workers (default: CPU count, max 32) |
| `--tiny` | Synthetic tiny checkpoint — **no network** |
| `--config` | Path to alternate `config.yaml` |

## Output path convention

Always pass `--out`. Use a single root `./out/` and this name pattern:

```text
./out/<model-slug>_<quant>
```

| Part | Rule | Examples |
|------|------|----------|
| `<model-slug>` | Same as the family folder name | `gemma-4-e2b-it`, `qwen3.5-2b`, `lfm2.5-1.2b-instruct` |
| `<quant>` | Bit label with `.` removed | `q4`, `q8`, `q15`, `q254`, `q326` |

Examples: `./out/gemma-4-e2b-it_q4`, `./out/qwen3.5-2b_q8`, `./out/gemma-4-e2b-it_q15`.

Bundle layout:

```text
out/gemma-4-e2b-it_q4/
  config.json
  weight.bin
  tokenizer.*      # when not using --tiny
```

```python
from common.bundle import load_bundle
from common.quant import dequantize

cfg, tensors = load_bundle("./out/gemma-4-e2b-it_q4")
W = dequantize(tensors["blk.0.attn_q.weight"])  # rotated-space reconstruction
```

## Offline smoke (`--tiny`)

```bash
python gemma/gemma-4-e2b-it/quantize.py --tiny --bits 4 --out ./out/gemma-4-e2b-it_tiny_q4
python gemma/gemma-4-e2b-it/quantize.py --tiny --bits 8 --out ./out/gemma-4-e2b-it_tiny_q8
python qwen/qwen3.5-2b/quantize.py --tiny --bits 4 --out ./out/qwen3.5-2b_tiny_q4
python lfm/lfm2-350m/quantize.py --tiny --bits 4 --out ./out/lfm2-350m_tiny_q4
```

## Full-model commands (int4 + int8)

All commands download from the default `base_model` in each `config.yaml` unless `--model` is set.
Add `--workers 16` (H200) or `--workers 24` (RTX PRO 6000) on large hosts as needed.

### Qwen

```bash
# qwen3-0.6b
python qwen/qwen3-0.6b/quantize.py --bits 4 --out ./out/qwen3-0.6b_q4
python qwen/qwen3-0.6b/quantize.py --bits 8 --out ./out/qwen3-0.6b_q8

# qwen3-1.7b
python qwen/qwen3-1.7b/quantize.py --bits 4 --out ./out/qwen3-1.7b_q4
python qwen/qwen3-1.7b/quantize.py --bits 8 --out ./out/qwen3-1.7b_q8

# qwen3.5-0.8b
python qwen/qwen3.5-0.8b/quantize.py --bits 4 --out ./out/qwen3.5-0.8b_q4
python qwen/qwen3.5-0.8b/quantize.py --bits 8 --out ./out/qwen3.5-0.8b_q8

# qwen3.5-2b
python qwen/qwen3.5-2b/quantize.py --bits 4 --out ./out/qwen3.5-2b_q4
python qwen/qwen3.5-2b/quantize.py --bits 8 --out ./out/qwen3.5-2b_q8
```

### Gemma

```bash
# gemma-3-270m-it
python gemma/gemma-3-270m-it/quantize.py --bits 4 --out ./out/gemma-3-270m-it_q4
python gemma/gemma-3-270m-it/quantize.py --bits 8 --out ./out/gemma-3-270m-it_q8

# gemma-3-1b-it
python gemma/gemma-3-1b-it/quantize.py --bits 4 --out ./out/gemma-3-1b-it_q4
python gemma/gemma-3-1b-it/quantize.py --bits 8 --out ./out/gemma-3-1b-it_q8

# gemma-3n-e2b-it
python gemma/gemma-3n-e2b-it/quantize.py --bits 4 --out ./out/gemma-3n-e2b-it_q4
python gemma/gemma-3n-e2b-it/quantize.py --bits 8 --out ./out/gemma-3n-e2b-it_q8

# gemma-3n-e4b-it
python gemma/gemma-3n-e4b-it/quantize.py --bits 4 --out ./out/gemma-3n-e4b-it_q4
python gemma/gemma-3n-e4b-it/quantize.py --bits 8 --out ./out/gemma-3n-e4b-it_q8

# gemma-4-e2b-it
python gemma/gemma-4-e2b-it/quantize.py --bits 4 --out ./out/gemma-4-e2b-it_q4
python gemma/gemma-4-e2b-it/quantize.py --bits 8 --out ./out/gemma-4-e2b-it_q8

# gemma-4-e4b-it
python gemma/gemma-4-e4b-it/quantize.py --bits 4 --out ./out/gemma-4-e4b-it_q4
python gemma/gemma-4-e4b-it/quantize.py --bits 8 --out ./out/gemma-4-e4b-it_q8
```

### LFM

```bash
# lfm2-350m
python lfm/lfm2-350m/quantize.py --bits 4 --out ./out/lfm2-350m_q4
python lfm/lfm2-350m/quantize.py --bits 8 --out ./out/lfm2-350m_q8

# lfm2-700m
python lfm/lfm2-700m/quantize.py --bits 4 --out ./out/lfm2-700m_q4
python lfm/lfm2-700m/quantize.py --bits 8 --out ./out/lfm2-700m_q8

# lfm2-1.2b
python lfm/lfm2-1.2b/quantize.py --bits 4 --out ./out/lfm2-1.2b_q4
python lfm/lfm2-1.2b/quantize.py --bits 8 --out ./out/lfm2-1.2b_q8

# lfm2-2.6b
python lfm/lfm2-2.6b/quantize.py --bits 4 --out ./out/lfm2-2.6b_q4
python lfm/lfm2-2.6b/quantize.py --bits 8 --out ./out/lfm2-2.6b_q8

# lfm2-8b-a1b
python lfm/lfm2-8b-a1b/quantize.py --bits 4 --out ./out/lfm2-8b-a1b_q4
python lfm/lfm2-8b-a1b/quantize.py --bits 8 --out ./out/lfm2-8b-a1b_q8

# lfm2-vl-450m (vision included)
python lfm/lfm2-vl-450m/quantize.py --bits 4 --out ./out/lfm2-vl-450m_q4
python lfm/lfm2-vl-450m/quantize.py --bits 8 --out ./out/lfm2-vl-450m_q8

# lfm2.5-350m
python lfm/lfm2.5-350m/quantize.py --bits 4 --out ./out/lfm2.5-350m_q4
python lfm/lfm2.5-350m/quantize.py --bits 8 --out ./out/lfm2.5-350m_q8

# lfm2.5-1.2b-instruct
python lfm/lfm2.5-1.2b-instruct/quantize.py --bits 4 --out ./out/lfm2.5-1.2b-instruct_q4
python lfm/lfm2.5-1.2b-instruct/quantize.py --bits 8 --out ./out/lfm2.5-1.2b-instruct_q8

# lfm2.5-1.2b-thinking
python lfm/lfm2.5-1.2b-thinking/quantize.py --bits 4 --out ./out/lfm2.5-1.2b-thinking_q4
python lfm/lfm2.5-1.2b-thinking/quantize.py --bits 8 --out ./out/lfm2.5-1.2b-thinking_q8

# lfm2.5-2.6b
python lfm/lfm2.5-2.6b/quantize.py --bits 4 --out ./out/lfm2.5-2.6b_q4
python lfm/lfm2.5-2.6b/quantize.py --bits 8 --out ./out/lfm2.5-2.6b_q8

# lfm2.5-vl-1.6b (vision included)
python lfm/lfm2.5-vl-1.6b/quantize.py --bits 4 --out ./out/lfm2.5-vl-1.6b_q4
python lfm/lfm2.5-vl-1.6b/quantize.py --bits 8 --out ./out/lfm2.5-vl-1.6b_q8
```

### Nanbeige / Bonsai / Inkling

```bash
# nanbeige4.2-3b
python nanbeige/nanbeige4.2-3b/quantize.py --bits 4 --out ./out/nanbeige4.2-3b_q4
python nanbeige/nanbeige4.2-3b/quantize.py --bits 8 --out ./out/nanbeige4.2-3b_q8

# bonsai-27b (BF16 safetensors ~54GB source; large output)
python bonsai/bonsai-27b/quantize.py --bits 4 --workers 16 --out ./out/bonsai-27b_q4
python bonsai/bonsai-27b/quantize.py --bits 8 --workers 16 --out ./out/bonsai-27b_q8

# inkling-small
python inkling/inkling-small/quantize.py --bits 4 --out ./out/inkling-small_q4
python inkling/inkling-small/quantize.py --bits 8 --out ./out/inkling-small_q8
```

### OpenVLA / OpenPI / LingBot (VLA; vision + action heads included)

```bash
# openvla-7b  (HF: openvla/openvla-7b)
python openvla/openvla-7b/quantize.py --bits 4 --out ./out/openvla-7b_q4
python openvla/openvla-7b/quantize.py --bits 8 --out ./out/openvla-7b_q8

# openpi-pi0-3b  (HF: lerobot/pi0_base)
python openpi/openpi-pi0-3b/quantize.py --bits 4 --out ./out/openpi-pi0-3b_q4
python openpi/openpi-pi0-3b/quantize.py --bits 8 --out ./out/openpi-pi0-3b_q8

# openpi-pi0.5-3b  (HF: lerobot/pi05_base)
python openpi/openpi-pi0.5-3b/quantize.py --bits 4 --out ./out/openpi-pi0.5-3b_q4
python openpi/openpi-pi0.5-3b/quantize.py --bits 8 --out ./out/openpi-pi0.5-3b_q8

# lingbot-vla-v2-6b  (HF: robbyant/lingbot-vla-v2-6b)
python lingbot/lingbot-vla-v2-6b/quantize.py --bits 4 --out ./out/lingbot-vla-v2-6b_q4
python lingbot/lingbot-vla-v2-6b/quantize.py --bits 8 --out ./out/lingbot-vla-v2-6b_q8
```

## Optional mixed-precision commands

```bash
# PLE-weighted (target <1GB on Gemma-4-E2B)
python gemma/gemma-4-e2b-it/quantize.py --bits 1.5 --workers 16 --out ./out/gemma-4-e2b-it_q15
python gemma/gemma-4-e4b-it/quantize.py --bits 1.5 --workers 16 --out ./out/gemma-4-e4b-it_q15
python gemma/gemma-3n-e2b-it/quantize.py --bits 1.5 --workers 16 --out ./out/gemma-3n-e2b-it_q15
python gemma/gemma-3n-e4b-it/quantize.py --bits 1.5 --workers 16 --out ./out/gemma-3n-e4b-it_q15

# Layer-count mixed (2.54 / 3.26)
python qwen/qwen3.5-2b/quantize.py --bits 2.54 --out ./out/qwen3.5-2b_q254
python qwen/qwen3.5-2b/quantize.py --bits 3.26 --out ./out/qwen3.5-2b_q326
python gemma/gemma-4-e2b-it/quantize.py --bits 2.54 --out ./out/gemma-4-e2b-it_q254
python gemma/gemma-4-e2b-it/quantize.py --bits 3.26 --out ./out/gemma-4-e2b-it_q326
```

## Quality audit

```bash
# A) Stratified layer RMSE (offline vs tiny ref — only for --tiny bundles)
python -m common.audit_cli layer \
  --bundle ./out/gemma-4-e2b-it_tiny_q4 \
  --ref tiny \
  --sample 8 \
  --family gemma-4-e2b-it \
  --report ./out/gemma-4-e2b-it_tiny_q4/audit_layer.json

# A) Against HF reference weights (full quant bundles; --family resolves base_model)
python -m common.audit_cli layer \
  --bundle ./out/qwen3.5-2b_q4 \
  --family qwen3.5-2b \
  --sample 8 \
  --report ./out/qwen3.5-2b_q4/audit_layer.json
# or explicitly: --model Qwen/Qwen3.5-2B

# B) Text short generation compare (Qwen / Gemma / LFM / Inkling / …)
python -m common.audit_cli gen \
  --bundle ./out/qwen3.5-2b_q4 \
  --model Qwen/Qwen3.5-2B \
  --kind text \
  --max-new-tokens 32 \
  --report ./out/qwen3.5-2b_q4/audit_gen.json

# B) VLA forward / param-drift compare (OpenVLA / OpenPI / LingBot)
python -m common.audit_cli gen \
  --bundle ./out/openvla-7b_q4 \
  --model openvla/openvla-7b \
  --kind vla \
  --report ./out/openvla-7b_q4/audit_gen.json
```

Report thresholds (orig-space `rel_rmse_orig`): q8 ≤ 0.15, q4 ≤ 0.35, other/mixed ≤ 0.50 (embed/PLE ≤ 0.80). Stored as `pass` per layer only.

## Tests

```bash
python -m unittest discover -s tests -t .
```

## Notes

- Streaming I/O only limits peak memory; **every 2D weight** still runs full
  **Hadamard rotation (`H @ W`, axis=0)** then **Lloyd-Max codebook quantization**.
  Large matrices use column-chunked FWHT (same math as one-shot `H @ W`), never skip rotation.
- Default **`--codebook-share group`**: one codebook per row-group (shared across channels).
  With CUDA + torch, group Lloyd-Max uses batched GPU (`lloyd_max_batched_torch`); otherwise
  CPU numpy (+ `--workers`). Use `--codebook-share channel` for higher fidelity (larger `weight.bin`;
  also GPU when available).
- **`--bits 1.5`**: PLE / large embeddings default **1-bit**, compute layers **2–3 bit**,
  **parameter-weighted** average in ~1.35–1.55; Gemma-4-E2B target `weight.bin` **&lt; 1 GB**.
- Only **2D** weights are codebook-quantized; 1D tensors (e.g. RMSNorm) are stored as raw fp16.
- Product dirs (`./out/`, `**/weights/`, `*.bin`) are gitignored — do not commit multi-GB artifacts.
- This repo does **not** emit GGUF; live `engine` quant loaders are out of scope for now.

## Engineering Conventions

This repository follows the Harness Engineering philosophy:

- [`AGENTS.md`](AGENTS.md): Agent engineering context entry and directory index
- [`requirements.md`](requirements.md): Requirements spec (feature boundaries/exceptions/acceptance criteria, human-review-gated)
- [`task.md`](task.md): Implementation task checklist
