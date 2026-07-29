# aria model

[English](README.md) | [中文](README_cn.md)

Python toolkit for **Hadamard rotation + Lloyd-Max codebook quantization** for on-device LLM weights.

Target models:

| Family | Script | Default HuggingFace id |
|--------|--------|------------------------|
| Gemma 4 E2B IT | [`gemma/gemma-4-e2b-it/quantize.py`](gemma/gemma-4-e2b-it/quantize.py) | `google/gemma-4-E2B-it` |
| Qwen 3.5 2B | [`qwen/qwen3.5-2b/quantize.py`](qwen/qwen3.5-2b/quantize.py) | `Qwen/Qwen3.5-2B` |

Output is an Aria-style bundle: `weight.bin` + `config.json` (+ tokenizer files when downloading from HF). Spec: [`requirements.md`](requirements.md). Agent index: [`AGENTS.md`](AGENTS.md).

## Setup

```bash
# from repo root
uv venv .venv && source .venv/bin/activate   # or: python3 -m venv .venv
uv pip install -r requirements.txt           # or: pip install -r requirements.txt
# optional, for full HF model load path:
# uv pip install torch transformers
```

Core quant path needs **numpy** (+ **pyyaml** for `config.yaml`). Real HF downloads also need **safetensors** and **huggingface_hub**.

## CLI flags (both scripts)

| Flag | Description |
|------|-------------|
| `--bits` | `1` / `2` / `3` / `4`, or mixed `2.54` / `3.26` |
| `--model` | Override HF repo id (default from `config.yaml`) |
| `--group-size` | Codebook group size (default `32`) |
| `--seed` | Hadamard randomization seed (default `0`) |
| `--out` | Output directory (default `…/weights/qN` under the family folder) |
| `--tiny` | Synthetic tiny checkpoint — **no network** |
| `--config` | Path to alternate `config.yaml` |

## Gemma (`gemma-4-e2b-it`)

Config: [`gemma/gemma-4-e2b-it/config.yaml`](gemma/gemma-4-e2b-it/config.yaml).

```bash
# Smoke test (offline)
python gemma/gemma-4-e2b-it/quantize.py --tiny --bits 4

# 4-bit from default HF model (downloads weights)
python gemma/gemma-4-e2b-it/quantize.py --bits 4

# Mixed precision ~2.54 bit average
python gemma/gemma-4-e2b-it/quantize.py --bits 2.54 --out ./out/gemma_q254

# Mixed precision ~3.26 bit average
python gemma/gemma-4-e2b-it/quantize.py --bits 3.26 --out ./out/gemma_q326

# Override model id / seed / group size
python gemma/gemma-4-e2b-it/quantize.py \
  --model google/gemma-4-E2B-it \
  --bits 4 \
  --group-size 32 \
  --seed 0 \
  --out ./out/gemma_q4
```

## Qwen (`qwen3.5-2b`)

Config: [`qwen/qwen3.5-2b/config.yaml`](qwen/qwen3.5-2b/config.yaml).

```bash
# Smoke test (offline)
python qwen/qwen3.5-2b/quantize.py --tiny --bits 4

# 4-bit from default HF model
python qwen/qwen3.5-2b/quantize.py --bits 4

# Mixed precision
python qwen/qwen3.5-2b/quantize.py --bits 2.54 --out ./out/qwen_q254
python qwen/qwen3.5-2b/quantize.py --bits 3.26 --out ./out/qwen_q326

# Override model id
python qwen/qwen3.5-2b/quantize.py \
  --model Qwen/Qwen3.5-2B \
  --bits 4 \
  --out ./out/qwen_q4
```

## Output layout

Example for `--out ./out/gemma_q4`:

```
out/gemma_q4/
  config.json      # format aria-quant-bundle, per-tensor offsets / bits
  weight.bin       # packed indices + fp16 codebook / scales / norms
  tokenizer.*      # copied from HF when not using --tiny
```

Load and dequantize in Python:

```python
from common.bundle import load_bundle
from common.quant import dequantize

cfg, tensors = load_bundle("./out/gemma_q4")
W = dequantize(tensors["blk.0.attn_q.weight"])  # rotated-space reconstruction
```

## Tests

```bash
python -m unittest discover -s tests -t .
```

## Notes

- Only **2D** weights are codebook-quantized; 1D tensors (e.g. RMSNorm) are stored as raw fp16.
- Weight product dirs (`**/weights/`, `*.bin`) are gitignored — do not commit multi-GB artifacts.
- This repo does **not** emit GGUF; live `engine` quant loaders are out of scope for now.
