# AGENTS.md — aria model

工程上下文入口。逐层展开：先看「概述/架构/目录」，动手时再看「规范/命令/进行中/注意」。

## 概述
`model` 仓库 = aria model 层：Python 侧 **Hadamard + Lloyd-Max 码本量化**。
多家族（Qwen / Gemma / LFM / Nanbeige / Bonsai / Inkling）薄封装；产物 Aria bundle
（`weight.bin` + `config.json` + tokenizer）；Python 反量化 roundtrip 验收。

## 架构
`hf_utils` → `hadamard` → `codebook`/`quant` → `pack`/`bundle` → 家族 `quantize.py`。
位宽：1–4 / **8**（int8=码本 K=256）；混合 `1.5` / `2.54` / `3.26`。
默认 `group_size=32`、`codebook_share=group`；VL **默认量化 vision**。

## 目录
- `common/`：errors / hadamard / codebook / quant / pack / bundle / hf_utils / cli / runtime
- `qwen/` `gemma/` `lfm/` `nanbeige/` `bonsai/` `inkling/`：各模型 `quantize.py` + `config.yaml`
- `tests/`：unittest；家族表见 `requirements.md` §1.1
- 根：`AGENTS.md` / `requirements.md` / `task.md` / `README.md` / `requirements.txt`

## 开发规范
- Python ≥3.10；核心仅 numpy；HF 路径可选 torch/transformers/safetensors/huggingface_hub。
- 统一 `ModelError` 派生；禁止 `except: pass`；权重产物不入 Git。
- Harness：半天以上须 `requirements.md`（人审）→ `task.md` → 编码。
- AGENTS.md ≤100 行；家族清单下沉 `requirements.md`。

## 常用命令
- `pip install -r requirements.txt`
- `python -m unittest discover -s tests -t .`
- `python gemma/gemma-4-e2b-it/quantize.py --tiny --bits 4`
- `python gemma/gemma-4-e2b-it/quantize.py --bits 8`          # int8 codebook
- `python gemma/gemma-4-e2b-it/quantize.py --bits 1.5 --workers 16`
- `python qwen/qwen3.5-2b/quantize.py --bits 4`
- `python lfm/lfm2-350m/quantize.py --tiny --bits 4`

## 进行中需求
见 `task.md`。Spec 见 `requirements.md`。

## 注意事项
- 黄金路径：q4/q8 tiny → `load_bundle` → `dequantize` 误差有界。
- 流式不得跳过 Hadamard/Lloyd-Max；需 **safetensors**（不解析 GGUF）。
- 不做剪枝/蒸馏/对称 int8 旁路；不动 live `engine`/`serve`。
