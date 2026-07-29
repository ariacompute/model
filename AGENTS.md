# AGENTS.md — aria model

工程上下文入口。逐层展开：先看「概述/架构/目录」，动手时再看「规范/命令/进行中/注意」。

## 概述
`model` 仓库 = aria model 层：Python 侧模型量化工具与产物。
首期聚焦 **Hadamard 旋转预处理 + Lloyd-Max 码本量化**，目标模型
`gemma/gemma-4-e2b-it` 与 `qwen/qwen3.5-2b`。产物为 Aria 风格 bundle
（`weight.bin` + `config.json` + tokenizer），供后续 `engine` 对接；本阶段以 Python
反量化 roundtrip 验收。

## 架构
`hf_utils`(拉取/流式) → `hadamard` → `codebook`/`quant` → `pack`/`bundle` → 家族脚本。
位宽：1–4 bit；混合精度 `--bits 2.54` / `3.26`。默认 `group_size=32`。

## 目录
- `common/`：共享内核（errors / hadamard / codebook / quant / pack / bundle / hf_utils / cli）
- `gemma/gemma-4-e2b-it/`：`quantize.py` + `config.yaml`
- `qwen/qwen3.5-2b/`：同上
- `tests/`：unittest（Hadamard、码本、pack、混合精度、bundle roundtrip、`--tiny`）
- 根：`AGENTS.md` / `requirements.md` / `task.md` / `README.md` / `requirements.txt` / `pyproject.toml`

## 开发规范
- Python ≥3.10；核心算法仅依赖 numpy；HF 路径可选 torch/transformers/safetensors/huggingface_hub。
- 统一异常 `ModelError` + 派生；禁止 `except: pass` 静默失败。
- 新增功能须同步单测（正常 + 异常）；权重产物目录不入 Git。
- Harness：半天以上需求须有 `requirements.md`（人工审核）→ `task.md` → 编码；严禁无 Spec Coding。
- AGENTS.md ≤100 行；细节下沉 `requirements.md` / 各目录 README。

## 常用命令
- `pip install -r requirements.txt`
- `python -m unittest discover -s tests -t .`
- `python gemma/gemma-4-e2b-it/quantize.py --tiny --bits 4`
- `python qwen/qwen3.5-2b/quantize.py --bits 2.54 --out ./out/qwen_q254`
- `python gemma/gemma-4-e2b-it/quantize.py --model google/gemma-4-E2B-it --bits 4`

## 进行中需求
见 `task.md`。Spec 见 `requirements.md`（Hadamard + 码本 + Aria bundle）。

## 注意事项
- 黄金路径：q4 tiny bundle 写出 → `load_bundle` → `dequantize` 重建误差有界。
- 非 2 幂行维：pad 到下一 2 幂再裁回，meta 记录 pad。
- 不做剪枝/蒸馏/GGUF；不做 embedding 专用标量路径 / 跨层可学习旋转吸收；不动 live `engine`。
