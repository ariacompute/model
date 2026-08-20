# aria model

[English](README.md) | [中文](README_cn.md)

Python 端 **Hadamard 旋转预处理 + Lloyd-Max 码本量化** 工具，面向端侧 LLM 权重。

支持家族（Qwen / Gemma / LFM / Nanbeige / Bonsai / Inkling / OpenVLA / OpenPI / LingBot）完整表见
[`requirements.md` §1.1](requirements.md)。每目录含 `quantize.py` + `config.yaml`。

产物为 Aria bundle：`weight.bin` + `config.json`（+ tokenizer）。规格见
[`requirements.md`](requirements.md)，Agent 入口见 [`AGENTS.md`](AGENTS.md)。

**int4** = `--bits 4`（码本 K=16）。**int8** = `--bits 8`（码本 K=256，仅 Hadamard+Lloyd-Max）。
**q326+channel** = `--bits 3.26 --codebook-share channel`（推荐生成质量配方；产物后缀 `_q326_channel`）。
VL 默认量化 vision。权重须为 **safetensors**（不解析 GGUF）。

## 环境安装

```bash
# 在仓库根目录
uv venv .venv && source .venv/bin/activate   # 或: python3 -m venv .venv
uv pip install -r requirements.txt           # 或: pip install -r requirements.txt
# 可选 HF 辅助（audit gen 需要 torch + transformers；layer 审计不需要）：
uv pip install torch transformers

export HF_TOKEN=...   # 提高 Hub 限流（全量下载建议设置）
```

### CUDA 版 PyTorch（GPU Lloyd-Max）

`codebook_share=group` / `channel` 仅在 `torch.cuda.is_available()` **且** wheel 含本机 GPU 架构时走 GPU。
按显卡选择 index：

| 主机 GPU | 架构 | 安装 |
|----------|------|------|
| **H200**（Hopper） | `sm_90` | `cu124` 或 `cu128` |
| **RTX PRO 6000**（Blackwell Server Edition） | **`sm_120`** | **仅 `cu128`**（PyTorch ≥ 2.7）— **不要**用 `cu124` |

```bash
# H200（Hopper）— 两者均可
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
# 或: uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# RTX PRO 6000 Blackwell — 必须 CUDA 12.8+ wheel（sm_120）
uv pip uninstall torch torchvision torchaudio   # 先卸掉误装的 cu124
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
# 若仍有问题: uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

量化前自检（Blackwell 尤其重要）：

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("capability", torch.cuda.get_device_capability(0))
print("arch", torch.cuda.get_arch_list())
print("ok", torch.randn(4, device="cuda").sum().item())
PY
```

在 RTX PRO 6000 上，`arch` 须包含 **`sm_120`**。若出现
`no kernel image is available for execution on the device`，说明仍是 `cu124`（或更旧）构建，
请改用 `cu128` 重装。

核心量化依赖 **numpy**（读 `config.yaml` 还需 **pyyaml**）。真实 HF 下载另需 **safetensors**、**huggingface_hub**。

### 推荐主机（参考配置）

完整模型量化可用下列任一配置（显存决定 GPU 批大小）：

| | **H200** | **RTX PRO 6000** |
|--|----------|------------------|
| CPU | **16 vCPU** | **24 vCPU** |
| 主机内存 | **200 GiB** | **218 GiB** |
| GPU | **1× NVIDIA H200 NVLink**（Hopper） | **1× NVIDIA RTX PRO 6000**（Blackwell，`sm_120`） |
| 显存 | **141 GiB** | **96 GiB** |
| 建议 `--workers` | 16 | 24 |
| PyTorch CUDA wheel | `cu124` 或 `cu128` | **`cu128`**（不要用 `cu124`） |

已安装 CUDA 版 torch 时，`codebook_share=group` 使用按显存自适应批大小的 GPU Lloyd-Max；无 GPU 时回退 CPU（`--workers`，默认 `min(32, cpu_count)`）。

## 通用 CLI 参数

| 参数 | 说明 |
|------|------|
| `--bits` | `1` / `2` / `3` / `4` / `8`（int8 码本），或混合 `1.5` / `2.54` / `3.26` |
| `--model` | 覆盖 HF 仓库 id（默认来自 `config.yaml`） |
| `--group-size` | 码本分组大小（默认 `32`） |
| `--seed` | Hadamard 随机化种子（默认 `0`） |
| `--out` | **下文示例一律显式指定** — 输出 bundle 目录 |
| `--codebook-share` | `group`（默认，体积小）或 `channel`（更大、精度更高） |
| `--ple-bits` / `--compute-bits` / `--hi-bits` | 仅 `--bits 1.5`（默认 1 / 2 / 3） |
| `--workers` | 并行 group worker 数（默认 CPU 核数，上限 32） |
| `--tiny` | 使用合成 tiny checkpoint，**无需联网** |
| `--config` | 指定其它 `config.yaml` 路径 |

## 输出路径约定

请始终传入 `--out`。统一写到 `./out/`，命名规则：

```text
./out/<model-slug>_<quant>
```

| 部分 | 规则 | 示例 |
|------|------|------|
| `<model-slug>` | 与家族目录名相同 | `gemma-4-e2b-it`、`qwen3.5-2b`、`lfm2.5-1.2b-instruct` |
| `<quant>` | 位宽标签，去掉 `.`；channel 后缀保留 | `q4`、`q8`、`q15`、`q254`、`q326`、`q326_channel` |

示例：`./out/gemma-4-e2b-it_q4`、`./out/qwen3.5-2b_q8`、`./out/qwen3-0.6b_q326_channel`。

产物结构：

```text
out/gemma-4-e2b-it_q4/
  config.json
  weight.bin
  tokenizer.*      # 非 --tiny 时从 HF 拷贝
```

```python
from common.bundle import load_bundle
from common.quant import dequantize

cfg, tensors = load_bundle("./out/gemma-4-e2b-it_q4")
W = dequantize(tensors["blk.0.attn_q.weight"])  # 旋转空间重建
```

## 离线冒烟（`--tiny`）

```bash
python gemma/gemma-4-e2b-it/quantize.py --tiny --bits 4 --out ./out/gemma-4-e2b-it_tiny_q4
python gemma/gemma-4-e2b-it/quantize.py --tiny --bits 8 --out ./out/gemma-4-e2b-it_tiny_q8
python qwen/qwen3.5-2b/quantize.py --tiny --bits 4 --out ./out/qwen3.5-2b_tiny_q4
python lfm/lfm2-350m/quantize.py --tiny --bits 4 --out ./out/lfm2-350m_tiny_q4
```

## 全量模型命令（int4 + int8 + q326_channel）

下列命令使用各目录 `config.yaml` 中的默认 `base_model`（可用 `--model` 覆盖）。
大机器可加 `--workers 16`（H200）或 `--workers 24`（RTX PRO 6000）。

**推荐（生成质量）**：`--bits 3.26 --codebook-share channel`（敏感层优先 4-bit，其余约 3-bit，全层 channel 码本）。
在 Qwen3-0.6B 上 gen 前缀一致性接近全量 q8，体积通常低于全 q8。输出目录后缀：`_q326_channel`。

### Qwen

```bash
# qwen3-0.6b
python qwen/qwen3-0.6b/quantize.py --bits 4 --out ./out/qwen3-0.6b_q4
python qwen/qwen3-0.6b/quantize.py --bits 8 --out ./out/qwen3-0.6b_q8
python qwen/qwen3-0.6b/quantize.py --bits 3.26 --codebook-share channel --out ./out/qwen3-0.6b_q326_channel

# qwen3-1.7b
python qwen/qwen3-1.7b/quantize.py --bits 4 --out ./out/qwen3-1.7b_q4
python qwen/qwen3-1.7b/quantize.py --bits 8 --out ./out/qwen3-1.7b_q8
python qwen/qwen3-1.7b/quantize.py --bits 3.26 --codebook-share channel --out ./out/qwen3-1.7b_q326_channel

# qwen3.5-0.8b
python qwen/qwen3.5-0.8b/quantize.py --bits 4 --out ./out/qwen3.5-0.8b_q4
python qwen/qwen3.5-0.8b/quantize.py --bits 8 --out ./out/qwen3.5-0.8b_q8
python qwen/qwen3.5-0.8b/quantize.py --bits 3.26 --codebook-share channel --out ./out/qwen3.5-0.8b_q326_channel

# qwen3.5-2b
python qwen/qwen3.5-2b/quantize.py --bits 4 --out ./out/qwen3.5-2b_q4
python qwen/qwen3.5-2b/quantize.py --bits 8 --out ./out/qwen3.5-2b_q8
python qwen/qwen3.5-2b/quantize.py --bits 3.26 --codebook-share channel --out ./out/qwen3.5-2b_q326_channel
```

### Gemma

```bash
# gemma-3-270m-it
python gemma/gemma-3-270m-it/quantize.py --bits 4 --out ./out/gemma-3-270m-it_q4
python gemma/gemma-3-270m-it/quantize.py --bits 8 --out ./out/gemma-3-270m-it_q8
python gemma/gemma-3-270m-it/quantize.py --bits 3.26 --codebook-share channel --out ./out/gemma-3-270m-it_q326_channel

# gemma-3-1b-it
python gemma/gemma-3-1b-it/quantize.py --bits 4 --out ./out/gemma-3-1b-it_q4
python gemma/gemma-3-1b-it/quantize.py --bits 8 --out ./out/gemma-3-1b-it_q8
python gemma/gemma-3-1b-it/quantize.py --bits 3.26 --codebook-share channel --out ./out/gemma-3-1b-it_q326_channel

# gemma-3n-e2b-it
python gemma/gemma-3n-e2b-it/quantize.py --bits 4 --out ./out/gemma-3n-e2b-it_q4
python gemma/gemma-3n-e2b-it/quantize.py --bits 8 --out ./out/gemma-3n-e2b-it_q8
python gemma/gemma-3n-e2b-it/quantize.py --bits 3.26 --codebook-share channel --out ./out/gemma-3n-e2b-it_q326_channel

# gemma-3n-e4b-it
python gemma/gemma-3n-e4b-it/quantize.py --bits 4 --out ./out/gemma-3n-e4b-it_q4
python gemma/gemma-3n-e4b-it/quantize.py --bits 8 --out ./out/gemma-3n-e4b-it_q8
python gemma/gemma-3n-e4b-it/quantize.py --bits 3.26 --codebook-share channel --out ./out/gemma-3n-e4b-it_q326_channel

# gemma-4-e2b-it
python gemma/gemma-4-e2b-it/quantize.py --bits 4 --out ./out/gemma-4-e2b-it_q4
python gemma/gemma-4-e2b-it/quantize.py --bits 8 --out ./out/gemma-4-e2b-it_q8
python gemma/gemma-4-e2b-it/quantize.py --bits 3.26 --codebook-share channel --out ./out/gemma-4-e2b-it_q326_channel

# gemma-4-e4b-it
python gemma/gemma-4-e4b-it/quantize.py --bits 4 --out ./out/gemma-4-e4b-it_q4
python gemma/gemma-4-e4b-it/quantize.py --bits 8 --out ./out/gemma-4-e4b-it_q8
python gemma/gemma-4-e4b-it/quantize.py --bits 3.26 --codebook-share channel --out ./out/gemma-4-e4b-it_q326_channel
```

### LFM

```bash
# lfm2-350m
python lfm/lfm2-350m/quantize.py --bits 4 --out ./out/lfm2-350m_q4
python lfm/lfm2-350m/quantize.py --bits 8 --out ./out/lfm2-350m_q8
python lfm/lfm2-350m/quantize.py --bits 3.26 --codebook-share channel --out ./out/lfm2-350m_q326_channel

# lfm2-700m
python lfm/lfm2-700m/quantize.py --bits 4 --out ./out/lfm2-700m_q4
python lfm/lfm2-700m/quantize.py --bits 8 --out ./out/lfm2-700m_q8
python lfm/lfm2-700m/quantize.py --bits 3.26 --codebook-share channel --out ./out/lfm2-700m_q326_channel

# lfm2-1.2b
python lfm/lfm2-1.2b/quantize.py --bits 4 --out ./out/lfm2-1.2b_q4
python lfm/lfm2-1.2b/quantize.py --bits 8 --out ./out/lfm2-1.2b_q8
python lfm/lfm2-1.2b/quantize.py --bits 3.26 --codebook-share channel --out ./out/lfm2-1.2b_q326_channel

# lfm2-2.6b
python lfm/lfm2-2.6b/quantize.py --bits 4 --out ./out/lfm2-2.6b_q4
python lfm/lfm2-2.6b/quantize.py --bits 8 --out ./out/lfm2-2.6b_q8
python lfm/lfm2-2.6b/quantize.py --bits 3.26 --codebook-share channel --out ./out/lfm2-2.6b_q326_channel

# lfm2-8b-a1b
python lfm/lfm2-8b-a1b/quantize.py --bits 4 --out ./out/lfm2-8b-a1b_q4
python lfm/lfm2-8b-a1b/quantize.py --bits 8 --out ./out/lfm2-8b-a1b_q8
python lfm/lfm2-8b-a1b/quantize.py --bits 3.26 --codebook-share channel --out ./out/lfm2-8b-a1b_q326_channel

# lfm2-vl-450m（含 vision）
python lfm/lfm2-vl-450m/quantize.py --bits 4 --out ./out/lfm2-vl-450m_q4
python lfm/lfm2-vl-450m/quantize.py --bits 8 --out ./out/lfm2-vl-450m_q8
python lfm/lfm2-vl-450m/quantize.py --bits 3.26 --codebook-share channel --out ./out/lfm2-vl-450m_q326_channel

# lfm2.5-350m
python lfm/lfm2.5-350m/quantize.py --bits 4 --out ./out/lfm2.5-350m_q4
python lfm/lfm2.5-350m/quantize.py --bits 8 --out ./out/lfm2.5-350m_q8
python lfm/lfm2.5-350m/quantize.py --bits 3.26 --codebook-share channel --out ./out/lfm2.5-350m_q326_channel

# lfm2.5-1.2b-instruct
python lfm/lfm2.5-1.2b-instruct/quantize.py --bits 4 --out ./out/lfm2.5-1.2b-instruct_q4
python lfm/lfm2.5-1.2b-instruct/quantize.py --bits 8 --out ./out/lfm2.5-1.2b-instruct_q8
python lfm/lfm2.5-1.2b-instruct/quantize.py --bits 3.26 --codebook-share channel --out ./out/lfm2.5-1.2b-instruct_q326_channel

# lfm2.5-1.2b-thinking
python lfm/lfm2.5-1.2b-thinking/quantize.py --bits 4 --out ./out/lfm2.5-1.2b-thinking_q4
python lfm/lfm2.5-1.2b-thinking/quantize.py --bits 8 --out ./out/lfm2.5-1.2b-thinking_q8
python lfm/lfm2.5-1.2b-thinking/quantize.py --bits 3.26 --codebook-share channel --out ./out/lfm2.5-1.2b-thinking_q326_channel

# lfm2.5-2.6b
python lfm/lfm2.5-2.6b/quantize.py --bits 4 --out ./out/lfm2.5-2.6b_q4
python lfm/lfm2.5-2.6b/quantize.py --bits 8 --out ./out/lfm2.5-2.6b_q8
python lfm/lfm2.5-2.6b/quantize.py --bits 3.26 --codebook-share channel --out ./out/lfm2.5-2.6b_q326_channel

# lfm2.5-vl-1.6b（含 vision）
python lfm/lfm2.5-vl-1.6b/quantize.py --bits 4 --out ./out/lfm2.5-vl-1.6b_q4
python lfm/lfm2.5-vl-1.6b/quantize.py --bits 8 --out ./out/lfm2.5-vl-1.6b_q8
python lfm/lfm2.5-vl-1.6b/quantize.py --bits 3.26 --codebook-share channel --out ./out/lfm2.5-vl-1.6b_q326_channel
```

### Nanbeige / Bonsai / Inkling

```bash
# nanbeige4.2-3b
python nanbeige/nanbeige4.2-3b/quantize.py --bits 4 --out ./out/nanbeige4.2-3b_q4
python nanbeige/nanbeige4.2-3b/quantize.py --bits 8 --out ./out/nanbeige4.2-3b_q8
python nanbeige/nanbeige4.2-3b/quantize.py --bits 3.26 --codebook-share channel --out ./out/nanbeige4.2-3b_q326_channel

# bonsai-27b（BF16 safetensors 源约 54GB；产物体积大）
python bonsai/bonsai-27b/quantize.py --bits 4 --workers 16 --out ./out/bonsai-27b_q4
python bonsai/bonsai-27b/quantize.py --bits 8 --workers 16 --out ./out/bonsai-27b_q8
python bonsai/bonsai-27b/quantize.py --bits 3.26 --codebook-share channel --workers 16 --out ./out/bonsai-27b_q326_channel

# inkling-small
python inkling/inkling-small/quantize.py --bits 4 --out ./out/inkling-small_q4
python inkling/inkling-small/quantize.py --bits 8 --out ./out/inkling-small_q8
python inkling/inkling-small/quantize.py --bits 3.26 --codebook-share channel --out ./out/inkling-small_q326_channel
```

### OpenVLA / OpenPI / LingBot（VLA；含 vision / action）

`config_from_hf` 会从 `llm_config` / `language_model_config` / `paligemma_config` /
`vlm_config` 展开嵌套 LLM 几何；若无法解析 `hidden_size`（或等价字段），则以明确
`ConfigError` 失败，而不是写出坏 bundle。

```bash
# openvla-7b  (HF: openvla/openvla-7b)
python openvla/openvla-7b/quantize.py --bits 4 --out ./out/openvla-7b_q4
python openvla/openvla-7b/quantize.py --bits 8 --out ./out/openvla-7b_q8
python openvla/openvla-7b/quantize.py --bits 3.26 --codebook-share channel --out ./out/openvla-7b_q326_channel

# openpi-pi0-3b  (HF: lerobot/pi0_base)
python openpi/openpi-pi0-3b/quantize.py --bits 4 --out ./out/openpi-pi0-3b_q4
python openpi/openpi-pi0-3b/quantize.py --bits 8 --out ./out/openpi-pi0-3b_q8
python openpi/openpi-pi0-3b/quantize.py --bits 3.26 --codebook-share channel --out ./out/openpi-pi0-3b_q326_channel

# openpi-pi0.5-3b  (HF: lerobot/pi05_base)
python openpi/openpi-pi0.5-3b/quantize.py --bits 4 --out ./out/openpi-pi0.5-3b_q4
python openpi/openpi-pi0.5-3b/quantize.py --bits 8 --out ./out/openpi-pi0.5-3b_q8
python openpi/openpi-pi0.5-3b/quantize.py --bits 3.26 --codebook-share channel --out ./out/openpi-pi0.5-3b_q326_channel

# lingbot-vla-v2-6b  (HF: robbyant/lingbot-vla-v2-6b)
python lingbot/lingbot-vla-v2-6b/quantize.py --bits 4 --out ./out/lingbot-vla-v2-6b_q4
python lingbot/lingbot-vla-v2-6b/quantize.py --bits 8 --out ./out/lingbot-vla-v2-6b_q8
python lingbot/lingbot-vla-v2-6b/quantize.py --bits 3.26 --codebook-share channel --out ./out/lingbot-vla-v2-6b_q326_channel
```

## 可选混合精度命令

各家族的 **`--bits 3.26 --codebook-share channel`**（`_q326_channel`）已写在上方全量命令中，作为推荐生成质量配方。

```bash
# PLE 加权（Gemma-4-E2B 目标 <1GB）
python gemma/gemma-4-e2b-it/quantize.py --bits 1.5 --workers 16 --out ./out/gemma-4-e2b-it_q15
python gemma/gemma-4-e4b-it/quantize.py --bits 1.5 --workers 16 --out ./out/gemma-4-e4b-it_q15
python gemma/gemma-3n-e2b-it/quantize.py --bits 1.5 --workers 16 --out ./out/gemma-3n-e2b-it_q15
python gemma/gemma-3n-e4b-it/quantize.py --bits 1.5 --workers 16 --out ./out/gemma-3n-e4b-it_q15

# 按层数混合、默认 group 码本（无 channel）
python qwen/qwen3.5-2b/quantize.py --bits 2.54 --out ./out/qwen3.5-2b_q254
python qwen/qwen3.5-2b/quantize.py --bits 3.26 --out ./out/qwen3.5-2b_q326
python gemma/gemma-4-e2b-it/quantize.py --bits 2.54 --out ./out/gemma-4-e2b-it_q254
python gemma/gemma-4-e2b-it/quantize.py --bits 3.26 --out ./out/gemma-4-e2b-it_q326
```

## 质量审计

```bash
# A) 分层 RMSE（离线 tiny 对照 — 仅适用于 --tiny 产物）
python -m common.audit_cli layer \
  --bundle ./out/gemma-4-e2b-it_tiny_q4 \
  --ref tiny \
  --sample 8 \
  --family gemma-4-e2b-it \
  --report ./out/gemma-4-e2b-it_tiny_q4/audit_layer.json

# A) 对照 HF 原权重（全量量化产物；--family 可解析 base_model）
python -m common.audit_cli layer \
  --bundle ./out/qwen3.5-2b_q4 \
  --family qwen3.5-2b \
  --sample 8 \
  --report ./out/qwen3.5-2b_q4/audit_layer.json
# 或显式: --model Qwen/Qwen3.5-2B

# B) 纯文本短生成对比（Qwen / Gemma / LFM / Inkling / …）
# 默认 completion 风格 prompt；--min-new-tokens 8；报告 exact_prefix + logprob delta
python -m common.audit_cli gen \
  --bundle ./out/qwen3.5-2b_q4 \
  --model Qwen/Qwen3.5-2B \
  --kind text \
  --min-new-tokens 8 \
  --max-new-tokens 32 \
  --report ./out/qwen3.5-2b_q4/audit_gen.json

# B) VLA 前向 / 参数漂移对比（OpenVLA / OpenPI / LingBot）
python -m common.audit_cli gen \
  --bundle ./out/openvla-7b_q4 \
  --model openvla/openvla-7b \
  --kind vla \
  --report ./out/openvla-7b_q4/audit_gen.json
```

报告阈值（blocked 逆变换原域 `rel_rmse_orig`）：q8 ≤ 0.15，q4 ≤ 0.35，其它/混合 ≤ 0.50（embed/PLE ≤ 0.80）。新产物 `format_version=2` + blocked Hadamard（旧 pad-crop bundle 需重量化）。

短生成配方对比归档（Qwen3-0.6B：q4 / q4+channel / q326+channel / q8；未采纳异构方案）：
[`docs/gen_quant_eval_qwen3-0.6b.md`](docs/gen_quant_eval_qwen3-0.6b.md)。

### Qwen3 对话诊断

当 `aria-engine` 聊天乱码（例如 Hello → `"olum啦…"`）但层 RMSE 正常时，用这一对脚本拆分问题。两侧共用同一段 ChatML（`enable_thinking=False` / 空 `<think>`），默认 user `Hello`，greedy `max_tokens=32`。

| 脚本 | 隔离什么 |
|------|----------|
| [`scripts/diag_qwen3_chat.py`](scripts/diag_qwen3_chat.py) | **量化 + 模板**：HF fp32 vs 把 `reconstruct_weight` 注入同一张 HF 图 |
| [`../engine/scripts/diag_qwen3_chat.py`](../engine/scripts/diag_qwen3_chat.py) | **引擎图**：OpenAI `/v1/chat/completions` vs model 侧 JSON（`--peer-report`） |

**1. model 侧**（建议 GPU；tokenizer 从 `--hf` 加载，不用 Aria bundle 里的）：

```bash
# 在本仓库根目录；需要 CUDA torch + transformers
pip install torch transformers
python scripts/diag_qwen3_chat.py \
  --bundle ~/.ariacompute/models/qwen3-0.6b_q4 \
  --hf Qwen/Qwen3-0.6B \
  --device cuda \
  --report ./out/model_diag_qwen3.json
```

`--device auto` 在有 CUDA 时选用 GPU。Attention 走 eager（关闭 hub CUDA JIT）。若仍编译失败：`sudo apt install python3-dev`。

对照 `chat.fp32` 与 `chat.reconstruct`，以及 `template_string_match` / `prompt_ids_match`。若 reconstruct 已能打出类似 `"Hello! How can I assist you today?"` 且 `exact_prefix_len >= 4`，说明 **bundle 在 HF 上可用**，引擎乱码不是量化问题。

**2. engine 侧**（必须 `serve` **同一份** bundle，且不走云 handoff）：

```bash
# 在并列的 engine 仓库；aria-engine 已在监听
./aria-engine serve qwen3-0.6b_q4 \
  --bind 127.0.0.1:8080 \
  --hybrid-execution device
python scripts/diag_qwen3_chat.py \
  --url http://127.0.0.1:8080 \
  --bundle ~/.ariacompute/models/qwen3-0.6b_q4 \
  --peer-report ../model/out/model_diag_qwen3.json \
  --report ./out/engine_diag_qwen3.json
```

`--timeout` 默认 300s（CPU decode 可能很慢）。可选 `pip install tokenizers`，以便用 `bundle/tokenizer.json` 编 prompt ids。

**如何读 `hints`**

| 现象 | 更可能的原因 |
|------|----------------|
| `template_string_match` / `prompt_ids_match` 为 false | ChatML / tokenizer 编码不一致 |
| reconstruct greedy 已相对 fp32 发散（`QUANT:…`） | 码本 / Hadamard / 注入 |
| reconstruct 对话正常，引擎 `content` 仍乱（`ENGINE_GRAPH`） | Rust 前向（HDM、embed 按行取值、RoPE、QK-norm 等） |

引擎的教师信号是 **HF + reconstruct 注入**，不是裸 fp32。

### Gemma-4 对话诊断

当 `gemma-4-e2b-it_q4` 聊天乱码（`engine.log` Hello → `"uhnyaчь…"`）时，用这一对脚本拆分。两侧共用引擎 `gemma4_it` 字符串（`<bos><|turn>user…<turn|>\n<|turn>model\n`），默认 user `Hello`，greedy `max_tokens=32`。官方 Hello prompt 为 **10** token（`<|turn>` / `<turn|>`）；旧的 Gemma-3 `<start_of_turn>` 会碎成 28 段。

| 脚本 | 隔离什么 |
|------|----------|
| [`scripts/diag_gemma4_chat.py`](scripts/diag_gemma4_chat.py) | **量化 + 模板**：HF fp32 vs 把 `reconstruct_weight` 注入同一张 HF 图（VL 封装） |
| [`../engine/scripts/diag_gemma4_chat.py`](../engine/scripts/diag_gemma4_chat.py) | **引擎图**：OpenAI `/v1/chat/completions` vs model 侧 JSON（`--peer-report`） |

**1. model 侧**（建议 GPU；tokenizer 从 `--hf` 加载；E2B 两份 fp32 需要显存）：

```bash
# 在本仓库根目录；需要 CUDA torch + transformers
pip install torch transformers
python scripts/diag_gemma4_chat.py \
  --bundle ~/.ariacompute/models/gemma-4-e2b-it_q4 \
  --hf google/gemma-4-E2B-it \
  --device cuda \
  --report ./out/model_diag_gemma4.json
```

对照 `chat.fp32` 与 `chat.reconstruct`，以及 `template_string_match` / `prompt_ids_match` / `inject`。若 reconstruct 已能正常问候且 `exact_prefix_len >= 4`，说明 **bundle 在 HF 上可用**，引擎乱码不是量化问题。Hello 的 `prompt_ids_len_engine_template` 应为 **10**。

**2. engine 侧**（必须 `serve` **同一份** bundle，且不走云 handoff）：

```bash
# 在并列的 engine 仓库；aria-engine 已在监听
./aria-engine serve gemma-4-e2b-it_q4 \
  --bind 127.0.0.1:8080 \
  --hybrid-execution device
python scripts/diag_gemma4_chat.py \
  --url http://127.0.0.1:8080 \
  --bundle ~/.ariacompute/models/gemma-4-e2b-it_q4 \
  --peer-report ../model/out/model_diag_gemma4.json \
  --report ./out/engine_diag_gemma4.json
```

`hints` 读法与 Qwen3 表相同（`TEMPLATE` / `QUANT` / `ENGINE_GRAPH`）；若 bundle 名对不上 HF VL 的 `language_model` 键，会多一条 `INJECT`。

## 测试

```bash
python -m unittest discover -s tests -t .
```

## 说明

- 流式 I/O 只降低峰值内存；**每张 2D 权重**仍完整执行 **blocked Hadamard**（greedy 2 幂行分块，`H_B@S_B`）与 **Lloyd-Max 码本量化**。无全局 pad/裁剪；原域经 `S_B@H_B` 精确重建。engine HDM 对激活使用相同分块。
- 默认 **`--codebook-share group`**：每个 row-group 共用一份码本。已装 CUDA torch 时，group 路径自动用
  `lloyd_max_batched_torch`；否则 CPU numpy（可用 `--workers`）。`--codebook-share channel` 精度更高、体积更大，亦可走 GPU。
- **`--bits 3.26 --codebook-share channel`**：敏感层优先 4-bit、其余约 3-bit，全层 channel 码本；推荐用于抬升短生成一致性（相对纯 q4+group）。产物目录用 `_q326_channel`。
- **`--bits 1.5`**：PLE / 大 embedding 默认 **1 bit**，计算层 **2–3 bit**，按**参数量加权**平均约 1.35–1.55；Gemma-4-E2B 目标 `weight.bin` **&lt; 1 GB**。
- 仅对 **2D** 权重做码本量化；1D（如 RMSNorm）以 raw fp16 旁路写入。
- 产物目录（`./out/`、`**/weights/`、`*.bin`）已 gitignore，勿提交大文件。
- 本仓库**不**导出 GGUF；live `engine` 的量化加载器暂不在范围内。

## 工程规范

本仓库遵循 Harness Engineering 理念：

- [`AGENTS.md`](AGENTS.md)：Agent 工程上下文入口与目录索引
- [`requirements.md`](requirements.md)：需求规格（功能边界/异常/验收标准，人工审核制）
- [`task.md`](task.md)：实施任务清单
