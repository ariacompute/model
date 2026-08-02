# aria model

[English](README.md) | [中文](README_cn.md)

Python 端 **Hadamard 旋转预处理 + Lloyd-Max 码本量化** 工具，面向端侧 LLM 权重。

支持家族（Qwen / Gemma / LFM / Nanbeige / Bonsai / Inkling / OpenVLA / OpenPI / LingBot）完整表见
[`requirements.md` §1.1](requirements.md)。每目录含 `quantize.py` + `config.yaml`。

产物为 Aria bundle：`weight.bin` + `config.json`（+ tokenizer）。规格见
[`requirements.md`](requirements.md)，Agent 入口见 [`AGENTS.md`](AGENTS.md)。

**int4** = `--bits 4`（码本 K=16）。**int8** = `--bits 8`（码本 K=256，仅 Hadamard+Lloyd-Max）。
VL 默认量化 vision。权重须为 **safetensors**（不解析 GGUF）。

## 环境安装

```bash
# 在仓库根目录
uv venv .venv && source .venv/bin/activate   # 或: python3 -m venv .venv
uv pip install -r requirements.txt           # 或: pip install -r requirements.txt
# 完整 HF 路径可选：
# uv pip install torch transformers
# 可选 CUDA 加速：
# uv pip install torch --index-url https://download.pytorch.org/whl/cu124

export HF_TOKEN=...   # 提高 Hub 限流（全量下载建议设置）
```

核心量化依赖 **numpy**（读 `config.yaml` 还需 **pyyaml**）。真实 HF 下载另需 **safetensors**、**huggingface_hub**。

### 推荐主机（参考配置）

完整模型量化参考环境：**1× NVIDIA H100**、**16 vCPU**、**200 GiB 内存**（`gpu-h100-sxm`）。

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
| `<quant>` | 位宽标签，去掉 `.` | `q4`、`q8`、`q15`、`q254`、`q326` |

示例：`./out/gemma-4-e2b-it_q4`、`./out/qwen3.5-2b_q8`、`./out/gemma-4-e2b-it_q15`。

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

## 全量模型命令（int4 + int8）

下列命令使用各目录 `config.yaml` 中的默认 `base_model`（可用 `--model` 覆盖）。
大机器可加 `--workers 16`。

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

# lfm2-vl-450m（含 vision）
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

# lfm2.5-vl-1.6b（含 vision）
python lfm/lfm2.5-vl-1.6b/quantize.py --bits 4 --out ./out/lfm2.5-vl-1.6b_q4
python lfm/lfm2.5-vl-1.6b/quantize.py --bits 8 --out ./out/lfm2.5-vl-1.6b_q8
```

### Nanbeige / Bonsai / Inkling

```bash
# nanbeige4.2-3b
python nanbeige/nanbeige4.2-3b/quantize.py --bits 4 --out ./out/nanbeige4.2-3b_q4
python nanbeige/nanbeige4.2-3b/quantize.py --bits 8 --out ./out/nanbeige4.2-3b_q8

# bonsai-27b（BF16 safetensors 源约 54GB；产物体积大）
python bonsai/bonsai-27b/quantize.py --bits 4 --workers 16 --out ./out/bonsai-27b_q4
python bonsai/bonsai-27b/quantize.py --bits 8 --workers 16 --out ./out/bonsai-27b_q8

# inkling-small
python inkling/inkling-small/quantize.py --bits 4 --out ./out/inkling-small_q4
python inkling/inkling-small/quantize.py --bits 8 --out ./out/inkling-small_q8
```

### OpenVLA / OpenPI / LingBot（VLA；含 vision / action）

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

## 可选混合精度命令

```bash
# PLE 加权（Gemma-4-E2B 目标 <1GB）
python gemma/gemma-4-e2b-it/quantize.py --bits 1.5 --workers 16 --out ./out/gemma-4-e2b-it_q15
python gemma/gemma-4-e4b-it/quantize.py --bits 1.5 --workers 16 --out ./out/gemma-4-e4b-it_q15
python gemma/gemma-3n-e2b-it/quantize.py --bits 1.5 --workers 16 --out ./out/gemma-3n-e2b-it_q15
python gemma/gemma-3n-e4b-it/quantize.py --bits 1.5 --workers 16 --out ./out/gemma-3n-e4b-it_q15

# 按层数混合（2.54 / 3.26）
python qwen/qwen3.5-2b/quantize.py --bits 2.54 --out ./out/qwen3.5-2b_q254
python qwen/qwen3.5-2b/quantize.py --bits 3.26 --out ./out/qwen3.5-2b_q326
python gemma/gemma-4-e2b-it/quantize.py --bits 2.54 --out ./out/gemma-4-e2b-it_q254
python gemma/gemma-4-e2b-it/quantize.py --bits 3.26 --out ./out/gemma-4-e2b-it_q326
```

## 测试

```bash
python -m unittest discover -s tests -t .
```

## 说明

- 流式 I/O 只降低峰值内存；**每张 2D 权重**仍完整执行 **Hadamard 旋转（`H @ W`，axis=0）** 与 **Lloyd-Max 码本量化**。大矩阵用按列分块 FWHT（与一次性 `H @ W` 数学等价），**不会跳过旋转**。
- 默认 **`--codebook-share group`**：每个 row-group 共用一份码本。需要更高精度时用 `--codebook-share channel`（`weight.bin` 更大）。
- **`--bits 1.5`**：PLE / 大 embedding 默认 **1 bit**，计算层 **2–3 bit**，按**参数量加权**平均约 1.35–1.55；Gemma-4-E2B 目标 `weight.bin` **&lt; 1 GB**。
- 仅对 **2D** 权重做码本量化；1D（如 RMSNorm）以 raw fp16 旁路写入。
- 产物目录（`./out/`、`**/weights/`、`*.bin`）已 gitignore，勿提交大文件。
- 本仓库**不**导出 GGUF；live `engine` 的量化加载器暂不在范围内。
