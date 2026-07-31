# aria model

[English](README.md) | [中文](README_cn.md)

Python 端 **Hadamard 旋转预处理 + Lloyd-Max 码本量化** 工具，面向端侧 LLM 权重。

目标模型：

| 家族 | 脚本 | 默认 HuggingFace 仓库 |
|------|------|------------------------|
| Gemma 4 E2B IT | [`gemma/gemma-4-e2b-it/quantize.py`](gemma/gemma-4-e2b-it/quantize.py) | `google/gemma-4-E2B-it` |
| Qwen 3.5 2B | [`qwen/qwen3.5-2b/quantize.py`](qwen/qwen3.5-2b/quantize.py) | `Qwen/Qwen3.5-2B` |

产物为 Aria 风格 bundle：`weight.bin` + `config.json`（从 HF 下载时附带 tokenizer）。规格见 [`requirements.md`](requirements.md)，Agent 入口见 [`AGENTS.md`](AGENTS.md)。

## 环境安装

```bash
# 在仓库根目录
uv venv .venv && source .venv/bin/activate   # 或: python3 -m venv .venv
uv pip install -r requirements.txt           # 或: pip install -r requirements.txt
# 完整 HF 路径可选：
# uv pip install torch transformers
```

核心量化依赖 **numpy**（读 `config.yaml` 还需 **pyyaml**）。真实 HF 下载另需 **safetensors**、**huggingface_hub**。

### 推荐主机（参考配置）

完整模型量化参考环境：**1× NVIDIA H100**、**16 vCPU**、**200 GiB 内存**（`gpu-h100-sxm`）。在此类机器上工具会：

- 按检测到的内存自动放大 Hadamard 工作缓冲（约 25%）
- 用最多 16 个 CPU worker 做分组码本（`--workers`）
- 若已安装 `torch` 且有 GPU，则用 **CUDA** 批量 Lloyd-Max（`ARIA_QUANT_FORCE_CPU=1` 可关闭）

```bash
# 可选 CUDA 加速
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

export HF_TOKEN=...   # 提高 Hub 限流
python gemma/gemma-4-e2b-it/quantize.py --bits 4 --workers 16
# Gemma-4-E2B 的 weight.bin 约 2.5–3 GB（若 --codebook-share channel 则约 8 GB）
```

## 通用 CLI 参数（两个脚本相同）

| 参数 | 说明 |
|------|------|
| `--bits` | `1` / `2` / `3` / `4`，或混合精度 `1.5` / `2.54` / `3.26` |
| `--model` | 覆盖 HF 仓库 id（默认来自 `config.yaml`） |
| `--group-size` | 码本分组大小（默认 `32`） |
| `--seed` | Hadamard 随机化种子（默认 `0`） |
| `--out` | 输出目录（默认写到各家族目录下 `weights/qN`） |
| `--codebook-share` | `group`（默认，体积小）或 `channel`（更大、精度更高） |
| `--ple-bits` / `--compute-bits` / `--hi-bits` | 仅 `--bits 1.5`（默认 1 / 2 / 3） |
| `--workers` | 并行 group worker 数（默认 CPU 核数，上限 32） |
| `--tiny` | 使用合成 tiny checkpoint，**无需联网** |
| `--config` | 指定其它 `config.yaml` 路径 |

## Gemma（`gemma-4-e2b-it`）

配置：[`gemma/gemma-4-e2b-it/config.yaml`](gemma/gemma-4-e2b-it/config.yaml)。

```bash
# 离线冒烟
python gemma/gemma-4-e2b-it/quantize.py --tiny --bits 4

# 默认 HF 模型 4-bit（会下载权重）
python gemma/gemma-4-e2b-it/quantize.py --bits 4

# q1.5：PLE@1 + 参数加权（Gemma-4-E2B 目标 weight.bin < 1GB）
python gemma/gemma-4-e2b-it/quantize.py --bits 1.5 --workers 16 --out ./out/gemma_q15

# 混合精度，平均约 2.54 bit（按张量个数）
python gemma/gemma-4-e2b-it/quantize.py --bits 2.54 --out ./out/gemma_q254

# 混合精度，平均约 3.26 bit
python gemma/gemma-4-e2b-it/quantize.py --bits 3.26 --out ./out/gemma_q326

# 覆盖模型 id / 种子 / group size
python gemma/gemma-4-e2b-it/quantize.py \
  --model google/gemma-4-E2B-it \
  --bits 4 \
  --group-size 32 \
  --seed 0 \
  --out ./out/gemma_q4
```

## Qwen（`qwen3.5-2b`）

配置：[`qwen/qwen3.5-2b/config.yaml`](qwen/qwen3.5-2b/config.yaml)。

```bash
# 离线冒烟
python qwen/qwen3.5-2b/quantize.py --tiny --bits 4

# 默认 HF 模型 4-bit
python qwen/qwen3.5-2b/quantize.py --bits 4

# 混合精度
python qwen/qwen3.5-2b/quantize.py --bits 2.54 --out ./out/qwen_q254
python qwen/qwen3.5-2b/quantize.py --bits 3.26 --out ./out/qwen_q326

# 覆盖模型 id
python qwen/qwen3.5-2b/quantize.py \
  --model Qwen/Qwen3.5-2B \
  --bits 4 \
  --out ./out/qwen_q4
```

## 产物结构

例如 `--out ./out/gemma_q4`：

```
out/gemma_q4/
  config.json      # format=aria-quant-bundle，含每张量 offsets / bits
  weight.bin       # packed 索引 + fp16 码本 / scale / norm
  tokenizer.*      # 非 --tiny 时从 HF 拷贝
```

Python 加载与反量化：

```python
from common.bundle import load_bundle
from common.quant import dequantize

cfg, tensors = load_bundle("./out/gemma_q4")
W = dequantize(tensors["blk.0.attn_q.weight"])  # 旋转空间重建
```

## 测试

```bash
python -m unittest discover -s tests -t .
```

## 说明

- 流式 I/O 只降低峰值内存；**每张 2D 权重**仍完整执行 **Hadamard 旋转（`H @ W`，axis=0）** 与 **Lloyd-Max 码本量化**。大矩阵用按列分块 FWHT（与一次性 `H @ W` 数学等价），**不会跳过旋转**。
- 默认 **`--codebook-share group`**：每个 row-group 共用一份码本，`weight.bin` 约等于 4-bit 索引体积（Gemma-4-E2B 约 2.5 GB）。需要更高精度时用 `--codebook-share channel`（约 8 GB）。
- **`--bits 1.5`**：PLE / 大 embedding 默认 **1 bit**，计算层 **2–3 bit**，按**参数量加权**平均约 1.35–1.55；Gemma-4-E2B 目标 `weight.bin` **&lt; 1 GB**。
- 仅对 **2D** 权重做码本量化；1D（如 RMSNorm）以 raw fp16 旁路写入。
- 权重产物目录（`**/weights/`、`*.bin`）已 gitignore，勿提交大文件。
- 本仓库**不**导出 GGUF；live `engine` 的量化加载器暂不在范围内。
