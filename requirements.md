# requirements.md — Hadamard + 码本量化（Python）

> 本文件为 `model` 仓库多家族 **Hadamard 旋转 + Lloyd-Max 码本量化** 的功能边界、API、产物布局、异常与验收标准。**须经人工逐项审核**，审核通过后方可据其生成 / 执行 `task.md`。
>
> 算法参考：旋转预处理 + Lloyd-Max / K-Means 码本量化调研笔记。

## 1. 目标与范围

对 HuggingFace 真实 base model（见 §1.1，可用 `--model` 覆盖）的权重做
**Walsh–Hadamard 旋转预处理 + Lloyd-Max 码本量化**，导出 **Aria model bundle**
（`weight.bin` + `config.json` + tokenizer），以 Python 反量化 roundtrip 为黄金验收路径。

- **位宽**：1 / 2 / 3 / 4 / **8** bit（`q8` = 码本 `K=256`，**唯一 int8 路径**）；混合精度 `1.5` / `2.54` / `3.26`。
- **默认 `group_size=32`**；**默认 `codebook_share=group`**。
- **VL / 多模态**：默认**量化 vision（及存在的 audio）塔**全部 2D 权重；不提供 text-only 跳过开关（本阶段）。
- **体积目标（可选）**：Gemma-4-E2B + `--bits 1.5` + `group` → `weight.bin` **&lt; 1 GB**（已验收 ~993 MB）。
- **不动 live `engine` / `serve`**。

### 1.1 家族注册表

| 目录 | `base_model` |
|------|----------------|
| `qwen/qwen3-0.6b` | `Qwen/Qwen3-0.6B` |
| `qwen/qwen3-1.7b` | `Qwen/Qwen3-1.7B` |
| `qwen/qwen3.5-0.8b` | `Qwen/Qwen3.5-0.8B` |
| `qwen/qwen3.5-2b` | `Qwen/Qwen3.5-2B` |
| `gemma/gemma-3-270m-it` | `google/gemma-3-270m-it` |
| `gemma/gemma-3-1b-it` | `google/gemma-3-1b-it` |
| `gemma/gemma-3n-e2b-it` | `google/gemma-3n-E2B-it` |
| `gemma/gemma-3n-e4b-it` | `google/gemma-3n-E4B-it` |
| `gemma/gemma-4-e2b-it` | `google/gemma-4-E2B-it` |
| `gemma/gemma-4-e4b-it` | `google/gemma-4-E4B-it` |
| `lfm/lfm2-350m` | `LiquidAI/LFM2-350M` |
| `lfm/lfm2-700m` | `LiquidAI/LFM2-700M` |
| `lfm/lfm2-1.2b` | `LiquidAI/LFM2-1.2B` |
| `lfm/lfm2-2.6b` | `LiquidAI/LFM2-2.6B` |
| `lfm/lfm2-8b-a1b` | `LiquidAI/LFM2-8B-A1B` |
| `lfm/lfm2-vl-450m` | `LiquidAI/LFM2-VL-450M` |
| `lfm/lfm2.5-350m` | `LiquidAI/LFM2.5-350M` |
| `lfm/lfm2.5-1.2b-instruct` | `LiquidAI/LFM2.5-1.2B-Instruct` |
| `lfm/lfm2.5-1.2b-thinking` | `LiquidAI/LFM2.5-1.2B-Thinking` |
| `lfm/lfm2.5-vl-1.6b` | `LiquidAI/LFM2.5-VL-1.6B` |
| `nanbeige/nanbeige4.2-3b` | `Nanbeige/Nanbeige4.2-3B` |
| `bonsai/bonsai-27b` | `prism-ml/Bonsai-27B-unpacked` |
| `inkling/inkling-small` | `thinkingmachines/Inkling-Small` |
| `openvla/openvla-7b` | `openvla/openvla-7b` |
| `openpi/openpi-pi0-3b` | `lerobot/pi0_base` |
| `openpi/openpi-pi0.5-3b` | `lerobot/pi05_base` |
| `lingbot/lingbot-vla-v2-6b` | `robbyant/lingbot-vla-v2-6b` |

每个家族目录：`quantize.py` + `config.yaml`（`base_model` / `default_bits` / `group_size` / `hadamard_seed`）。  
权重须以 **safetensors** 可流式读取；若 Hub 仅有 GGUF，量化失败并抛明确错误（**不**解析 GGUF）。Bonsai 使用 unpacked FP16/BF16 仓，非 `*-gguf`。  
VLA（OpenVLA / OpenPI π₀·π₀.₅ / LingBot）：默认量化全部 2D（含 vision / action head）；OpenPI 使用 LeRobot HF 权重，非 GCS 原版。

## 2. 功能边界

| # | 特性 | 实现深度 |
|---|------|----------|
| 1 | Hadamard | `H@W` axis=0；分块 FWHT；**禁止跳过** |
| 2 | 码本 | Lloyd-Max，`K=2^bits`（含 **K=256 for bits=8**） |
| 3 | 位宽 | `1/2/3/4/8` → `q1`–`q4`/`q8`；混合 `1.5`/`2.54`/`3.26` |
| 4 | int8 | **仅**码本 8-bit；无对称 int8 / GPTQ 旁路 |
| 5 | 混合精度 | §3.4（`2.54`/`3.26` 层数；`1.5` 参数加权） |
| 6 | Bundle | 流式 `BundleWriter`；VL 全量 2D（含 vision） |
| 7 | 旁路 | 仅 ndim==2 码本；1D raw fp16/fp32 |
| 8 | 主机 | `runtime.py`；`--workers`；可选 CUDA Lloyd |

### 2.1 非目标
- QuaRot / SpinQuant；Embedding 专用标量路径；GPTQ / AWQ；剪枝 / 蒸馏
- **解析 / 导出 GGUF**；对称 per-channel int8（非码本）
- 修改 live `engine` / `serve`；提交多 GB 权重进 Git
- text-only 跳过 vision（本阶段）

## 3. API 边界

### 3.1–3.2 hadamard / codebook
不变；`lloyd_max(..., k=256)` 必须可用。

### 3.3 `common/quant.py` + `pack.py`
- 整数位宽集合：**`{1,2,3,4,8}`**。
- `pack` / `unpack`：1–4 为 LSB-first 位打包；**8-bit 为每索引 1 字节**（`uint8` 原始字节，等价于满字节打包）。
- `quantize_weight` / `dequantize`：支持 `bits=8`；索引仍存 `uint8`（0–255）。
- `codebook_share` group/channel 行为不变。

### 3.4 混合精度
- `2.54` / `3.26` / `1.5`：既有行为不变；混合档**不**分配 bit=8（hi 上限仍 4）。

### 3.5–3.8 bundle / hf / cli
- `quantization` label 含 **`q8`**。
- CLI：`--bits` 接受 `8`；其余标志不变。
- 流式：`load_model_info` →（若 `1.5`）加权分配 → `stream_weights` 逐张量量化。

## 4. 产物与体积

- `q4` / `q8`：索引约 `params×0.5` / `params×1` 字节（`group` 码本额外较小）。
- Gemma-4-E2B `q1.5`：`weight.bin` &lt; 1GB（已达成）。
- 推荐：小模型默认跑 `q4` 与 `q8`；大模型 / PLE 优先 `q4` 或 `q1.5`。

## 5. 异常
同前；非法 bits、Hadamard 未 applied、非 safetensors 权重源 → 明确 `ModelError` 派生。

## 6. 验收标准

1. `unittest` 全绿；含 pack/quant **q8** roundtrip；group 模式 q8 RMSE 上界（高斯）≤ **0.25**。
2. 既有 q1–q4 / 1.5 / 2.54 / 3.26 / core guarantees 保持。
3. 每个注册家族目录存在 `quantize.py` + `config.yaml`，且 `base_model` 与 §1.1 一致。
4. `--tiny --bits 8` 写出 `quantization=q8`。
5. `parse_bits(8)==8`；`parse_bits(5)` 仍拒绝。
6. `python -m common.audit_cli layer …` 对 tiny bundle 写出 JSON 报告；**超阈值只记入报告，exit code 仍为 0**（不 fail CI）。

## 6.1 质量审计（本增量）

入口：**独立** `python -m common.audit_cli`（不挂在各家族 `quantize.py`）。

### A — 层抽检（全模型）
- 从 bundle 分层抽样若干 2D codebook 张量（默认 8；策略 `stratified`：embed / attn / ffn / vision|action / other）。
- 对照源权重（`--model` HF 流式，或 `--ref tiny` 用合成 dict）：  
  `rel_rmse_rot`（旋转域 dequant vs Hadamard(W)）与 `rel_rmse_orig`（逆 Hadamard 后 vs W）。
- **报告阈值**（只写入报告的 `threshold` / `pass` 字段，**不**因此非零退出）：

| bits | 原域 `rel_rmse_orig` 参考上界 |
|------|-------------------------------|
| 8 | ≤ 0.15 |
| 4 | ≤ 0.35 |
| 其它 / 混合 | ≤ 0.50（PLE/`q1.5` 层可再放宽至 0.80，报告内标注） |

### B — 少量生成 / 前向对比（区分家族）
- **`kind=text`**（Qwen / Gemma / LFM / Inkling / Nanbeige / Bonsai）：可选 `torch`+`transformers`，短 prompt 生成对比（token overlap 等）；缺依赖则报告 `skipped`。
- **`kind=vla`**（OpenVLA / OpenPI π₀·π₀.₅ / LingBot）：不做文本生成；对少量 dummy 输入做 **action / 末层输出** 对比（余弦或 L2）；缺依赖或无法加载则 `skipped`。
- 同样：**只出报告，不 fail CI**。

## 7. 目录
`common/*`（含 `audit.py` / `gen_compare.py` / `audit_cli.py`）+ §1.1 各家族 + `tests/` + 根文档。

## 8. 审核检查表（本增量）
- [x] int8 = 码本 8-bit（K=256）为**唯一**路径
- [x] VL 默认量化 vision
- [x] §1.1 HF id（含 bonsai / inkling / nanbeige / lfm2.5-thinking）可接受
- [x] Hub 无 safetensors 时失败（不解析 GGUF）可接受
- [x] 批量家族薄封装（共用 `common.cli`）可接受
- [x] A+B 审计：独立 `audit_cli`；阈值仅报告；text vs VLA 分流
