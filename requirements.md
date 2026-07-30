# requirements.md — Hadamard + 码本量化（Python）

> 本文件为 `model` 仓库「gemma-4-e2b-it / qwen3.5-2b 的 Hadamard 旋转 + 码本量化」功能的功能边界、API、产物布局、异常与验收标准。**须经人工逐项审核**，审核通过后方可据其生成 `task.md` 分步实施。
>
> 算法参考：旋转预处理 + Lloyd-Max / K-Means 码本量化调研笔记（伪代码与矩阵字段约定）。

## 1. 目标与范围

在 `model` 仓库实现独立 Python 量化管线：对 HuggingFace 真实 base model
（默认 `google/gemma-4-E2B-it`、`Qwen/Qwen3.5-2B`，可用 `--model` 覆盖）的权重做
**Walsh–Hadamard 旋转预处理 + Lloyd-Max / K-Means 码本量化**，导出 **Aria model bundle**
（`weight.bin` + `config.json` + tokenizer），以 Python 反量化 roundtrip 为黄金验收路径。

- **位宽**：1 / 2 / 3 / 4 bit；混合精度 `--bits 2.54` / `3.26`。
- **默认 `group_size=32`**；**默认 `codebook_share=group`**（体积优先）。
- 本特性**不动 live `engine`**；bundle 字段预留 `aria.quant_*` 说明供后续对接。

## 2. 功能边界

| # | 特性 | 实现深度 |
|---|------|----------|
| 1 | Hadamard 旋转 | `W_rot = H @ W`（**仅 axis=0**）；非 2 幂 pad 后 FWHT；内存不足时**按列分块**（与整矩阵等价），**禁止跳过** |
| 2 | 码本量化 | Lloyd-Max，`K=2^bits`；`--codebook-share group`（默认）或 `channel`；index-only |
| 3 | 位宽 | `--bits 1/2/3/4` → `q1`–`q4`；`--bits 2.54` / `3.26` 混合精度 |
| 4 | 混合精度分配 | `allocate_mixed_bits`：见 §3.4 |
| 5 | Bundle 导出 | 流式 `BundleWriter` → `weight.bin` + `config.json` + tokenizer |
| 6 | 旁路张量 | 仅 **ndim==2** 走码本；1D 以 fp16/fp32 raw 写入 |
| 7 | Embedding | 与其它 2D 相同路径（含 PLE 大表）；靠 `group` 共享码本控制体积 |
| 8 | 流式 / tiny | 逐张量量化写出（不削弱 Hadamard+Lloyd-Max）；`--tiny` smoke |
| 9 | 主机适配 | `common/runtime.py`：按 RAM 定 FWHT 预算；可选 CUDA Lloyd；`--workers` |
| 10 | 统一异常 / 单测 | `ModelError` 派生；unittest 覆盖核心不变量与体积模式 |

### 2.1 非目标
- QuaRot / SpinQuant 可学习旋转与跨层吸收
- Embedding 专用 2-bit 标量路径 / TurboQuant-H
- GPTQ / AWQ；剪枝 / 蒸馏；GGUF / ONNX
- 修改 live `engine`；提交多 GB 权重进 Git

## 3. API 边界

### 3.1 `common/hadamard.py`
- `hadamard_matrix` / `next_pow2` / `fwht_inplace` / `hadamard_rotate(W, axis=0, seed=None)`
- `axis` 必须为 0；`meta`：`applied=true`、`chunked`、`row_pad`、`chunk_width`

### 3.2 `common/codebook.py`
- `lloyd_max` / `quantize_group` / `lloyd_max_columns` / `lloyd_max_columns_torch`（可选 CUDA）

### 3.3 `common/quant.py`
- `QuantTensor`：`bits, group_size, shape, packed_indices, codebook, hadamard_meta, row_pad, codebook_share`；`input_scale` / `input_scale_recip` / `norms` 可为空（默认不落盘）。
- `quantize_weight(..., codebook_share="group"|"channel")`  
  - **group（默认）**：每 row-group 一份码本，形状 `(num_groups, 2^bits)`；对 group 内全部元素做 Lloyd-Max。  
  - **channel**：每 (group, channel) 一份，形状 `(num_groups, N, 2^bits)`（体积约 +2× indices）。  
  流程恒为：Hadamard → 切 group → Lloyd-Max → pack。流式必须调用本函数。
- `dequantize(t)`：按 `codebook_share` / `codebook.ndim` 重建旋转空间权重。

### 3.4 混合精度
- 同前：`allocate_mixed_bits` / `parse_bits` / `quantization_label`；敏感层启发式与平均 bit 区间不变。

### 3.5–3.6 pack / bundle
- `pack_indices` / `unpack_indices`（LSB-first）。
- `BundleWriter.add` / `close`：流式写 `weight.bin`；仅写 `packed_indices` + `codebook`（及非空 aux）；`config.tensors[name].codebook_share`。
- `load_bundle`：兼容旧 bundle（无 `codebook_share` 时按字节布局推断；可缺省 aux）。

### 3.7–3.8 hf_utils / cli
- `load_model_info` / `stream_weights`（BF16→f32，无 numpy bfloat16）。
- CLI：`--model --bits --group-size --out --seed --tiny --config --workers --codebook-share`。

### 3.9 `common/runtime.py`
- `total_ram_bytes` / `max_work_elems` / `default_workers` / `cuda_available` / `runtime_summary`。
- 环境变量：`ARIA_QUANT_MAX_ELEMS`、`ARIA_QUANT_WORKERS`、`ARIA_QUANT_FORCE_CPU`。

## 4. 产物布局

### 4.1 `config.json`（默认 group）
```json
{
  "format": "aria-quant-bundle",
  "format_version": 1,
  "quantization": "q4",
  "group_size_default": 32,
  "hadamard_seed": 0,
  "model": { "...": "..." },
  "tensors": {
    "<name>": {
      "kind": "codebook",
      "bits": 4,
      "group_size": 32,
      "shape": [K, N],
      "row_pad": 0,
      "codebook_share": "group",
      "hadamard": { "seed": 0, "applied": true, "chunked": false },
      "offsets": {
        "packed_indices": [start, length],
        "codebook": [start, length]
      }
    }
  }
}
```

### 4.2 体积预期（Gemma-4-E2B ≈ 5.1B 参数，q4）
| 模式 | `weight.bin` 约 |
|------|-----------------|
| `codebook_share=group`（默认） | **~2.5–3 GB**（≈ 4-bit indices） |
| `codebook_share=channel` | **~8 GB**（indices + 大体码本） |

### 4.3 `weight.bin`
- 小端；`packed_indices` 位打包；`codebook` fp16 LE。
- raw 旁路：fp16/fp32 LE。

## 5. 异常
同前（`QuantError` / `FormatError` / …）；另：Hadamard 未 `applied`、非法 `codebook_share` → `QuantError`。

## 6. 验收标准

1. `python -m unittest discover -s tests -t .` 全绿（含 `test_core_guarantees`）。
2. Hadamard：正交 / 保范；分块 FWHT ≡ 整矩阵；禁止 skip。
3. group 模式 RMSE 上界（高斯、`group_size=32`）：q4 ≤ 0.45，q3 ≤ 0.60，q2 ≤ 0.85，q1 ≤ 1.20；channel 模式 q4 ≤ 0.35。
4. `group` 码本 nbytes ≪ `channel`（同形状单测）。
5. pack/unpack 一致；混合精度平均 bit 区间；`--tiny` roundtrip。
6. CLI：`--codebook-share group|channel`；非法 bits → `QuantError`。

## 7. 目录
`common/{errors,hadamard,codebook,quant,pack,bundle,hf_utils,cli,runtime}.py` + 两家族脚本 + `tests/`。

## 8. 审核检查表
- [ ] 默认 `codebook_share=group` 与体积目标可接受
- [ ] `channel` 作为高精度可选可接受
- [ ] 流式不削弱 Hadamard + Lloyd-Max 可接受
- [ ] 默认不再落盘 scale/recip/norms 可接受
