# requirements.md — Hadamard + 码本量化（Python）

> 本文件为 `model` 仓库「gemma-4-e2b-it / qwen3.5-2b 的 Hadamard 旋转 + 码本量化」功能的功能边界、API、产物布局、异常与验收标准。**须经人工逐项审核**，审核通过后方可据其生成 / 执行 `task.md` 分步实施。
>
> 算法参考：旋转预处理 + Lloyd-Max / K-Means 码本量化调研笔记（伪代码与矩阵字段约定）。

## 1. 目标与范围

在 `model` 仓库实现独立 Python 量化管线：对 HuggingFace 真实 base model
（默认 `google/gemma-4-E2B-it`、`Qwen/Qwen3.5-2B`，可用 `--model` 覆盖）的权重做
**Walsh–Hadamard 旋转预处理 + Lloyd-Max / K-Means 码本量化**，导出 **Aria model bundle**
（`weight.bin` + `config.json` + tokenizer），以 Python 反量化 roundtrip 为黄金验收路径。

- **位宽**：1 / 2 / 3 / 4 bit；混合精度 `--bits 2.54` / `3.26` / **`1.5`**。
- **默认 `group_size=32`**；**默认 `codebook_share=group`**（体积优先）。
- **体积目标（可选档）**：Gemma-4-E2B + `--bits 1.5` + `codebook_share=group` → `weight.bin` **&lt; 1 GB**。
- 本特性**不动 live `engine`**；bundle 字段预留 `aria.quant_*` 说明供后续对接。

## 2. 功能边界

| # | 特性 | 实现深度 |
|---|------|----------|
| 1 | Hadamard 旋转 | `W_rot = H @ W`（**仅 axis=0**）；非 2 幂 pad 后 FWHT；内存不足时**按列分块**（与整矩阵等价），**禁止跳过** |
| 2 | 码本量化 | Lloyd-Max，`K=2^bits`；`--codebook-share group`（默认）或 `channel`；index-only |
| 3 | 位宽 | `--bits 1/2/3/4` → `q1`–`q4`；`--bits 2.54` / `3.26` / **`1.5`** 混合精度 |
| 4 | 混合精度分配 | 见 §3.4（含 PLE 默认 1 bit + **参数加权**） |
| 5 | Bundle 导出 | 流式 `BundleWriter` → `weight.bin` + `config.json` + tokenizer |
| 6 | 旁路张量 | 仅 **ndim==2** 走码本；1D 以 fp16/fp32 raw 写入 |
| 7 | Embedding / PLE | 与其它 2D 相同量化路径；`--bits 1.5` 时大表默认更低 bit（见 §3.4） |
| 8 | 流式 / tiny | 逐张量量化写出（不削弱 Hadamard+Lloyd-Max）；`--tiny` smoke |
| 9 | 主机适配 | `common/runtime.py`：按 RAM 定 FWHT 预算；可选 CUDA Lloyd；`--workers` |
| 10 | 统一异常 / 单测 | `ModelError` 派生；unittest 覆盖核心不变量、混合档与体积估算 |

### 2.1 非目标
- QuaRot / SpinQuant 可学习旋转与跨层吸收
- Embedding 专用标量路径 / TurboQuant-H（仍走 Hadamard + 码本）
- GPTQ / AWQ；剪枝 / 蒸馏；GGUF / ONNX
- 修改 live `engine`；提交多 GB 权重进 Git
- 本阶段不强制砍掉 vision/audio 塔（可选后续）；`--bits 1.5` 体积验收以**全量 2D 权重**计入

## 3. API 边界

### 3.1 `common/hadamard.py`
- `hadamard_matrix` / `next_pow2` / `fwht_inplace` / `hadamard_rotate(W, axis=0, seed=None)`
- `axis` 必须为 0；`meta`：`applied=true`、`chunked`、`row_pad`、`chunk_width`

### 3.2 `common/codebook.py`
- `lloyd_max` / `quantize_group` / `lloyd_max_columns` / `lloyd_max_columns_torch`（可选 CUDA）

### 3.3 `common/quant.py`
- `QuantTensor`：`bits, group_size, shape, packed_indices, codebook, hadamard_meta, row_pad, codebook_share`；`input_scale` / `input_scale_recip` / `norms` 可为空（默认不落盘）。
- `quantize_weight(..., codebook_share="group"|"channel")`  
  - **group（默认）**：每 row-group 一份码本，形状 `(num_groups, 2^bits)`。  
  - **channel**：每 (group, channel) 一份，形状 `(num_groups, N, 2^bits)`。  
  流程恒为：Hadamard → 切 group → Lloyd-Max → pack。流式必须调用本函数。
- `dequantize(t)`：按 `codebook_share` / `codebook.ndim` 重建旋转空间权重。

### 3.4 混合精度

#### 3.4.1 保留：`2.54` / `3.26`（按张量个数）
- `allocate_mixed_bits(names, target)`：现有行为不变（敏感名优先拿 `hi`，按**层数**凑平均）。
- `2.54` → {2,3}，层均 ∈ [2.45, 2.65]；`3.26` → {3,4}，层均 ∈ [3.15, 3.40]。
- 敏感名启发式（**仅用于 2.54/3.26**）：`embed` / `embd` / `lm_head` / `output` / attn qkv/o 等（现有 `SENSITIVE_SUBSTR`）。

#### 3.4.2 新增：`--bits 1.5`（PLE 默认 1 bit + 参数加权）
目标：Gemma-4-E2B 在 `codebook_share=group` 下 **`weight.bin` &lt; 1 GB**（索引主导，加权平均 bit ≲ 1.57）。

**分类**（对每个 2D 张量名 + shape / numel）：

| 类别 | 判定（满足任一即可；实现可细化） | 默认 bit |
|------|----------------------------------|----------|
| **PLE / 大 embedding** | 名含 `embed`/`embd`/`per_layer`/`ple`/`embedding`；或 `numel ≥ 50M` 且行维像词表（≥ 32k） | **`ple_bits`，默认 1** |
| **计算敏感** | 名含 `lm_head` / `attn` qkv/o / `q_proj`…（**不含**纯 embedding 大表） | **`hi_bits`，默认 3** |
| **其余计算层** | FFN 等 | **`compute_bits`，默认 2** |

注意：与 2.54/3.26 **相反**——大 embedding / PLE 在 `1.5` 档拿**更低** bit。

**分配 API**（建议签名，实现可等价拆分）：

```text
allocate_mixed_bits_weighted(
  layers: list[(name, numel)],
  target=1.5,
  *,
  ple_bits=1, compute_bits=2, hi_bits=3,
) -> dict[str, int]
```

- **加权平均** `Σ(bits_i * numel_i) / Σ numel_i` 须落入 **[1.35, 1.55]**（可调窄，但须覆盖「PLE@1 + 计算@2/少量@3」）。
- 若默认 `(1,2,3)` 已在区间内则直接使用；若偏高：优先把非敏感计算层从 3→2，再必要时把少数敏感从 3→2（**不得**把 PLE 升到 &gt; `ple_bits` 来凑数）；若偏低：优先升高敏感计算层 bit（上限 4），**不得**为凑数把 PLE 升到 &gt;2（除非 CLI 显式 `--ple-bits`）。
- `parse_bits` / `VALID_BIT_VALUES` 增加 `1.5`；`quantization_label(1.5) == "q1.5"`。
- CLI 可选覆盖：`--ple-bits {1,2}`、`--compute-bits {1,2,3}`、`--hi-bits {2,3,4}`（仅对 `bits=1.5` 生效；非法组合 → `QuantError`）。

**无 numel 的退化**（仅 `--tiny` / 单元测试无真实 shape）：允许仅按名字分类，并用均匀 numel=1 做加权（文档化）；真实 HF 流式路径**必须**带 numel。

### 3.5–3.6 pack / bundle
- `pack_indices` / `unpack_indices`（LSB-first）。
- `BundleWriter.add` / `close`：流式写 `weight.bin`；仅写 `packed_indices` + `codebook`（及非空 aux）；`config.tensors[name].codebook_share`。
- `load_bundle`：兼容旧 bundle（无 `codebook_share` 时按字节布局推断；可缺省 aux）。
- `config.json` 在 `quantization == "q1.5"` 时可增加摘要字段（可选）：`bit_policy: "ple_weighted"`、`avg_bits_weighted`、`ple_bits` / `compute_bits` / `hi_bits`。

### 3.7–3.8 hf_utils / cli
- `load_model_info` / `stream_weights`（BF16→f32）。
- 流式量化：先扫一遍（或从 index）得到 `name → numel`，再 `allocate_mixed_bits_weighted`，再逐张量量化写出。
- CLI：`--model --bits --group-size --out --seed --tiny --config --workers --codebook-share`
  + `--ple-bits` / `--compute-bits` / `--hi-bits`（仅 `1.5`）。

### 3.9 `common/runtime.py`
- 不变：`total_ram_bytes` / `max_work_elems` / `default_workers` / `cuda_available` / `runtime_summary`。

## 4. 产物布局

### 4.1 `config.json`
同前（`format: aria-quant-bundle`）；`quantization` 可为 `q1`–`q4` / `q2.54` / `q3.26` / **`q1.5`**；每张量 `bits` 为实际整数位宽。

### 4.2 体积预期（Gemma-4-E2B ≈ 5.1B 参数，`codebook_share=group`）

| 模式 | `weight.bin` 约 |
|------|-----------------|
| `q4` | **~2.5–3 GB** |
| `q2.54` / `q3.26` | ~1.6 / ~2.1 GB（层数混合，非加权） |
| **`q1.5`（PLE@1，计算@2/3）** | **&lt; 1 GB**（目标；索引 ≈ 0.9 GB 量级） |
| `codebook_share=channel`（任意位宽） | 显著更大；**不作为 &lt;1GB 验收路径** |

### 4.3 `weight.bin`
- 小端；`packed_indices` 位打包；`codebook` fp16 LE；raw 旁路 fp16/fp32 LE。

## 5. 异常
同前；另：Hadamard 未 `applied`、非法 `codebook_share`、非法 `1.5` 覆盖位宽、加权平均无法落入 band → `QuantError`。

## 6. 验收标准

1. `python -m unittest discover -s tests -t .` 全绿（含 `test_core_guarantees`）。
2. Hadamard / group·channel 码本 / pack 既有验收不变。
3. `2.54` / `3.26` 层均 bit 区间单测保持。
4. **`1.5`**：合成 name+numel 列表下，PLE 类默认 `ple_bits==1`；加权平均 ∈ [1.35, 1.55]；升高 PLE 不被「凑平均」逻辑误用（单测锁定）。
5. CLI：`--bits 1.5` + `--tiny` 写出 `quantization=q1.5`；非法 bits → `QuantError`。
6. **人工 / 可选 CI**：全量 Gemma-4-E2B `--bits 1.5 --codebook-share group` 后 `weight.bin` **&lt; 1_073_741_824** 字节（文档验收；单测可用估算函数 `estimate_index_bytes(bit_map, numels)` 断言合成场景 &lt; 1GB）。

## 7. 目录
`common/{errors,hadamard,codebook,quant,pack,bundle,hf_utils,cli,runtime}.py` + 两家族脚本 + `tests/`。

## 8. 审核检查表（本增量）
- [x] `--bits 1.5`：PLE / 大 embedding **默认 1 bit** 可接受
- [x] 计算层默认 2、少量敏感 3；CLI `--ple-bits/--compute-bits/--hi-bits` 可接受
- [x] **参数加权**平均（非按层数）与 band [1.35, 1.55] 可接受
- [x] Gemma-4-E2B `q1.5` + `group` → `weight.bin` **&lt; 1GB** 作为体积目标可接受
- [x] 保留现有 `2.54`/`3.26` 行为不变可接受
- [x] 质量降级（相对 q4）可接受；本阶段以体积 + dequant roundtrip 为主
