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
- **默认 `group_size=32`**。
- 本特性**不动 live `engine`**；bundle 字段预留 `aria.quant_*` 说明供后续对接。

## 2. 功能边界

| # | 特性 | 实现深度 |
|---|------|----------|
| 1 | Hadamard 旋转 | 固定或随机化（`seed`）Walsh–Hadamard；`W_rot = H @ W`（行维 axis=0）；非 2 幂行维 pad 到下一 2 幂再裁回，meta 记录 `row_pad` |
| 2 | 码本量化 | per-group × per-channel Lloyd-Max；`K=2^bits`；index-only（1-bit 亦为 2 个码本点 + 索引，不用 sign-magnitude） |
| 3 | 位宽 | `--bits 1/2/3/4` → `q1`–`q4`；`--bits 2.54` / `3.26` 混合精度 |
| 4 | 混合精度分配 | `allocate_mixed_bits`：见 §3.4；可测加权平均 bit |
| 5 | Bundle 导出 | `weight.bin` + `config.json` + 拷贝 tokenizer；**不**产出运行时 graph 二进制 |
| 6 | 旁路张量 | 仅 **ndim==2** 的权重走码本量化；1D（RMSNorm/bias 等）以 fp16 LE 原文写入 bundle |
| 7 | Embedding | 与其它 2D 权重相同量化路径（embedding 专用 2-bit 标量路径为非目标） |
| 8 | 流式 / tiny | 真实模型流式逐张量；`--tiny` 合成 checkpoint 离线 smoke |
| 9 | 统一异常 | `ModelError` 派生，禁止静默失败 |
| 10 | 单测 | unittest/pytest：Hadamard、码本、pack、混合精度、bundle roundtrip、异常路径 |

### 2.1 非目标
- QuaRot / SpinQuant 可学习旋转与跨层吸收
- Embedding 专用 2-bit 标量压缩路径
- GPTQ / AWQ 二次优化
- 剪枝、蒸馏、GGUF / safetensors 归档 / ONNX 导出
- 修改 live `engine` / 提交多 GB 权重进 Git

## 3. API 边界

### 3.1 `common/hadamard.py`
- `hadamard_matrix(n: int, seed: int | None = None) -> np.ndarray`  
  `n` 须为 2 的幂；`seed=None` 为标准 Sylvester Hadamard（归一化使 `H@H.T=I`）；`seed` 给定时对列施加随机 ±1 对角（随机化 Hadamard）。
- `next_pow2(n: int) -> int`
- `hadamard_rotate(W: np.ndarray, axis: int = 0, seed: int | None = None) -> tuple[np.ndarray, dict]`  
  返回 `(W_rot, meta)`；**仅 axis=0**（`W_rot = H @ W`）。非 2 幂行维 pad 到下一 2 幂再裁回。内存不足时**按列分块 FWHT**（与整矩阵 `H @ W` 等价），**禁止跳过旋转**。`meta` 含 `seed`、`row_dim`、`row_pad`、`applied: true`、`chunked`。

### 3.2 `common/codebook.py`
- `lloyd_max(x: np.ndarray, k: int, max_iter: int = 50, tol: float = 1e-6, seed: int | None = 0) -> np.ndarray`  
  一维向量 → `(k,)` 码本（float64 计算，调用方可转 fp16）。
- `quantize_group(col: np.ndarray, codebook: np.ndarray) -> np.ndarray`  
  最近邻索引，`uint8`，长度 = `col.size`。

### 3.3 `common/quant.py`
- `@dataclass QuantTensor`：  
  `bits: int`、`group_size: int`、`shape: tuple[int, int]`（原始 `K,N`）、  
  `packed_indices: bytes`、`codebook: np.ndarray`（fp16，布局见 §4）、  
  `input_scale: np.ndarray`（fp16）、`input_scale_recip: np.ndarray`（fp16）、  
  `norms: np.ndarray`（fp16）、`hadamard_meta: dict`、`row_pad: int`。
- `quantize_weight(W, bits, group_size=32, seed=None, max_iter=50) -> QuantTensor`  
  流程：**Hadamard（必做）** → 按 `group_size` 切行 → 每 (group, channel) **Lloyd-Max（必做）** → pack。  
  流式导出必须逐张量调用本函数；不得因流式而省略旋转或码本。行维 pad 后须能被 `group_size` 整除；否则 `QuantError`。
- `dequantize(t: QuantTensor) -> np.ndarray`：重建旋转后空间权重（与量化前 `W_rot` 对齐）；形状 `(K, N)`（裁掉 pad）。

**码本布局**：对每个 group `g`、每个输出通道 `n`，独立码本长度 `2^bits`。  
存储：`codebook` 展平为 fp16，顺序 `(num_groups, N, 2^bits)`。  
`input_scale[g,n]`：该 group-channel 向量的 `max(abs)`（全零时为 1.0）；`norms[g,n]=||col||_2`。本 Spec 先对 `col` 做 Lloyd-Max（不强制预归一化）。

### 3.4 `common/quant.py` — 混合精度
- `VALID_BITS = {1, 2, 3, 4, 2.54, 3.26}`（CLI 解析 float）。
- `allocate_mixed_bits(layer_names: list[str], target: float) -> dict[str, int]`  
  - **敏感层**（名称匹配，大小写不敏感）：含 `embed`、`embd`、`lm_head`、`output`、`attn_q`、`attn_k`、`attn_v`、`attn_output`、`q_proj`、`k_proj`、`v_proj`、`o_proj`。  
  - **`2.54`**：优先给敏感层更高位宽；按比例分配 2/3 bit，使平均 bit ∈ `[2.45, 2.65]`。  
  - **`3.26`**：优先给敏感层 4 bit，其余 3 bit，使平均 bit ∈ `[3.15, 3.40]`（层数过少时允许近似）。  
  - 1D 旁路层不进入分配表。
- `parse_bits(value) -> float`：非法则 `QuantError`。

### 3.5 `common/pack.py`
- `packed_size(count: int, bits: int) -> int`：`(count * bits + 7) // 8`
- `pack_indices(indices: np.ndarray, bits: int) -> bytes`：低位先填（LSB-first within byte）。
- `unpack_indices(data: bytes, count: int, bits: int) -> np.ndarray`

### 3.6 `common/bundle.py`
- `write_bundle(out_dir, tensors: dict[str, QuantTensor | np.ndarray], model_config: dict, quantization: str, tokenizer_src: str | None = None) -> Path`  
  写出 `config.json`、`weight.bin`；若给 `tokenizer_src` 则拷贝常见 tokenizer 文件。
- `load_bundle(out_dir) -> tuple[dict, dict]`：返回 `(config, name -> QuantTensor|ndarray)`。
- 1D / 旁路：`ndarray` fp16/fp32 写入独立段，`config.tensors[name].kind = "raw"`。
- 码本权重：`kind = "codebook"`，含 `bits, group_size, shape, offsets, hadamard`。

### 3.7 `common/hf_utils.py`
- `make_tiny_state_dict(...)` / `tiny_model_config(...)`：合成 2 层、名称 `token_embd` / `blk.*` / `output*`。
- `load_model_config(repo) -> dict`、`stream_weights(repo) -> Iterator[(name, ndarray)]`（可选依赖；缺失则 `ModelFetchError`）。
- 权重流式读取须将 `BF16`/`F16`/`F32`/`F64` 统一为 `float32`；不得依赖 numpy 原生 `bfloat16` dtype（`safe_open(..., framework="np")` 在常见 numpy 上会报 `data type 'bfloat16' not understood`）。
- `copy_tokenizer(repo_or_path, dest_dir)`。

### 3.8 `common/cli.py` / 家族脚本
- 参数：`--model`、`--bits`、`--group-size`、`--out`、`--seed`、`--tiny`、`--config`。
- 入口：  
  `python gemma/gemma-4-e2b-it/quantize.py --bits 4`  
  `python qwen/qwen3.5-2b/quantize.py --bits 2.54 --out ...`

### 3.9 `common/errors.py`
- `ModelError`、`QuantError`、`FormatError`、`ShapeMismatchError`、`UnsupportedError`、`ModelFetchError`、`ConfigError`。

## 4. 产物布局（Aria bundle）

输出目录示例：`gemma/gemma-4-e2b-it/weights/q4/` 或 `--out`。

### 4.1 `config.json`
```json
{
  "format": "aria-quant-bundle",
  "format_version": 1,
  "quantization": "q4",
  "group_size_default": 32,
  "hadamard_seed": 0,
  "model": { "hidden_size": "...", "num_layers": "...", "vocab_size": "..." },
  "tensors": {
    "<name>": {
      "kind": "codebook",
      "bits": 4,
      "group_size": 32,
      "shape": [K, N],
      "row_pad": 0,
      "hadamard": { "seed": 0, "applied": true },
      "offsets": {
        "packed_indices": [start, length],
        "codebook": [start, length],
        "input_scale": [start, length],
        "input_scale_recip": [start, length],
        "norms": [start, length]
      }
    },
    "<norm>": { "kind": "raw", "dtype": "f16", "shape": [H], "offsets": { "data": [start, length] } }
  }
}
```
`quantization` 取值：`q1`|`q2`|`q3`|`q4`|`q2.54`|`q3.26`。

### 4.2 `weight.bin`
- 小端；各段连续拼接；`offsets` 为字节 `[start, length]`。
- codebook：`packed_indices` = uint8 紧凑位打包；其余附属为 **fp16 LE** 原始字节。
- raw：fp16 LE（默认）或 fp32 LE（`dtype` 标明）。

### 4.3 tokenizer
从 HF 拷贝若存在的文件：`tokenizer.json`、`tokenizer.model`、`tokenizer_config.json`、`special_tokens_map.json`、`vocab.json`、`merges.txt`。

### 4.4 预留（不实现 loader）
字段：`codebook`、`input_scale`、`input_scale_recip`、`norms`、`packed_indices`。可选后续：`rotation`、`permutation`（本版本不写旋转矩阵进 bin，旋转已吸收进量化前权重）。

## 5. 异常

| 类型 | 触发 |
|------|------|
| `QuantError` | 非法 bits；非有限权重；`group_size` 非法；pad 后仍无法分组；码本 `k<1` |
| `FormatError` | bundle 缺文件、offsets 越界、`format` 不匹配、段长度不符 |
| `ShapeMismatchError` | 索引数与 shape 不符、dequant 形状错误 |
| `ModelFetchError` | HF 拉取失败（network/auth/missing）或依赖未安装 |
| `ConfigError` | `config.yaml` / 模型 config 缺关键字段 |
| `UnsupportedError` | 未支持的操作（保留） |

禁止 `except: pass` / bare except 吞错。

## 6. 验收标准

1. `python -m unittest discover -s tests -t .` 全绿。
2. Hadamard：`H@H.T ≈ I`（atol 1e-5）；旋转保范（相对误差 < 1e-4）；非 2 幂 pad 路径可测。
3. 1–4 bit：随机 `W` → quant → dequant，相对 RMSE  
   `rmse / (rms(W_rot)+ε)` 上界：q4 ≤ 0.35，q3 ≤ 0.50，q2 ≤ 0.75，q1 ≤ 1.10（高斯权重、`group_size=32`）。
4. pack/unpack 往返索引完全一致。
5. `allocate_mixed_bits` 对合成层名列表平均 bit 落入 §3.4 区间。
6. `--tiny`：写出 bundle → `load_bundle` → 各 codebook 张量可 dequant；`config.quantization` 正确。
7. CLI 可解析 `--bits 1|2|3|4|2.54|3.26`；非法 bits 抛 `QuantError`。
8. 真实模型（手工/可选）：至少对一层 2D 权重跑 q4，相对重建误差中位数 ≤ 0.35。

## 7. 目录与依赖

```
model/
  AGENTS.md, requirements.md, task.md, requirements.txt, pyproject.toml
  common/{errors,hadamard,codebook,quant,pack,bundle,hf_utils,cli}.py
  gemma/gemma-4-e2b-it/{quantize.py,config.yaml}
  qwen/qwen3.5-2b/{quantize.py,config.yaml}
  tests/test_*.py
```

依赖：`numpy`（必需）；`safetensors`、`huggingface_hub`、`pyyaml`（HF/配置路径）；`torch`/`transformers`（可选，完整模型）。

## 8. 审核检查表（人工）

- [ ] §2 功能边界与非目标可接受
- [ ] §3 API 与码本/scale 语义可接受（index-only 1-bit）
- [ ] §3.4 混合精度启发式可接受
- [ ] §4 bundle 布局可接受（无运行时 graph 二进制）
- [ ] §5–6 异常与验收阈值可接受
- [ ] 默认 HF model id 正确（或接受 `--model` 覆盖）

审核通过后：根据本文件生成 `task.md`（T0–T6）并实施。
