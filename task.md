# task.md — 旋转码本量化实施清单

依据 [`requirements.md`](requirements.md)。完成后勾选。

## T0 — 脚手架
- [x] `requirements.txt` / `pyproject.toml`
- [x] `common/__init__.py`、`errors.py`
- [x] `.gitignore`：`**/weights/`、`*.bin`、`__pycache__` / `*.pyc`
- [x] 家族目录 `config.yaml` 占位

## T1 — Hadamard
- [x] `common/hadamard.py`：FWHT / 分块 `H@W`（axis=0，禁止 skip）
- [x] `tests/test_hadamard.py`

## T2 — 码本
- [x] `common/codebook.py`：`lloyd_max` / columns / 可选 CUDA
- [x] `tests/test_codebook.py`

## T3 — quant + pack + bundle
- [x] `common/pack.py` / `quant.py` / `bundle.py`（`BundleWriter` 流式）
- [x] `tests/test_pack.py`、`test_quant.py`、`test_bundle.py`

## T4 — 混合精度（层数）
- [x] `allocate_mixed_bits` 2.54 / 3.26 单测

## T5 — HF + CLI + 家族
- [x] `hf_utils`（BF16→f32 流式）、`cli`、两家族 `quantize.py`

## T6 — 回归
- [x] `test_cli_tiny` / `test_errors`；`unittest` 全绿

## T7 — 流式不变量 + 体积（group 码本）
- [x] `common/runtime.py`：RAM / workers / CUDA 探测
- [x] 流式路径强制 Hadamard+Lloyd-Max（`test_core_guarantees`）
- [x] 默认 `codebook_share=group`；可选 `channel`；去掉未用 aux 落盘
- [x] Spec / README 同步体积预期（Gemma-4-E2B q4：group ~2.5–3 GB，channel ~8 GB）

## T8 — `q1.5`：PLE@1 + 参数加权
- [x] `classify_tensor_role` / `allocate_mixed_bits_weighted`（PLE 默认 1，计算 2/3）
- [x] `parse_bits` / label 支持 `1.5`；CLI `--ple-bits` / `--compute-bits` / `--hi-bits`
- [x] 流式：先收集 `name→numel` 再分配再量化；`config` 可选 `bit_policy` / `avg_bits_weighted`
- [x] 单测：加权 band、PLE 不被凑高 bit、`estimate_index_bytes`、`--tiny --bits 1.5`
- [x] README / AGENTS 文档：`q1.5` 与 Gemma &lt;1GB 目标
