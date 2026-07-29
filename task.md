# task.md — 旋转码本量化实施清单

依据已审核的 [`requirements.md`](requirements.md)。完成后勾选。

## T0 — 脚手架
- [x] `requirements.txt` / `pyproject.toml`
- [x] `common/__init__.py`、`errors.py`
- [x] `.gitignore` 增加 `**/weights/`、`*.bin` 权重产物
- [x] 家族目录 `config.yaml` 占位

## T1 — Hadamard
- [x] `common/hadamard.py`：`hadamard_matrix` / `next_pow2` / `hadamard_rotate`
- [x] `tests/test_hadamard.py`：正交、保范、非 2 幂 pad

## T2 — 码本
- [x] `common/codebook.py`：`lloyd_max` / `quantize_group`
- [x] `tests/test_codebook.py`

## T3 — quant + pack + bundle
- [x] `common/pack.py`
- [x] `common/quant.py`：`QuantTensor` / `quantize_weight` / `dequantize` / `allocate_mixed_bits` / `parse_bits`
- [x] `common/bundle.py`：`write_bundle` / `load_bundle`
- [x] `tests/test_pack.py`、`test_quant.py`、`test_bundle.py`

## T4 — 混合精度
- [x] `allocate_mixed_bits` 对 2.54 / 3.26 平均 bit 区间验收（单测）

## T5 — HF utils + CLI + 家族脚本
- [x] `common/hf_utils.py`：tiny + 可选 HF 流式
- [x] `common/cli.py`
- [x] `gemma/gemma-4-e2b-it/quantize.py` + `config.yaml`
- [x] `qwen/qwen3.5-2b/quantize.py` + `config.yaml`

## T6 — 端到端与回归
- [x] `tests/test_cli_tiny.py`：`--tiny` bundle roundtrip
- [x] `tests/test_errors.py`：非法 bits 等
- [x] `python -m unittest discover -s tests -t .` 全绿
