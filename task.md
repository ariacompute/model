# task.md — 旋转码本量化实施清单

依据 [`requirements.md`](requirements.md)。完成后勾选。

## T0–T8 — 既有管线
- [x] 脚手架 / Hadamard / 码本 / pack·bundle / 混合 2.54·3.26 / HF·CLI / 回归
- [x] 流式不变量 + `codebook_share=group` + `q1.5` PLE 加权

## T9 — int8 码本（K=256）
- [x] `pack` / `quant` / `cli` 支持 `--bits 8` → `q8`
- [x] 单测：pack roundtrip、RMSE、`--tiny --bits 8`

## T10 — 多家族注册
- [x] §1.1 全部目录：`quantize.py` + `config.yaml`
- [x] `tests/test_families.py` 锁定 base_model
- [x] Spec / AGENTS / README 同步
