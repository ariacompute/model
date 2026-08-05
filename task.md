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

## T11 — VLA 家族
- [x] `openvla-7b` / `openpi-pi0-3b`（`lerobot/pi0_base`）/ `openpi-pi0.5-3b`（`lerobot/pi05_base`）/ `lingbot-vla-v2-6b`
- [x] README 命令（`--out` 规范化）+ `test_families`

## T12 — 质量审计 A+B
- [x] `common/audit.py`：分层抽检 + rot/orig RMSE + 阈值字段
- [x] `common/gen_compare.py`：text 短生成 / VLA action 前向（可 skip）
- [x] `python -m common.audit_cli`；报告不 fail CI
- [x] 单测 + README

## T13 — group 路径 GPU Lloyd-Max
- [x] `lloyd_max_batched_torch`；`codebook_share=group` 在 CUDA 可用时自动选用
- [x] 无 CUDA 时 CPU numpy（workers 并行）；单测（torch CPU batched）
