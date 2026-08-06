# Gen quant eval archive — Qwen3-0.6B

归档日期：2026-08-06。  
对象：`Qwen/Qwen3-0.6B` Aria bundles（`format_version=2`，blocked Hadamard）。  
目的：对比若干量化配方的 **短生成一致性**，并记录「新异构方案」为何不采纳。

本地 JSON 报告（勿提交 Git）：`audit_layer.json`、`audit-gen.json`、
`qwen3-0.6b_q4_channel-audit_gen.json`、`qwen3-0.6b_q326_channel-audit_gen.json`、
`qwen3-0.6b_q8-audit_gen.json`。

---

## 1. 评测方法

入口：`python -m common.audit_cli gen`（见 `common/gen_compare.py`）。

| 项 | 设置 |
|----|------|
| Model | `Qwen/Qwen3-0.6B` |
| Kind | `text` |
| Prompts（默认 completion 风格） | `The capital of France is` / `2 + 2 =` / `Complete: The sky is` |
| Generate | greedy，`min_new_tokens=8`，`max_new_tokens=32` |
| 权重注入 | `reconstruct_weight` → HF `state_dict`（约 198 张量） |
| 指标 | **new tokens only**：`token_overlap`、`exact_prefix_*`、teacher-forced `mean_logprob_*` on **baseline** continuation |
| CI | `ci_fail: false`（仅报告） |

层抽检（对照）：`audit_cli layer --ref hf`，q4+group 采样 8 层，`rel_rmse_orig≈0.10`，`fail_count=0`。

---

## 2. 方案定义

| 代号 | CLI | 说明 |
|------|-----|------|
| **A — q4+group**（基线） | `--bits 4`（默认 `--codebook-share group`） | 全层 4-bit，group 码本 |
| **B — q4+channel** | `--bits 4 --codebook-share channel` | 全层 4-bit，channel 码本 |
| **C — q326+channel** | `--bits 3.26 --codebook-share channel` | 敏感层优先 4、其余约 3；**全层** channel |
| **D — q8+group** | `--bits 8` | 全层 8-bit，group 码本（似然/位宽天花板对照） |
| **E — 新异构（未实现）** | 提议：其余 `4+group`，敏感 compute `8+channel` | 按层 bits + 按层 share；**不采纳**（见 §5） |

产物命名约定：`./out/<slug>_<quant>`，其中 C 使用 `_q326_channel`（见 README）。

---

## 3. Gen 总表（三 prompt 均值）

| 方案 | `mean_token_overlap` | `mean_exact_prefix_frac` | `mean_logprob_delta` |
|------|---------------------:|-------------------------:|---------------------:|
| A q4+group | 0.1878 | 0.0729 | −0.172159 |
| B q4+channel | 0.2886 | 0.0312 | **+0.046007** |
| **C q326+channel** | **0.5244** | **0.3854** | −0.059476 |
| D q8+group | **0.6429** | **0.3854** | **+0.00685** |

要点：

- **前缀一致性**：A/B 很弱；**C 与 D 均值并列最高（0.385）**。
- **似然贴合（Δlogprob）**：D ≈ 0 最好；B 亦为正；A 最差。
- **仅开 channel（B）**：抬高 overlap / 似然，但 **压低** greedy 前缀 → 不能单靠全层 channel 抬 gen 前缀。

---

## 4. 分样本摘要

### 4.1 The capital of France is

| 方案 | prefix_len / frac | exact | Δlogprob | 备注 |
|------|------------------:|:-----:|---------:|------|
| A | 5 / 0.156 | no | −0.170 | 对齐到 “ Paris.” 后复读 |
| B | 1 / 0.031 | no | +0.007 | 标点分叉；似然已对齐 |
| **C** | **32 / 1.0** | **yes** | +0.010 | 与 FP **逐 token 一致** |
| D | 1 / 0.031 | no | −0.003 | 全 q8 仍可能在 “ Paris” 后分叉 |

### 4.2 2 + 2 =

| 方案 | prefix_len / frac | Δlogprob | 备注 |
|------|------------------:|---------:|------|
| A | 2 / 0.063 | −0.276 | 先到 “ 4”，后加法链 |
| B | 2 / 0.063 | +0.035 | 解题模板风格 |
| C | **5 / 0.156** | −0.124 | 最长前缀 |
| D | 4 / 0.125 | −0.038 | 略逊于 C |

### 4.3 Complete: The sky is

| 方案 | prefix_len / frac | exact | Δlogprob | 备注 |
|------|------------------:|:-----:|---------:|------|
| A | 0 | no | −0.070 | clear vs dark |
| B | 0 | no | +0.096 | clear vs blue；似然最好之一 |
| C | 0 | no | −0.064 | 仍零前缀 |
| **D** | **32 / 1.0** | **yes** | +0.061 | 开放补全被全 q8 救回 |

均值相同的 C/D：C 赢在 France exact；D 赢在 sky exact；算术两者接近。

---

## 5. 新异构方案（E）vs C — 为何不采纳

提议 E：**非敏感 `4+group` + 敏感 compute（attn/lm_head）`8+channel`**（按层 share；embed 不进敏感）。

| 维度 | C q326+channel（已实测） | E 新异构（推断） |
|------|--------------------------|------------------|
| 最高 bit | 4 | 8（仅敏感） |
| 最低 bit | ~3 | 4 |
| Channel | **全层** | **仅敏感** |
| Gen 前缀 | 已与全 q8 **同级均值** | **难再数量级提升**；方差仍由单条 prompt 主导 |
| 工程 | 现成 CLI | 需改 Spec（混合出 8）+ 分配器 |
| 主要价值叙事 | 生成质量配方 | 体积/结构权衡，非稳压 C |

**结论：** 以「抬 gen」为第一目标时，**采用 C（文档化为全家族推荐）**；不实现 E。若将来要省全层 channel 体积，再单独开 Spec。

---

## 6. 决策与文档落点

| 决策 | 内容 |
|------|------|
| 推荐配方 | `--bits 3.26 --codebook-share channel` → `./out/<slug>_q326_channel` |
| 基线对照 | 保留 q4 / q8；channel-only q4 仅作消融 |
| README | 全家族已补充 `_q326_channel` 命令（`README.md` / `README_cn.md`） |
| serve | 目录仍为 int4/int8 两槽；**不**因本评测改 harness（serve 无 AGENTS/requirements/task） |
| engine | 已支持按张量 `bits` / `codebook_share`；本评测无引擎改动 |

复现示例：

```bash
python qwen/qwen3-0.6b/quantize.py \
  --bits 3.26 --codebook-share channel \
  --out ./out/qwen3-0.6b_q326_channel

python -m common.audit_cli gen \
  --bundle ./out/qwen3-0.6b_q326_channel \
  --model Qwen/Qwen3-0.6B --kind text \
  --min-new-tokens 8 --max-new-tokens 32 \
  --report ./out/qwen3-0.6b_q326_channel/audit_gen.json
```

---

## 7. 局限

- 仅 **3 条短 prompt × 32 new tokens**；均值对单条（France / sky）极敏感。
- Baseline 自身存在复读；不完全是量化独有问题。
- 仅 Qwen3-0.6B；其它家族需按 README 命令各自复测。
- Layer RMSE 有界 **≠** greedy 前缀高；以 `exact_prefix_frac` + `mean_logprob_delta` 为准看 gen。
