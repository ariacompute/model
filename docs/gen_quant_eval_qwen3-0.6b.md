# Gen quant eval archive — Qwen3-0.6B

Subject: `Qwen/Qwen3-0.6B` Aria bundles (`format_version=2`, blocked Hadamard).  
Goal: compare short-generation consistency across quant recipes, and record why the proposed **heterogeneous** profile was not adopted.

Local JSON reports (do not commit): `qwen3-0.6b_q4-audit_layer.json`, `qwen3-0.6b_q4-audit-gen.json`,
`qwen3-0.6b_q4_channel-audit_gen.json`, `qwen3-0.6b_q326_channel-audit_gen.json`,
`qwen3-0.6b_q8-audit_gen.json`.

---

## 1. Method

Entry point: `python -m common.audit_cli gen` (see `common/gen_compare.py`).

| Item | Setting |
|------|---------|
| Model | `Qwen/Qwen3-0.6B` |
| Kind | `text` |
| Prompts (default completion-style) | `The capital of France is` / `2 + 2 =` / `Complete: The sky is` |
| Generate | greedy, `min_new_tokens=8`, `max_new_tokens=32` |
| Weight inject | `reconstruct_weight` → HF `state_dict` (~198 tensors) |
| Metrics | **new tokens only**: `token_overlap`, `exact_prefix_*`, teacher-forced `mean_logprob_*` on the **baseline** continuation |
| CI | `ci_fail: false` (report-only) |

Layer audit (context): `audit_cli layer --ref hf`, q4+group, sample 8 layers, `rel_rmse_orig≈0.10`, `fail_count=0`.

---

## 2. Recipes

| ID | CLI | Notes |
|----|-----|-------|
| **A — q4+group** (baseline) | `--bits 4` (default `--codebook-share group`) | All layers 4-bit, group codebooks |
| **B — q4+channel** | `--bits 4 --codebook-share channel` | All layers 4-bit, channel codebooks |
| **C — q326+channel** | `--bits 3.26 --codebook-share channel` | Sensitive layers preferentially 4-bit, others ~3-bit; **global** channel |
| **D — q8+group** | `--bits 8` | All layers 8-bit, group codebooks (likelihood / bit-width ceiling) |
| **E — hetero (not implemented)** | Proposed: rest `4+group`, sensitive compute `8+channel` | Per-tensor bits + share; **rejected** (see §5) |

Output naming: `./out/<slug>_<quant>`; recipe C uses `_q326_channel` (see README).

---

## 3. Gen summary (mean over 3 prompts)

| Recipe | `mean_token_overlap` | `mean_exact_prefix_frac` | `mean_logprob_delta` |
|--------|---------------------:|-------------------------:|---------------------:|
| A q4+group | 0.1878 | 0.0729 | −0.172159 |
| B q4+channel | 0.2886 | 0.0312 | **+0.046007** |
| **C q326+channel** | **0.5244** | **0.3854** | −0.059476 |
| D q8+group | **0.6429** | **0.3854** | **+0.00685** |

Takeaways:

- **Prefix match:** A/B are weak; **C and D tie for best mean (0.385)**.
- **Likelihood fit (Δlogprob):** D ≈ 0 is best; B is also positive; A is worst.
- **Channel-only (B):** improves overlap / likelihood but **hurts** greedy prefix → full-layer channel alone does not raise gen prefix.

---

## 4. Per-prompt notes

### 4.1 The capital of France is

| Recipe | prefix_len / frac | exact | Δlogprob | Notes |
|--------|------------------:|:-----:|---------:|-------|
| A | 5 / 0.156 | no | −0.170 | Matches through “ Paris.” then loops |
| B | 1 / 0.031 | no | +0.007 | Punctuation fork; likelihood already aligned |
| **C** | **32 / 1.0** | **yes** | +0.010 | **Token-identical** to FP |
| D | 1 / 0.031 | no | −0.003 | Full q8 can still fork after “ Paris” |

### 4.2 2 + 2 =

| Recipe | prefix_len / frac | Δlogprob | Notes |
|--------|------------------:|---------:|-------|
| A | 2 / 0.063 | −0.276 | Hits “ 4”, then additive chain |
| B | 2 / 0.063 | +0.035 | Solution-template style |
| C | **5 / 0.156** | −0.124 | Longest prefix |
| D | 4 / 0.125 | −0.038 | Slightly behind C |

### 4.3 Complete: The sky is

| Recipe | prefix_len / frac | exact | Δlogprob | Notes |
|--------|------------------:|:-----:|---------:|-------|
| A | 0 | no | −0.070 | clear vs dark |
| B | 0 | no | +0.096 | clear vs blue; strong likelihood |
| C | 0 | no | −0.064 | Still zero prefix |
| **D** | **32 / 1.0** | **yes** | +0.061 | Open completion recovered by full q8 |

Same mean for C/D: C wins France exact; D wins sky exact; arithmetic is close.

---

## 5. Heterogeneous recipe (E) vs C — why not adopted

Proposed E: **non-sensitive `4+group` + sensitive compute (attn / lm_head) `8+channel`** (per-tensor share; keep embed out of the sensitive set).

| Dimension | C q326+channel (measured) | E hetero (inferred) |
|-----------|---------------------------|---------------------|
| Max bits | 4 | 8 (sensitive only) |
| Min bits | ~3 | 4 |
| Channel | **all layers** | **sensitive only** |
| Gen prefix | Already **ties** full-q8 mean | **Unlikely another order-of-magnitude gain**; variance still prompt-dominated |
| Engineering | Existing CLI | Spec change (mixed may emit 8) + allocator |
| Product story | Generation-quality recipe | Size / structure tradeoff, not a guaranteed win over C |

**Decision:** With “raise gen quality” as the primary goal, **ship C** (documented as the recommended recipe for all families); do **not** implement E. Revisit E only if saving full-layer channel size becomes a separate Spec.

---

## 6. Decisions and doc surface

| Decision | Detail |
|----------|--------|
| Recommended recipe | `--bits 3.26 --codebook-share channel` → `./out/<slug>_q326_channel` |
| Baselines | Keep q4 / q8; channel-only q4 is ablation only |
| README | All families list `_q326_channel` commands (`README.md` / `README_cn.md`) |
| serve | Catalog remains int4/int8 slots; **no** harness change from this eval (serve has no AGENTS/requirements/task) |
| engine | Already supports per-tensor `bits` / `codebook_share`; no engine change for this eval |

Reproduce:

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

## 7. Limitations

- Only **3 short prompts × 32 new tokens**; means are highly sensitive to single rows (France / sky).
- The FP baseline itself can loop; not all failure modes are quant-only.
- Qwen3-0.6B only; re-run other families via README commands as needed.
- Bounded layer RMSE **≠** high greedy prefix; judge gen with `exact_prefix_frac` + `mean_logprob_delta`.
