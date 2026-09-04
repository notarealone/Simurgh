# Stage 1, run B (nano-era labels) — anchored `hard_neg` (filters off)
> **Nano-era report.** This run trained on the retired `gpt-5.4-nano` labels, now archived at
> `data/ropg_kd/old_v2/`; its baseline row is the nano-era baseline. The run B reported by the
> thesis is the post-`ff90cf6` retrain in [ropg-runs-comparison-v1](ropg-runs-comparison-v1.md)
> (best epoch 1, nDCG@5 0.7307, Recall@5 0.6461). Numbers from these two files must never be
> mixed.

**Config:** `configs/train_ropg.yaml` at `mode: hard_neg`, `format: triplets`,
`lr: 5.0e-5`, `epochs: 3`, `max_negatives: 8`, `anchor.mode: both`
(`lambda_doc: 0.5`, `lambda_query: 0.05`), `seed: 42`, `world_size: 2` (2×T4).
**Data:** `data/ropg_kd`, derived by `configs/datagen_ropg.yaml` with
`triplets.filters.enabled: false` — 1296/1296 train groups retained, 8 negatives per
group. **Eval:** 276 val groups (92 per persona) ranked against the full 171-chunk
corpus.

Row B of the A–D table in [experiment-design](../experiment-design.md). **Writing-time state
(2026-08-20):** runs A, C and D had not been run when this was written, so **nothing here was
yet attributable to anchoring specifically** — run B differed from the earlier failed `hard_neg`
arm in four ways at once (lr 2e-4 → 5e-5, epochs 5 → 3, negatives 4 → 8, anchoring off → on).
The completed A–D set is in [ropg-runs-comparison-v1](ropg-runs-comparison-v1.md).

## Result

Best epoch 2 (selected by overall Recall@5, per `is_better`):

| Metric | baseline (epoch 0) | epoch 2 | Δ |
|---|---|---|---|
| **nDCG@1** | 0.543 | **0.615** | **+0.072** |
| **Recall@5** | 0.3961 | **0.4324** | **+0.0363** |
| MRR | 0.6019 | 0.6561 | +0.0542 |
| Hit@1 | 0.460 | 0.547 | +0.087 |
| Hit@5 | 0.786 | 0.819 | +0.033 |
| nDCG@5 | **0.548** | 0.522 | **−0.026** |
| val MNRL loss | 1.7905 | 1.0609 | −0.7296 |

Full curves:

| Epoch | train | val | nDCG@1..5 | Hit@1..5 | Recall@5 | MRR |
|---|---|---|---|---|---|---|
| 0 (untrained) | — | 1.7905 | 0.543 0.526 0.533 0.541 0.548 | 0.460 0.565 0.678 0.736 0.786 | 0.3961 | 0.6019 |
| 1 | 1.2090 | 1.1142 | 0.609 0.558 0.538 0.526 0.515 | 0.547 0.652 0.710 0.772 0.801 | 0.4130 | 0.6603 |
| **2** | 0.6811 | **1.0609** | 0.615 0.559 0.540 0.525 0.522 | 0.547 0.638 0.710 0.757 0.819 | **0.4324** | 0.6561 |
| 3 | 0.5710 | 1.1031 | 0.570 0.519 0.494 0.480 0.473 | 0.493 0.601 0.659 0.707 0.750 | 0.3853 | 0.6102 |

Per persona at epoch 2 (baseline in brackets):

| Persona | nDCG@5 | Recall@5 | MRR |
|---|---|---|---|
| crammer | 0.5420 (0.5487) | 0.4819 (0.4275) | 0.7201 (0.5857) |
| scholar | 0.4726 (0.5368) | 0.3696 (0.3623) | 0.5829 (0.5768) |
| steady | 0.5503 (0.5591) | 0.4457 (0.3986) | 0.6653 (0.6432) |

Epoch 3 degrades on every metric while train loss keeps falling — the 3-epoch budget is
right, and epoch 2 is the checkpoint.

## Reading

**This run did not meet its stated success criterion.** That criterion was the nano-era
`nDCG@5 > 0.548`, and nDCG@5 fell to 0.522. Five of six metrics improved and one fell,
and the one that fell was the one being scored on. Recording that plainly is the point
of this file.

**The split is the objective's signature, not a contradiction.** MNRL with hard
negatives trains only the rank-1-vs-tail contrast, where the mean teacher gap is 0.711.
nDCG@5 spends **53.4% of its mass** on slots 2–5, where adjacent teacher gaps are 0.075
and 0.049 — at the single-label noise floor. A rank-1 objective improving nDCG@1 by
0.072 while nDCG@5 slips 0.026 is what that combination predicts. This is why the
primary metric was revised to the nDCG@1 + Recall@5 pair; the reasoning, and the fact
that it was revised after seeing this result, are recorded in
[experiment-design](../experiment-design.md), "Primary metric — revised after run B".

**A confound this run cannot rule out.** `evaluate_retrieval` assigns gain 0 to any
corpus chunk outside a group's ~20 judged ones — about 150 of 171 chunks. A model that
improved by surfacing *unjudged but relevant* chunks into slots 2–5 is penalised for it.
Recall@5 up with nDCG@5 down fits that story as well as it fits "the graded middle got
worse". `judged@5` was added afterwards precisely to separate the two, so **this
question stays open for run B and is answerable from run C onward.**

**Personalisation is not yet demonstrated.** The gains are broad rather than
persona-differentiated: MRR rises for all three personas, and scholar's nDCG@5 actually
falls 0.064. Nothing in the MNRL objective forces the encoder to use the persona prefix,
so a purely generic retrieval improvement would produce a table that looks like this
one. The `persona_swap` control was added afterwards to settle it and, like `judged@5`,
was not available for this run.

## Significance

The shipped significance numbers were produced by `benchmarks/compare_runs.py` over the
`models/ropg/ropg_kd_run*_model/*/training_log.json` files. Its outputs are committed in
`models/ropg/comparisons/`.

For example, the run-A-vs-run-B comparison used:

```bash
uv run python benchmarks/compare_runs.py \
    models/ropg/ropg_kd_runA_model/*/training_log.json \
    models/ropg/ropg_kd_runB_model/*/training_log.json \
    --out models/ropg/comparisons/runA_vs_runB.md
```

The same command produced the other pairwise reports and, with `--swap`, the
persona-swap reports. This nano-era report's figures remain separate from those
completed Luna-era comparisons.

## What this changed

- Primary metric → nDCG@1 + Recall@5, pre-registered before runs C and A.
- `judged@5` and the `persona_swap` control added to `evaluate_retrieval`.
- `benchmarks/compare_runs.py` written.
- Next: **run C** (`data/ropg_kd_filtered`, anchor `both`), then **run A**
  (`data/ropg_kd`, anchor `none`) — the control that makes anchoring attributable.
