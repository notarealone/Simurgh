# Results

Benchmark results and analysis land here — per-rung tables, per-persona retrieval
diagnostics, and significance reports (paired tests + bootstrap CIs over ≥3 seeds).

Layout (created as runs produce them):

- One subdirectory or file per baseline-ladder rung (see [experiment-design](../experiment-design.md)).
- Each result references the `configs/*.yaml` and seed that produced it (reproducible by construction).

Committed results:

- [stage1-ropg-runB](stage1-ropg-runB.md) — Stage 1 ROPG, selected run.
- [ropg-runs-comparison-v1](ropg-runs-comparison-v1.md) — Stage 1 run comparison.
- [dpo-arms-seed42-v1](dpo-arms-seed42-v1.md) — Stage 2 DPO/WPO/robust screening at seed 42 on v3 pair data; no arm promoted.
- [dpo-arms-seed42-v2](dpo-arms-seed42-v2.md) — the same three arms rerun on v4 pair data with the repaired trainer; `wpo` beats the untrained policy, with figures under `figures/dpo-v4-vs-v3/`.
- [eval-ablation-v1](eval-ablation-v1.md) — the seven-arm ladder on the held-out test split: ROPG is the load-bearing rung, and rewriting is a regression whether it targets the query vector (rungs 4, 5) or the question the generator answers (rungs 4.1, 5.1), with figures under `figures/eval-ablation-v1/`.
- [judge-agreement-luna-gemini-v1](judge-agreement-luna-gemini-v1.md) — how far the Luna preference labels and the Gemini tournament judge agree, and the noise floor that implies.
- [judge-noise-luna-v1](judge-noise-luna-v1.md) — Luna replicate noise at `temperature: 0`, and the `filters.min_margin` calibration it produced.
- [dpo-data-v4-audit](dpo-data-v4-audit.md) — as-built audit of the fourth DPO pair generation (second schema): yield, label noise, persona skew, cleanliness.

Raw model weights, large prediction dumps, and the built index are **not** committed here
(see the repo rules in [CLAUDE](../../CLAUDE.md)); only summary tables/plots and analysis.
