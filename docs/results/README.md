# Results

Benchmark results and analysis land here — per-rung tables, per-persona retrieval
diagnostics, and significance reports (paired tests + bootstrap CIs over ≥3 seeds).

Layout (created as runs produce them):

- One subdirectory or file per baseline-ladder rung (see [experiment-design](../experiment-design.md)).
- Each result references the `configs/*.yaml` and seed that produced it (reproducible by construction).

Raw model weights, large prediction dumps, and the built index are **not** committed here
(see the repo rules in [CLAUDE](../../CLAUDE.md)); only summary tables/plots and analysis.
