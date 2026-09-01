# Judge agreement: Luna labels vs. Gemini tournament judges, v1

## Finding

Two Gemini judges agree with the `gpt-5.6-luna` preference labels about **65–70%** of the
time on decisive verdicts — far above chance, far below reproducible. Both are consistent
with a single underlying rate near 0.70.

The more useful number is not in either printed report. On the 100 pairs both judges saw,
the two Gemini models agree **with each other** on 65 of 89 decisive verdicts (73.0%,
95% CI [63.0%, 81.2%]). Gemini-vs-Gemini reproducibility is statistically
indistinguishable from Gemini-vs-Luna agreement. The ~30% disagreement is therefore a
property of the task under this rubric, not a Luna idiosyncrasy: on roughly a third of
pairs the rubric does not determine a winner, and any judge — including the one scoring the
tournament — will split near-randomly on them.

This bounds what Stage 2 can measure. A trained arm that improved the *true* quality of
every rewrite would still lose ~30% of its tournament pairings to rubric ambiguity, and the
tournament's ±5-point resolution at $n = 272$ sits inside that noise floor. It does not
explain the seed-42 negative result — the training metrics do that, see
[dpo-arms-seed42-v1](dpo-arms-seed42-v1.md) — but it means no realistic amount of tuning
against this metric can move the score much past the low 0.5s.

## Inputs

| Input | Value |
|---|---|
| Script | `benchmarks/judge_agreement.py`, `--seed 42`, config `configs/train_dpo.yaml` |
| Pairs | `data/dpo/train.jsonl`, 1,290 question/persona keys after dropping normalized-equal pairs |
| Sampling | One pair per key, `random.sample` at seed 42 |
| Labeller under test | `gpt-5.6-luna`, `reasoning_effort: none` — the labels in `chosen`/`rejected` |
| Judges | `gemini-3.7-flash`, `gemini-3.5-flash-lite`, native GenAI protocol, `thinking_level: minimal` |
| Prompt and schema | `_JUDGE_SYSTEM` / `_JUDGE_SCHEMA` from `benchmarks/compare_dpo_rewriters.py`, byte-identical to the tournament's |
| Orientation | Balanced — the Luna-chosen rewrite is candidate A on even indices |
| Artifacts | `data/rl/agreement/train-n{10,100,250}-seed42.{json,jsonl}` (uncommitted) |

Reusing the tournament's exact prompt and schema is deliberate: a measured disagreement
cannot be dismissed as a prompt artifact, because this *is* the prompt that produced the
tournament ranking.

Every run judged every pair, `missing: 0` throughout.

## Runs as reported

| Judge | $n$ | Agree | Disagree | Tie | Decisive rate | 95% CI | $p$ vs chance |
|---|---:|---:|---:|---:|---:|---|---:|
| `gemini-3.7-flash` | 10 | 6 | 3 | 1 | 0.667 | — | pilot |
| `gemini-3.7-flash` | 100 | 64 | 29 | 7 | **0.688** | [0.588, 0.773] | 3.66e-04 |
| `gemini-3.5-flash-lite` | 250 | 150 | 84 | 16 | **0.641** | [0.578, 0.700] | 1.91e-05 |

Tie rates are stable and low: 7.0% and 6.4%. Position bias is absent in both — A 50 / B 43
($p = 0.534$) and A 109 / B 125 ($p = 0.327$) — so the balanced orientation worked and the
agreement figures are not contaminated by a left/right preference.

## The two runs are nested, and the headline gap is sampling noise

`random.sample` at a fixed seed draws sequentially, so the 100-key draw is a strict prefix
of the 250-key draw. Verified on the artifacts: all 100 keys of the `n=100` run appear in
the `n=250` run, and `chosen_position` matches on every one. The two rows in the table above
are **not** independent samples, and the apparent 4.7-point advantage for `gemini-3.7-flash`
compares different item sets.

Restricted to the 100 shared keys, the ordering reverses:

| Judge | Agree | Disagree | Tie | Decisive rate | 95% CI |
|---|---:|---:|---:|---:|---|
| `gemini-3.7-flash` | 64 | 29 | 7 | 0.688 | [0.588, 0.773] |
| `gemini-3.5-flash-lite` | 68 | 26 | 6 | **0.723** | [0.626, 0.804] |

Paired on the 89 keys where both judges were decisive: both agree with Luna on 51, both
disagree on 14, only `gemini-3.7-flash` agrees on 10, only `gemini-3.5-flash-lite` agrees on
14. Exact McNemar $p = 0.541$. **No detectable difference between the two judges.**

The same run also shows how much variance $n \approx 100$ carries. Splitting the
`gemini-3.5-flash-lite` run into the 100 shared keys and the 150 keys unique to it, one
judge on one dataset produces:

| Subset | Agree / decisive | Rate | 95% CI | $p$ vs chance |
|---|---:|---:|---|---:|
| Shared 100 keys | 68 / 94 | 0.723 | [0.626, 0.804] | 1.7e-05 |
| Extra 150 keys | 82 / 140 | 0.586 | [0.503, 0.664] | 0.052 |

A 13.7-point swing between two halves of a single run, one of which is barely
distinguishable from chance. Any future agreement number below $n \approx 400$ should be
read as ±8 points, and per-persona cells below $n \approx 30$ should not be read at all.

## Per persona

As printed, decisive verdicts only:

| Persona | `gemini-3.7-flash` $n{=}100$ | `gemini-3.5-flash-lite` $n{=}250$ |
|---|---:|---:|
| `crammer` | 27/29 = 0.931 | 45/72 = 0.625 |
| `scholar` | 18/31 = 0.581 | 48/80 = 0.600 |
| `steady` | 19/33 = 0.576 | 57/82 = 0.695 |

The `crammer` divergence looks dramatic and is not real. On the shared 100 keys the same
cells read `crammer` 0.931 vs 0.774, `scholar` 0.581 vs 0.594, `steady` 0.576 vs 0.806 —
the ranking of personas is not preserved between judges, and every cell has $n \le 34$.
Nothing here survives the variance shown in the previous section. Persona-level agreement is
**not established** by these runs.

Note this is the second place where `crammer` behaves unlike the others; the tournament's
per-persona split also singled it out (grok 0.821 crammer vs 0.510 steady). Whether those
are the same effect is untested.

## What this settles and what it does not

Settled:

- Luna labels are not noise. Agreement is 3–4 standard errors above chance in both runs.
- Luna labels are not a gold standard either. ~30% of pairs are rubric-ambiguous.
- The tournament judge is not more self-consistent than it is Luna-consistent (73.0% vs
  68.8% on identical items), so swapping the labeller would not raise the ceiling.
- Judge choice between `gemini-3.7-flash` and `gemini-3.5-flash-lite` does not matter for
  agreement (McNemar $p = 0.541$). The cheaper model is fine.

Not settled:

- Whether the ~30% disagreement is symmetric noise or a systematic rubric split. Reading
  the `reason` fields on the 24 keys where the two Gemini judges contradict each other would
  answer this, and would say whether the rubric can be tightened.
- Whether agreement differs by persona. Needs $n \ge 100$ per persona.
- Whether the same rate holds on `data/dpo/val.jsonl`, which is what the tournament actually
  scores. All runs here sampled the train split.

## Reproduction

```bash
uv run --frozen python benchmarks/judge_agreement.py --sample 250
```

Credentials come from `.env` (`DPO_EVAL_BASE_URL` as a bare host, `DPO_EVAL_API_KEY`,
`DPO_EVAL_MODEL`). The script preflights the endpoint with one request and exits non-zero if
no pair was judged, so a misconfigured run fails in a second instead of reporting `nan`.
Nested sampling at a fixed seed means `--sample 250` re-judges the `--sample 100` items
first; use a different `--seed` for a genuinely independent sample.
