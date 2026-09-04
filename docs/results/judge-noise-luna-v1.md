# Judge replicate noise: Luna at temperature 0, v1

## Finding

`gpt-5.6-luna` at `temperature: 0.0` is **not deterministic**. Rescoring 300 frozen rewrites
three times each, only **18.0%** of candidates returned byte-identical replies (54/300,
95% CI [14.1%, 22.7%]) and the single-draw aggregate score SD was **0.0265** — p90 0.041,
max 0.137. Zero calls failed, so this is the judge, not retry noise.

That falsifies the premise the v2 config was built on. `judge.samples_per_candidate: 1` was
chosen on the reasoning that "at temperature 0 the replies are near-copies," which is now
measured false; averaging repeated judgements does cancel noise here.

The second finding overturns the calibration this experiment was built to perform.
`filters.min_margin: 0.10` was a guess meant to clear the judge's noise floor. Measured
against the pair production *actually emits*, it was destroying **43% of groups to buy
essentially nothing**: 34/75 groups survive at 0.10 against 62/75 at 0.04, while the
orientation-error rate of the emitted row moves from 7.6% to 7.5%. The threshold was
calibrated against the wrong quantity — see
[the flip-rate-versus-orientation section](#why-the-flip-rate-overstates-the-error-rate).

Effective training signal $n(1-2p)^2$ peaks at **0.04**, on a plateau flat from 0.00 to 0.04.

Two consequences for the thesis narrative. First, the v1 label era — judge at
`temperature: 1.0`, `max_completion_tokens: 16`, bare scalar reply — sat on strictly more
noise than this, so the label-noise diagnosis of the seed-42 negative result
([dpo-arms-seed42-v1](dpo-arms-seed42-v1.md)) is now measured rather than argued. Second,
this measures the judge's **self-consistency, not its correctness**; the ~30% rubric
ambiguity in [judge-agreement-luna-gemini-v1](judge-agreement-luna-gemini-v1.md) is a
separate and larger ceiling that tightening `min_margin` does nothing about.

## Inputs

| Input | Value |
|---|---|
| Script | `benchmarks/judge_noise.py`, `--seed 42`, config `configs/datagen_dpo.yaml` |
| Questions | 25 sampled from `data/splits/train_qids.txt` — 22 `ai_generated_questions`, 2 `khordad1403-keshvari`, 1 `keshvari-unknown_1` |
| Candidates | 300 = 25 questions × 3 train personas × 4 rewriter temperatures, generated once and **frozen** before judging |
| Rewriter | `grok-4-1-fast`, temperatures `[0.2, 0.5, 0.8, 1.1]` |
| Judge under test | `gpt-5.6-luna`, `reasoning_effort: none`, `temperature: 0.0`, strict `json_schema` reply |
| Replicates | 3 per candidate, 900 judge calls, **0 failures** |
| Rubric | `meaning_preservation` 0.40, `persona_fit` 0.35, `specificity` 0.25 |
| Config at run time | `samples_per_candidate: 1`, `min_margin: 0.10`, `min_chosen_score: 0.55`, `max_pair_similarity: 0.95` |
| Artifacts | `data/dpo/judge_noise/samples.jsonl` (900 rows, one per call, raw reply retained), `report_flip10.json` (uncommitted) |
| Code state | `d6df96d`, judge-noise script untracked |

Judging reuses `Rubric`, `_build_judge_messages`, `RetryPolicy` and `_call_with_retry`
imported from `data.gen_dpo_data`, so this is the production judge and not a lookalike — a
measured inconsistency cannot be dismissed as a prompt artifact.

Judge latency: mean 1,566 ms, p90 2,019 ms.

## Determinism and score spread

| Quantity | Value |
|---|---|
| Candidates with byte-identical replies across 3 calls | 54/300 = **0.180** |
| Candidates with identical aggregate scores | 54/300 = 0.180 |
| Single-draw aggregate SD (pooled) | **0.0265** |
| Aggregate SD, mean / p50 / p90 / max | 0.0174 / 0.0114 / 0.0410 / 0.1371 |

Identical replies and identical scores coincide exactly: when the text differs at all, the
numbers differ too. There is no cheap tokenization-only jitter to discount.

### Noise is heteroscedastic, and worst where scores are worst

| Score band | Mean aggregate SD | $n$ |
|---|---:|---:|
| [0.00, 0.40) | 0.0217 | 2 |
| [0.40, 0.55) | **0.0427** | 12 |
| [0.55, 0.70) | **0.0411** | 36 |
| [0.70, 0.85) | 0.0180 | 85 |
| [0.85, 1.01) | 0.0101 | 165 |

**Four times more noise in the mid band than the top band.** The judge is confident about
good rewrites and unsure about mediocre ones. `min_chosen_score: 0.55` sits inside the
noisiest region, though it barely binds — only 1 of 75 groups has a best candidate below
0.55 (5 below 0.60, 7 below 0.70).

### Per criterion, `specificity` is the least reproducible

| Criterion | SD (mean / p90 / max) | Identical across 3 replicates |
|---|---|---:|
| `meaning_preservation` | 0.0174 / 0.0577 / 0.1155 | 175/300 = 0.583 |
| `persona_fit` | 0.0201 / 0.0577 / 0.2021 | 116/300 = 0.387 |
| `specificity` | 0.0261 / 0.0577 / 0.1747 | 99/300 = 0.330 |

The rubric's least stable criterion carries 0.25 weight, and `persona_fit` — the criterion
the whole personalization claim rests on — is reproducible on only 39% of candidates and has
the single worst outlier at 0.202. `meaning_preservation`, the most mechanical judgement, is
the most stable, which is the ordering one would predict and mild evidence the sub-scores
measure what their names say.

The judge uses a fine-grained scale unprompted: 36 distinct sub-score values, including
0.28, 0.42, 0.96, 0.97. Some of the observed SD is the judge resolving detail finer than the
rubric can support.

### Pair noise is independent, as designed

| Quantity | Value |
|---|---|
| Pairs | 450 (2 with identical text) |
| Mean observed \|Δ\| | 0.0736 |
| Pair-difference SD | 0.0271 |
| Independence prediction $\sigma\sqrt2$ | 0.0247 |
| Implied correlation | −0.204 |

Each candidate is scored in its own API call, so ρ ≈ 0 is the correct expectation and is
what the data shows; the −0.204 is estimator noise off 450 pairs × 3 replicates, not a real
anti-correlation. This matters because it licenses the $\sigma_\Delta = \sqrt2\,\sigma$
scaling used to reason about the threshold. Had the two scorings shared prompt context, the
pair noise would not be predictable from the single-candidate noise.

## Flip rate by observed margin

Replicate 1 is the observation; replicates 2 and 3 are the re-runs. A round flips when the
sign of the score difference disagrees with replicate 1's. Δ exactly 0 counts as a flip: the
judge expressed no preference, so the recorded ordering had no basis.

| Observed margin | Rounds | Flip rate |
|---|---:|---:|
| [0.00, 0.02) | 216 | **0.574** |
| [0.02, 0.05) | 236 | 0.250 |
| [0.05, 0.10) | 224 | 0.112 |
| [0.10, 0.15) | 86 | 0.093 |
| [0.15, 0.20) | 62 | 0.032 |
| [0.20, 0.30) | 48 | 0.042 |
| [0.30, 1.01) | 28 | 0.000 |

Pairs that look like a 0.00–0.02 preference are **coin tosses** (57.4% reversal), and 24% of
all candidate pairs fall in that bin. Tail rates, over all rounds with observed margin ≥ t:

| `min_margin` | Rounds | Tail flip rate | Groups yielding a pair |
|---|---:|---:|---:|
| 0.00 | 900 | 0.244 | 74/75 |
| 0.02 | 684 | 0.140 | 71/75 |
| 0.04 | 524 | 0.088 | 62/75 |
| 0.06 | 396 | 0.091 | 52/75 |
| 0.08 | 284 | 0.056 | 42/75 |
| 0.10 *(in force)* | 224 | 0.054 | 34/75 |
| 0.12 | 188 | 0.037 | 29/75 |
| 0.15 | 138 | 0.029 | 23/75 |
| 0.20 | 76 | 0.026 | 17/75 |
| 0.25 | 44 | 0.045 | 9/75 |
| 0.30 | 28 | 0.000 | 5/75 |

The curve is not monotone (0.088 at 0.04 against 0.091 at 0.06; 0.026 at 0.20 against 0.045
at 0.25). Those inversions are sampling noise — the 0.25 row rests on 44 rounds — so
differences of a few tenths of a point should not be read as ordering.

## Why the flip rate overstates the error rate

**The flip table scores all $\binom{4}{2} = 6$ candidate pairs per group. Production emits
only one — best versus worst.** The extreme pair of four candidates has a much wider true
gap than a randomly chosen pair, so the near-tied comparisons driving the 57.4% bin never
reach disk. Reading `min_margin` off the flip curve therefore over-corrects.

The true best-worst gap is wide: mean **0.1288**, p10 0.032, with 53% of groups above 0.10
and only 15% below 0.04.

Re-running the question on the emitted pair only, two ways. **Empirical** uses replicate 1
for selection and the mean of replicates 2–3 as reference ordering; distribution-free, but
the reference has standard error $\sigma/\sqrt2 = 0.019$, so it overstates error.
**Simulated** resamples Gaussian noise at the measured σ around each candidate's
three-replicate mean; it assumes Gaussian homoscedastic noise (false — see the band table)
and treats the replicate mean as truth, so it understates error. Truth is between them.

| `min_margin` | Groups | Empirical error | 95% CI | Simulated ($K{=}1$) | Signal $n(1-2p)^2$ |
|---|---:|---:|---|---:|---:|
| 0.00 | 74 | 7/74 = 0.095 | [0.047, 0.183] | 0.054 | 58.9 |
| 0.02 | 72 | 6/72 = 0.083 | — | 0.051 | 59.1 |
| **0.04** | **66** | **5/66 = 0.076** | [0.033, 0.165] | **0.038** | **59.2** |
| 0.06 | 56 | 4/56 = 0.071 | — | 0.021 | 56.4 |
| 0.08 | 45 | 3/45 = 0.067 | — | 0.009 | 50.0 |
| 0.10 *(in force)* | 40 | 3/40 = 0.075 | [0.026, 0.199] | 0.003 | 42.0 |
| 0.12 | 35 | 2/35 = 0.057 | — | 0.001 | 34.5 |
| 0.15 | 26 | 1/26 = 0.038 | [0.007, 0.189] | 0.000 | 25.9 |
| 0.20 | 18 | 1/18 = 0.056 | — | 0.000 | 17.4 |

Even unfiltered, the emitted pair is oriented correctly ~95% of the time in simulation,
against a 24.4% tail flip rate over all pairs. The CIs overlap heavily — the empirical arm
has 75 groups and cannot resolve these differences on its own — which is exactly why the
decision rests on the effective-signal column, where the two arms agree: **the optimum is
0.04, on a plateau flat from 0.00.**

Effective signal formalizes the tradeoff. A pairwise-preference gradient on labels wrong
with probability $p$ is the clean gradient scaled by $(1-2p)$, so efficiency scales with
$n(1-2p)^2$. Past 0.04 the accuracy gain no longer pays for the lost rows.

## What raising `samples_per_candidate` buys

Averaging $K$ judge calls cuts the per-candidate SD by $\sqrt K$: 0.0265 → **0.0153** at
$K = 3$. Same simulation, both arms:

| `min_margin` | $K{=}1$ groups | error | signal | $K{=}3$ groups | error | signal |
|---|---:|---:|---:|---:|---:|---:|
| 0.00 | 74.0 | 0.054 | 58.9 | 74.0 | 0.021 | **68.1** |
| 0.02 | 73.3 | 0.051 | 59.1 | 72.6 | 0.015 | **68.2** |
| 0.04 | 69.3 | 0.038 | 59.2 | 65.9 | **0.005** | 64.7 |
| 0.06 | 61.5 | 0.021 | 56.4 | 56.1 | 0.001 | 56.0 |
| 0.10 | 42.5 | 0.003 | 42.0 | 37.4 | 0.000 | 37.4 |

$K = 3$ raises effective signal ~15% and cuts orientation error at 0.04 from 3.8% to 0.5%.
Cost is 3× judge calls: roughly 9.4k → 28k calls, **$1.9 → ~$6** for a full 524-question
run. Averaging also de-noises the `chosen_score` / `margin` / sub-score fields written to
every row, which downstream analysis reads.

## The rewriter temperature ladder does nothing

`temperatures: [0.2, 0.5, 0.8, 1.1]` was configured on the reasoning that 1.1 "widens the
quality spread the judge has to rank." It does not.

| Rewriter temperature | Mean judge score | Pooled SD | Group wins | Group losses |
|---|---:|---:|---:|---:|
| 0.2 | 0.828 | 0.0266 | 18 | 16 |
| 0.5 | 0.824 | 0.0261 | 22 | 19 |
| 0.8 | 0.811 | 0.0289 | 18 | 23 |
| 1.1 | 0.825 | 0.0242 | 17 | 17 |

Flat means, and each rung wins about a quarter of its groups — indistinguishable from
random assignment. What actually produces the spread is the candidate **count**, as a plain
order statistic:

| Candidates per group | Mean best-worst margin | Groups with margin ≥ 0.04 |
|---|---:|---:|
| 4 (`0.2, 0.5, 0.8, 1.1`) | 0.1288 | 64/75 |
| 3 (`0.2, 0.5, 0.8`) | 0.1063 | 59/75 |
| 3 (`0.2, 0.8, 1.1`) | 0.1087 | 57/75 |
| 2 (`0.2, 0.5`) | 0.0619 | 39/75 |
| 2 (`0.2, 1.1`) | 0.0633 | 40/75 |

Four draws are worth keeping; the specific temperatures are arbitrary. Four samples at one
temperature would work as well, and dropping to two would halve the available margin.

## `scholar` is the hard persona, twice over

| Persona | Mean judge score | Pooled SD | $n$ |
|---|---:|---:|---:|
| `crammer` | 0.861 | 0.0189 | 100 |
| `scholar` | **0.742** | **0.0345** | 100 |
| `steady` | 0.863 | 0.0237 | 100 |

`scholar` scores lowest **and** noisiest — 1.8× the SD of `crammer`. This is the q97
mechanism at scale: many source questions ask for a plain-language treatment
("به زبان ساده"), so a scholar-appropriate rewrite is penalized on
`meaning_preservation` (0.40) even as it earns `persona_fit` (0.35). The two criteria
conflict by construction and meaning wins, leaving the judge both harsher and less certain
on this persona.

Note this is the third place `scholar`/`crammer` separate: the tournament's per-persona
split (grok 0.821 `crammer` against 0.510 `steady`) and the Gemini agreement runs also
singled personas out, though
[judge-agreement-luna-gemini-v1](judge-agreement-luna-gemini-v1.md) establishes that those
per-persona cells are too small to read. Whether all three are the same effect is untested.

## Config changes made

Applied to [`configs/datagen_dpo.yaml`](../../configs/datagen_dpo.yaml) after this run, so
the assets generated from it are the first to carry them:

| Parameter | Was | Now | Basis |
|---|---|---|---|
| `filters.min_margin` | 0.10 | **0.04** | Effective-signal peak; 0.10 cost 43% of groups for no measured accuracy gain |
| `judge.samples_per_candidate` | 1 | **3** | Single-draw SD is 0.0265, not 0; averaging → 0.0153 |
| `filters.min_judge_samples` | 1 | **2** | Still averages a candidate that lost one sample to retry exhaustion |

Left unchanged, deliberately:

- `min_chosen_score: 0.55` — filters 1 of 75 groups. Raising to 0.70 would filter 7, all in
  the noisiest score band. Defensible, but a 9% volume cut on a judgement call, not on
  evidence.
- `max_pair_similarity: 0.95` — 2 of 450 pairs were identical text. Barely binding, so
  correctly set.
- `cross_persona_min_margin: 0.10` — **not calibrated by this experiment.**
  `benchmarks/judge_noise.py` forms within-persona pairs only, so nothing here speaks to it.
  It is now deliberately 2.5× stricter than `min_margin`, and it was the top rejection
  reason in trial runs, which means it throttles precisely the rows carrying the
  personalization claim. Measuring it needs a pass that re-scores each rewrite under all
  three personas.
- `temperatures` — flat by the table above, but four draws are worth keeping and the values
  are harmless. Changing them would churn the assets for no measured gain.

## What this settles and what it does not

Settled:

- Luna at `temperature: 0.0` is not deterministic: 18.0% identical replies, single-draw SD
  0.0265, max 0.137. The v2 config's "replies are near-copies" premise was wrong.
- `min_margin: 0.10` was miscalibrated, costing 43% of groups for no measured accuracy gain.
  0.04 maximizes effective signal, on a flat plateau.
- Judge noise on the two sides of a pair is independent (ρ ≈ 0), so $\sigma_\Delta =
  \sqrt2\,\sigma$ holds and pair noise is predictable from single-candidate noise.
- Noise is strongly heteroscedastic: 4× larger below 0.70 than above 0.85.
- Rewriter temperature buys no quality spread. Candidate count does.
- The flip rate over all candidate pairs is not the error rate of the emitted dataset —
  24.4% against ~5–9% — because production selects the extreme pair.

Not settled:

- **Whether the judge is right.** Every number here is self-consistency. A judge that
  reliably prefers verbose rewrites would show σ → 0 and be reliably wrong. Correctness is
  the [Gemini-agreement](judge-agreement-luna-gemini-v1.md) question, where the ~30% rubric
  ambiguity is a far larger ceiling than replicate noise, and `min_margin` does nothing
  about it.
- Whether prompt-perturbation noise exceeds replicate noise. Reordering candidates or
  paraphrasing the rendered persona would test this; only re-run noise was measured.
- The `temperature: 1.0` arm, which would have quantified how much worse the v1 label era
  was. This run used `--judge-temps 0.0` only, so the v1-vs-v2 comparison stays qualitative.
- `cross_persona_min_margin`, untouched by this design (within-persona pairs only).
- Whether the `scholar` penalty is a rubric-weight artifact or a real quality gap. Would need
  the rubric re-run at different weights on the same frozen candidates — cheap, since the
  candidates are persisted.
- Whether 0.04 holds on `val`. All 25 questions came from `train`.

Sample sizes deserve care: 75 groups, and the per-band and per-persona cells run to $n = 2$.
Per the variance lesson in [judge-agreement-luna-gemini-v1](judge-agreement-luna-gemini-v1.md),
cells below $n \approx 30$ should not be read as estimates.

## Reproduction

The measurement:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/judge_noise.py \
    --config configs/datagen_dpo.yaml \
    --replicates 3 --judge-temps 0.0 \
    --target-flip-rate 0.10 \
    --report-out data/dpo/judge_noise/report_flip10.json
```

`--dry-run` prints the call budget first. Re-analysis needs no API calls:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/judge_noise.py \
    --config configs/datagen_dpo.yaml \
    --from-samples data/dpo/judge_noise/samples.jsonl
```

Note that `--config` supplies the rubric and filters at analysis time, so re-running the
command above now reads `min_margin: 0.04` and prints `<- current` on a different row than
the table in this document. The persisted samples are unaffected.

Question sampling is seeded (`--seed 42`) and drawn from `train` only. The rewriter and
judge are not seeded, so a fresh run redraws candidates; only the `--from-samples`
re-analysis is bit-reproducible.

### Derivation of the orientation and $K{=}3$ tables

Those two tables are not printed by the script — it reports the flip curve and the survival
curve, and the orientation analysis was derived from the persisted samples afterwards. This
is the recipe, runnable against `samples.jsonl` alone:

1. Group rows by `candidate_index`; per candidate take the mean `score` across replicates as
   its reference value and the sample SD as its noise. Pool the SDs in quadrature for σ
   (0.0265).
2. Group candidates by `(question_ref, persona_id)` — 75 groups of 4.
3. **Empirical arm.** Per group, select with `replicate == 1` scores: chosen = argmax,
   rejected = argmin, skip the group if the chosen score is below `min_chosen_score`. Record
   the observed margin. The reference ordering is the mean of replicates 2 and 3; the
   emitted pair is wrong if the reference does not rank chosen above rejected, with ties
   counted wrong. Tabulate over the threshold grid.
4. **Simulated arm.** Per trial, per group, draw each candidate's observed score as its
   reference value plus $\mathcal{N}(0, \sigma^2/K)$, select the same way, and compare
   against the reference values. 4,000 trials at seed 42 for the tables above.
5. Effective signal is $n(1-2p)^2$ with $n$ the surviving group count.

The empirical arm's group counts (74/72/66/56/45/40) run slightly above the script's
survival curve (74/71/62/52/42/34) because the survival curve selects on replicate means
while this arm selects on a single replicate, whose noise inflates observed margins. Both
are correct answers to slightly different questions; the survival curve is the one that
describes a `samples_per_candidate: 3` run.
