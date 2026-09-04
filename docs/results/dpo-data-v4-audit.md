# DPO Pair Data v4 — As-Built Audit

**Finding.** The fourth generation of the DPO pair files is the first one whose labels
are demonstrably better than noise. Modeled label-orientation error is **0.08%** against
format 1's implied ~27%, and the dataset is 53% larger than the set it replaces
(3,192 rows vs 2,090) rather than smaller — the usual price of a stricter filter was
avoided by *lowering* `min_margin` on the evidence in
[judge-noise-luna-v1](judge-noise-luna-v1.md) instead of raising it.

Two results are new and load-bearing for the thesis. A rewrite written for persona Y
scores **+0.180 lower** when judged for persona X, in **97.1% of 1,564 cross-persona
rows** — the judge discriminates personas rather than tracking generic quality, which
until now was an assumption. Against that, the `scholar` persona appears as the negative
1.6× more often than as the positive, which is a rubric artifact and the main risk this
audit surfaces.

This is an audit of shipped artifacts, not an experiment. It reports what the generator
produced; whether training on it works is [dpo-arms-seed42-v1](dpo-arms-seed42-v1.md)'s
successor's question.

## Inputs

| | |
|---|---|
| Pair files | `data/dpo/train.jsonl` (2,658 rows), `data/dpo/val.jsonl` (534) |
| Candidate dumps | `data/dpo/candidates/{train,val}.jsonl` (6,288 rows total) |
| Filtered dumps | `data/dpo/candidates_filtered/{train,val}.jsonl` (6,226) |
| Generated | 2026-09-04 (train 03:44, val 04:48) |
| Config | `configs/datagen_dpo.yaml` |
| Code state | `d6df96d` plus the uncommitted format-2 generator |
| Rewriter | `grok-4-1-fast`, temperatures 0.2 / 0.5 / 0.8 / 1.1, 4 candidates per group |
| Judge | `gpt-5.6-luna`, `temperature: 0.0`, `reasoning_effort: none`, strict `json_schema`, **3 samples per candidate** |
| Rubric | `meaning_preservation` 0.40, `persona_fit` 0.35, `specificity` 0.25 |
| Filters | `min_judge_samples: 2`, `min_chosen_score: 0.55`, `min_margin: 0.04`, `max_pair_similarity: 0.95`, `cross_persona_min_margin: 0.10` |
| Scoring calls | 18,864 candidate scorings + cross-persona re-scorings |

### Version lineage

Four generations of pair data exist under two schema formats. "v4" counts data
generations; `format_version` counts schemas, and this run is format 2.

| gen | date | train / val | train qrefs | schema | what changed |
|---|---|---:|---:|---|---|
| v1 | 2026-07-05 | 319 / 94 | 67 | unstamped | first pass, partial question coverage |
| v2 | 2026-07-10 | 1,826 / 368 | 431 | unstamped | full question coverage |
| v3 | 2026-08-28 | 1,739 / 351 | 432 | `format_version: 1` | rendered questions via `src/data/questions.py`, gold answer as judge context, `question_ref` added |
| **v4** | **2026-09-04** | **2,658 / 534** | **432** | **`format_version: 2`** | **rubric judge, 3 judge samples, persisted provenance, labelled cross-persona pairs, calibrated `min_margin`** |

v1 and v2 predate the version stamp and are format-1 precursors; `rl.dpo_train.load_pairs`
refuses all three. The seed-42 screening in
[dpo-arms-seed42-v1](dpo-arms-seed42-v1.md) ran on **v3**, so the 272-key constant and
every number in that report belong to that generation and are not comparable to v4.
Prior generations are retained under `data/dpo/old_v{1,2,3}/`.

## Yield and pair composition

**1,256 of 1,296 train groups (96.9%) and 268 of 276 val groups (97.1%) produced at least
one pair.** A group is one `(question_ref, persona_id)` key.

| split | rows | within-persona | cross-persona | rows/group |
|---|---:|---:|---:|---:|
| train | 2,658 | 1,094 (41.2%) | 1,564 (58.8%) | 2.12 |
| val | 534 | 220 (41.2%) | 314 (58.8%) | 1.99 |

Cross-persona rows outnumber within-persona ones because each group's winner can be
paired against up to two foreign rewrites but only one within-persona negative. Rows per
group: 250 groups yield 1 row, 610 yield 2, 396 yield 3.

The 40 empty train groups are **33 scholar, 7 steady** — no crammer group failed. That
distribution is the first symptom of the persona skew below; scholar groups fail because
their best candidate misses `min_chosen_score: 0.55`, not because no margin exists.

Only 46 of 5,184 train candidates were filtered, all `duplicate_rewrite` (0.9%). Every
candidate in both splits carries `judge_samples: 3` — no judge sample was ever lost, so
`min_judge_samples: 2` never bound and remains pure insurance.

## The `min_margin: 0.04` retune paid off

**36.6% of within-persona train rows (400 of 1,094) have a margin below 0.10** and exist
only because of the retune. What the old threshold would have cost:

| `min_margin` | within rows | retained | modeled flip rate |
|---:|---:|---:|---:|
| **0.04 (shipped)** | **1,094** | **100%** | 0.0018 |
| 0.06 | 957 | 87.5% | 0.0001 |
| 0.08 | 827 | 75.6% | 0.0000 |
| 0.10 (old) | 694 | 63.4% | 0.0000 |
| 0.15 | 446 | 40.8% | 0.0000 |

The 63.4% retention at 0.10 lands inside the noise study's predicted survival band
(34–62 of 75 groups). Discarding a third of the within-persona data to move modeled error
from 0.18% to 0.00% would have been a bad trade, which is what the effective-signal
argument $n(1-2p)^2$ predicted.

No emitted row has a margin below 0.04, in either split or either pair type — the filter
is doing exactly what it says.

## Label noise

With three judge samples the single-draw σ of 0.0265 becomes σ = 0.0153, so a pair
difference carries σ = 0.0217. Applying that to the shipped margins:

| split / type | n | mean margin | p10 | median | modeled flip rate |
|---|---:|---:|---:|---:|---:|
| train within | 1,094 | 0.158 | 0.055 | 0.128 | 0.0018 |
| train cross | 1,564 | 0.249 | 0.122 | 0.217 | 0.0000 |
| val within | 220 | 0.155 | 0.050 | 0.125 | 0.0025 |
| val cross | 314 | 0.233 | 0.118 | 0.211 | 0.0000 |

**Expected mislabeled rows: ~2 of 2,658 (0.08%).** No row anywhere has a flip probability
above 5%.

That figure conditions on the observed margin, which is inflated by selection — the
chosen candidate is the max of four noisy draws. Subtracting a winner's-curse correction:

| correction | mean flip | rows with p > 0.05 | expected bad rows |
|---:|---:|---:|---:|
| none | 0.0008 | 0 | 2 |
| −0.010 | 0.0022 | 46 | 6 |
| −0.020 | 0.0056 | 112 | 15 |
| −0.030 | 0.0120 | 172 | 32 |

Even the aggressive correction gives 1.2%. The honest upper bound remains the noise
study's empirical hold-out estimate of ~7.6%, which used single-replicate selection and
so overstates a $K{=}3$ run. The true value sits between 0.1% and 7.6%; all of that
range is far below format 1's ~27%.

**This measures orientation against the judge, not correctness.** Judge–Gemini agreement
of ~70% in [judge-agreement-luna-gemini-v1](judge-agreement-luna-gemini-v1.md) remains
the real ceiling, and no margin filter can raise it.

## The personalization signal is real

Every cross-persona negative was matched back to its own entry in the candidate dump
(**1,564 of 1,564 matched, zero unmatched** — the `scored_as` provenance plumbing works
end to end) to compare the same string scored under two personas:

> A rewrite written for persona Y scores **+0.180 higher** under Y than under the target
> persona X, and does so in **97.1%** of rows.

This is the measurement the cross-persona construction was built to produce. It rules out
the deflationary reading in which the judge simply ranks fluency and the persona label is
decoration.

The sub-score decomposition shows the two pair types teach different lessons:

| criterion | within-persona Δ | share > 0 | cross-persona Δ | share > 0 |
|---|---:|---:|---:|---:|
| `meaning_preservation` | +0.191 | 0.968 | +0.132 | 0.800 |
| `persona_fit` | +0.108 | 0.902 | **+0.468** | **1.000** |
| `specificity` | +0.176 | 0.964 | +0.132 | 0.850 |

Cross-persona rows are almost pure `persona_fit` contrast, with the correct sign on all
1,564. Within-persona rows spread across `meaning_preservation` and `specificity` — the
generic-quality axis. A future ablation can therefore separate "learned to rewrite well"
from "learned to rewrite *for this learner*" by training on the two subsets separately.

## Risk: the scholar persona is systematically the loser

| persona | chosen-side | rejected-side | ratio | candidate mean | sd | frac < 0.55 |
|---|---:|---:|---:|---:|---:|---:|
| crammer | 1,032 | 824 | 1.25 | 0.861 | 0.091 | 0.012 |
| **scholar** | **699** | **1,138** | **0.61** | **0.683** | **0.160** | **0.199** |
| steady | 927 | 696 | 1.33 | 0.846 | 0.120 | 0.036 |

Scholar-targeted rewrites appear as the negative 1.6× more often than as the positive,
and val reproduces the ratio (0.64). Scholar trails on all three criteria and worst on
`meaning_preservation` (0.653 vs 0.874 crammer, 0.859 steady) — the «به زبان ساده»
conflict predicted when the rubric weights were set: when a question's own wording
prescribes a simple register, a scholar-appropriate rewrite reads as drifting from the
question, and the 0.40-weighted fidelity criterion punishes it.

The direction counts make the asymmetry sharper:

| target ← source | train rows |
|---|---:|
| crammer ← scholar | 427 |
| steady ← scholar | 318 |
| scholar ← crammer | 288 |
| crammer ← steady | 291 |
| steady ← crammer | 222 |
| **scholar ← steady** | **18** |

`steady ← scholar` (318) against `scholar ← steady` (18) is a 17.7× imbalance: a scholar
rewrite is judged a poor fit for a steady learner constantly, while a steady rewrite is
almost never judged a poor fit for a scholar. That is not symmetric persona distance — it
says the judge treats "simpler than needed" as acceptable for a scholar but "more
advanced than needed" as a real failure for a steady learner. Defensible as pedagogy,
but it means the personalization gradient is much stronger in one direction.

**Consequence for training.** DPO will see scholar register as the thing to move away
from about twice as often as the thing to move toward. If the trained rewriter produces
flatter, less advanced rewrites for scholars than the prompted baseline does, this is the
cause, and it is a property of the rubric weights rather than of the data volume.
Options, in increasing order of cost: report the skew and check per-persona eval deltas
separately; re-weight `meaning_preservation` down for scholar; or add a rubric criterion
that credits appropriate elaboration. None was applied to v4.

## Cleanliness checks

All pass.

| check | train | val |
|---|---|---|
| Exact duplicate rows | 0 | 0 |
| `chosen == rejected` | 0 | 0 |
| Pairs ≥ 0.95 similar | 0 | 0 |
| Pairs ≥ 0.90 similar | 2.65% within, 0% cross | 2.73% within, 0% cross |
| Median chosen/rejected similarity | 0.687 within, 0.578 cross | 0.689 / 0.570 |
| **Same-persona contradictions** | **0** | **0** |
| `question_ref` crossing splits | 0 | — |
| Rewrite strings crossing splits | 0 | — |
| Query strings crossing splits | 0 | — |

The near-duplicate rate more than halved against format 1's 5.06%, and the cross-persona
subset contains no near-duplicates at all.

**On the 91.7% of rows whose text appears on both sides.** 1,033 train rewrite strings
appear as `chosen` in one row and `rejected` in another, touching 91.7% of rows. This is
the design, not contamination: the *same* rewrite is right for one learner and wrong for
another, which is the thesis claim stated as data. The failure mode it could hide —
the same text on both sides *for the same persona*, which would be a flat contradiction —
occurs **zero times** in either split.

## Two things confirmed at production scale

**The rewriter temperature ladder does nothing.** Over 5,184 train candidates, mean judge
score by temperature runs 0.796 / 0.798 / 0.797 / 0.795, and wins per group are
312 / 314 / 339 / 331 — indistinguishable from random assignment. The n=300 pilot finding
replicates at 17× the sample. The spread the pairs rely on comes from the candidate
*count* as an order statistic, not from the temperatures. Keep four draws; the values are
arbitrary. Any future budget cut should drop a judge replicate before dropping a
candidate.

**`min_chosen_score: 0.55` barely binds, and where it binds it is persona-specific.**
It fails 31 of 1,296 train groups (2.4%) and 3 of 276 val groups; 0.60 would fail 49 and
0.70 would fail 125 (9.6%). Since 33 of the 40 empty train groups are scholar, raising
this bar is really a decision to drop scholar coverage, and the bar sits in the score band
where the judge is least reliable (mean SD 0.041–0.043 between 0.40 and 0.70). Left at
0.55.

## Mild concern: length bias

| split / type | chosen | rejected | ratio | P(chosen longer) |
|---|---:|---:|---:|---:|
| train within | 113.1 | 105.3 | 1.074 | 0.583 |
| train cross | 117.7 | 113.6 | 1.036 | 0.529 |
| val within | 115.3 | 104.1 | 1.107 | 0.645 |
| val cross | 120.6 | 115.9 | 1.041 | 0.535 |

Chosen rewrites run about 7% longer, driven by the within-persona subset. The effect is
small, but DPO amplifies length preferences, so mean completion length per epoch should
be logged during training; a rewriter drifting longer is the first sign of the
length-hacking degeneration the DPO conventions warn about.

## What this settles

- `min_margin: 0.04` is correct for this data: it costs nothing measurable in label
  quality and retains 36.6% more within-persona rows than 0.10.
- Three judge samples put modeled label error below 1% by every correction assumption
  tried.
- The judge discriminates personas, measured rather than assumed (+0.180, 97.1%).
- The pair files are free of duplication, same-persona contradiction, and split leakage.
- The rewriter temperature ladder can be simplified without loss whenever it is convenient.

Not settled:

- **Whether the labels are right**, as opposed to self-consistent. Unchanged by this run;
  it needs the second judge.
- **The scholar skew's effect on the trained policy.** Predicted here, observable only
  after training. Per-persona eval deltas must be reported separately, not pooled.
- **Whether `cross_persona_min_margin: 0.10` is well set.** Still uncalibrated. It now
  admits every cross-persona row at margin ≥ 0.118 (p10), so it is not obviously binding,
  but no noise measurement exists for cross-persona deltas.
- **Whether the cross-persona majority helps or dominates.** At 58.8% of rows they are the
  easier discrimination; if eval accuracy is high, verify it is not carried entirely by
  that subset.
- **The 40 empty train groups.** 33 scholar groups contribute no training signal at all,
  compounding the skew above at the coverage level rather than the row level.

## Reproduction

All figures come from the four shipped JSONL files; no API calls are needed.

Group keys are `(question_ref, persona_id)`. Similarity is `difflib.SequenceMatcher`
over NFKC-normalized, whitespace-collapsed text, matching the generator's own
`_similarity`. Modeled flip rate is $\Phi(-m / \sigma_{\text{pair}})$ with
$\sigma_{\text{pair}} = \sigma_1\sqrt{2/K}$, $\sigma_1 = 0.02652$ from
[judge-noise-luna-v1](judge-noise-luna-v1.md) and $K = 3$; the winner's-curse variants
subtract a constant from every margin before applying it. The persona-discrimination
figure joins each cross-persona `rejected` back to `data/dpo/candidates/{split}.jsonl` on
`(question_ref, rejected_persona_id, normalized rewrite)` and compares that candidate's
own `score` against the row's `rejected_score`, which is the same string re-scored under
the target persona. Exposure counts treat `persona_id` as the chosen side and
`rejected_persona_id` as the rejected side; on within-persona rows the two are equal by
construction, verified over all 1,094 rows, so cross-persona direction counts must
exclude them to avoid double counting.
