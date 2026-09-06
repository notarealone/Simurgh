# Held-out ablation v1

## Decision

**The ladder is a gain up to Stage 1 and a loss after it.** Against naive RAG the assembled
system wins on four of five rubric fields, but the best arm is not the full system: it is
**rung 3, the ROPG embedder with persona prompting and no rewriter**. Both rewriting rungs sit
below it, and the WPO adapter selected in
[dpo-arms-seed42-v2](dpo-arms-seed42-v2.md) does not recover the loss — it merely ties the
untrained rewriter.

Three claims survive both judges and Holm correction:

1. **Persona prompting buys persona fit and pedagogy, and pays for it in faithfulness.**
   `persona_alignment` $+0.25$ / $+0.38$ and `pedagogical_quality` $+0.36$ / $+0.41$
   (primary / secondary), with `faithfulness` $-0.17$ on the primary judge.
2. **The ROPG embedder is the load-bearing component.** `context_utility` $+0.40$ / $+0.45$,
   and it is the only rung that improves `answer_correctness` ($+0.077$ / $+0.070$).
3. **Query rewriting is a regression, not an ablation-neutral addition.** The base rewriter
   costs $-0.15$ / $-0.16$ `context_utility`; the WPO rewriter changes nothing on top of it
   (every metric $p_{\text{holm}} = 1.0$).

The mechanism behind claim 3 is measurable in the rewrites themselves, not just in the
scores: the WPO adapter keeps only **61%** of each question's content tokens against the base
rewriter's **80%**, and injects persona vocabulary — the word for "grade nine" appears in
**22.6%** of its rewrites versus **0.3%** for the base rewriter. It learned to write
persona-flavoured queries at the cost of the question, and retrieval follows the query.

The honest thesis reading: Stage 1 (ROPG) works, Stage 2 (DPO/WPO rewriting) does not
transfer from its offline preference win to end-to-end quality.

## Experiment inputs

One run of `benchmarks/eval_runner.py` over `configs/eval_ablation.yaml`, config digest
`e504af60e27b5868753826b5fdf5a6116162c2321c95e3777424d8844f397882`, executed on 2 x T4 under
`torchrun` from `notebooks/eval_ablation.ipynb`.

| Arm | Rung | Embedder | Profile in prompt | Rewriter |
|---|---|---|---|---|
| `base_embedder_no_profile` | 1 — naive RAG | base `Qwen3-Embedding-0.6B` | no | none |
| `base_embedder` | 2 — persona prompting | base | yes | none |
| `trained_embedder` | 3 — Stage 1 | ROPG run B | yes | none |
| `trained_embedder_base_rewriter` | 4 — rewriting control | ROPG run B | yes | untrained `Qwen3-4B` |
| `trained_embedder_rewriter` | 5 — full system | ROPG run B | yes | WPO adapter |

Each rung adds exactly one component to the rung below, which is what makes the rung deltas
attributable. Adapters: ROPG `models/ropg-runB/checkpoint-best` (selected in
[ropg-runs-comparison-v1](ropg-runs-comparison-v1.md)); rewriter `models/rewriter-wpo/dpo_best`
(selected in [dpo-arms-seed42-v2](dpo-arms-seed42-v2.md)).

Generator `deepseek-v4-flash` at `temperature: 0.2`. Primary judge `gpt-5.6-luna`
(`reasoning_effort: low`), secondary judge `gemini-3.5-flash-lite`
(`thinking_level: minimal`), rubric in `benchmarks/judge.py`: `context_utility`,
`faithfulness`, `persona_alignment`, `pedagogical_quality` on 0–4 and `answer_correctness` on
0/1.

**Scale and integrity.** 94 held-out test questions x 4 personas x 5 arms x 1 replicate =
**1880 keys, all 1880 successful**, one config digest, 0 stale records, 5 retrieved chunks on
every row. 11 rows lack a secondary score (0.6%), so the secondary judge's paired $n$ runs
371–376 against the primary's 376. Wall clock 4h15m across two resumes; the first two attempts
died on endpoint credit exhaustion at 872 and 1834 rows and were resumed from their rank
shards, which is why the digest check in the notebook's seeding cell exists.

**Pairing.** Every arm scored the same 376 `(question, persona)` keys, so all comparisons here
are paired. `benchmarks/plot_eval_ablation.py` asserts that key-set equality, the single
digest, and the all-`ok` status before plotting anything.

## Results

Higher is better on every field.

![Ablation ladder](figures/eval-ablation-v1/ladder.png)

| Judge | Metric | naive RAG | +persona | +ROPG | +base rw | +WPO rw | Best |
|---|---|---:|---:|---:|---:|---:|---|
| primary | context_utility | 2.894 | 2.926 | **3.330** | 3.184 | 3.207 | ROPG |
| primary | faithfulness | **3.758** | 3.588 | 3.625 | 3.614 | 3.606 | naive |
| primary | persona_alignment | 3.263 | 3.513 | **3.684** | 3.625 | 3.628 | ROPG |
| primary | pedagogical_quality | 2.404 | 2.761 | **3.051** | 2.886 | 2.957 | ROPG |
| primary | answer_correctness | 0.654 | 0.678 | **0.755** | 0.707 | 0.750 | ROPG |
| secondary | context_utility | 2.845 | 2.771 | **3.227** | 3.064 | 3.022 | ROPG |
| secondary | faithfulness | 3.694 | **3.757** | 3.725 | 3.726 | 3.628 | +persona |
| secondary | persona_alignment | 3.147 | 3.523 | **3.618** | 3.548 | 3.585 | ROPG |
| secondary | pedagogical_quality | 2.858 | 3.267 | **3.444** | 3.338 | 3.356 | ROPG |
| secondary | answer_correctness | 0.670 | 0.699 | **0.770** | 0.731 | 0.757 | ROPG |

Rung 3 takes eight of the ten cells. The two it loses are both `faithfulness`, where the
ranking is inverted — the arm that personalizes least is the arm that stays closest to its
context.

### Significance, rung by rung

![Rung deltas](figures/eval-ablation-v1/rung_deltas.png)

Stars mark Holm-adjusted $p < 0.05$ **within one judge's five-metric family for one rung** — 5
tests per panel, 10,000 bootstrap resamples at seed 42. This is a deliberately different family
from `paired_deltas.csv`, which the runner corrects across all 200 of its rows at once (2
judges x 4 rungs x 5 metrics x 5 persona slices, heavily dependent tests). Both are reported:
the per-panel family below, the runner's 200-row family in the table at the end of this
section. Where they disagree, the runner's is the conservative bound.

**Rung 2, persona prompting** ($n = 376$ / $372$). The one rung whose effect is a genuine
trade. `pedagogical_quality` $+0.356$ $[+0.253, +0.463]$ and `persona_alignment` $+0.250$
$[+0.168, +0.332]$, both $p_{\text{holm}} = 0.0005$ on both judges. `faithfulness` moves
$-0.170$ $[-0.255, -0.088]$, $p_{\text{holm}} = 0.0005$ on the primary judge — the secondary
judge puts it at $+0.062$ and cannot resolve it. `context_utility` and `answer_correctness` do
not move on either judge, which is expected: this rung changes only the prompt, and the
retrieved chunks it changes it against are nearly the same ones.

**Rung 3, the ROPG embedder** ($n = 376$ / $373$). The largest and most consistent effect in
the run. `context_utility` $+0.404$ $[+0.290, +0.521]$ primary and $+0.450$ $[+0.332, +0.568]$
secondary; `pedagogical_quality` $+0.290$ / $+0.172$; `answer_correctness` $+0.077$
$[+0.043, +0.114]$ / $+0.070$ $[+0.030, +0.110]$, all $p_{\text{holm}} \le 0.0054$.
`faithfulness` does not move on either judge — better context did not cost grounding.
`persona_alignment` $+0.170$ is significant on the primary judge only.

**Rung 4, the untrained rewriter** ($n = 376$ / $374$). Every point estimate is negative on
both judges. `context_utility` $-0.146$ $[-0.239, -0.059]$ and `pedagogical_quality` $-0.165$
$[-0.277, -0.053]$ clear the per-panel family on the primary judge; `context_utility`
$-0.158$ $[-0.275, -0.043]$ clears it on the secondary. This is the rung that breaks the
ladder, and it does so *before* any trained policy is involved — inserting a query rewriter at
all is what costs, not the training.

**Rung 5, the WPO adapter** ($n = 376$ / $371$). Nothing. Every metric on both judges has
$p_{\text{holm}} = 1.0$ except primary `answer_correctness`
($+0.043$ $[+0.003, +0.082]$, $p_{\text{holm}} = 0.29$), and the two judges disagree on the
sign of three of five metrics. The WPO adapter's offline preference win over the untrained
policy does not appear here.

Under the runner's 200-row Holm family the picture is the same story told more conservatively —
rungs 2 and 3 keep their headline effects, and rung 4's losses drop below the bar:

| Rung | Primary metrics with $p_{\text{holm}} < 0.05$ (200-row family) | Secondary |
|---|---|---|
| 2 — persona prompt | faithfulness, persona_alignment, pedagogical_quality | persona_alignment, pedagogical_quality |
| 3 — ROPG embedder | context_utility, persona_alignment, pedagogical_quality, answer_correctness | context_utility |
| 4 — base rewriter | none | none |
| 5 — WPO rewriter | none | none |

### The means are not uniform effects

![Per-row wins, ties and losses](figures/eval-ablation-v1/per_question.png)

Most rows tie. Rung 3's `context_utility` gain is 72 wins against 18 losses out of 376, with
286 rows unchanged; its `pedagogical_quality` gain is 96 against 49. Rung 4's
`pedagogical_quality` loss is 54 wins against 76 losses. The rubric is coarse — five integer
levels — so a rung that shifts the mean by 0.15 is moving roughly a tenth of the rows by one
level, not nudging every row. Reporting means alone would present these as broad effects when
they are minority effects with a large indifferent majority.

## Headline comparisons

![Headline comparisons](figures/eval-ablation-v1/headline.png)

**Full system vs naive RAG.** The thesis-level claim, and it holds on both judges:
`pedagogical_quality` $+0.553$ / $+0.505$, `persona_alignment` $+0.364$ / $+0.443$,
`context_utility` $+0.314$ / $+0.179$, `answer_correctness` $+0.096$ / $+0.090$, all
$p_{\text{holm}} \le 0.019$. `faithfulness` is $-0.152$ $[-0.237, -0.067]$ on the primary judge
($p_{\text{holm}} = 0.0009$) and an unresolved $-0.065$ on the secondary.

**Full system vs rung 3.** The claim that does not hold. `context_utility` is $-0.122$
$[-0.223, -0.027]$ primary and $-0.192$ $[-0.298, -0.089]$ secondary
($p_{\text{holm}} = 0.002$); the other four metrics are indistinguishable from zero on both
judges. The full system is, at best, rung 3 with a significant retrieval-quality tax.

## Why rewriting loses

![What the rewriters did to the query](figures/eval-ablation-v1/rewriter_mechanism.png)

The scores say rewriting costs context utility. The rewritten queries say why.

Measuring each rewrite against its source question by Persian content-token retention (tokens
of ≥3 Persian characters, matched as sets):

| Rewriter | Mean token retention | Rows below 0.3 retention | "grade nine" in rewrite | "exam" in rewrite |
|---|---:|---:|---:|---:|
| untrained `Qwen3-4B` | 0.803 | 5.1% | 0.3% | 1.1% |
| WPO adapter | 0.611 | 11.7% | 22.6% | 21.0% |

The WPO adapter systematically discards question content and replaces it with persona
vocabulary. Neither word belongs to any test question; both belong to the persona profiles. In
one representative row the source question is a three-pair matching exercise with the six terms
listed inline, and the WPO rewrite drops all six, keeping only a request for a simple
step-by-step explanation "for the ninth-grade exam" — a query with no discriminative content
for a 171-chunk corpus. It also hallucinated a *fourth-grade elementary* learner in a handful
of rows, a grade level no persona uses.

The cost tracks the drift. Rewrites below 0.3 retention score `context_utility` 2.75 (WPO,
$n = 44$) and 2.68 (base, $n = 19$); rewrites at or above it score 3.27 and 3.21, against 3.33
for no rewriter at all. Retention correlates with `context_utility` at $r = +0.16$ (WPO), weak
per-row but consistent in the aggregate.

This is the failure mode `CLAUDE.md` warns about — a rewriter that games its objective. The
preference pairs rewarded persona-conditioned rewrites, so the adapter learned to inject
persona tokens; nothing in the DPO objective required it to preserve the question, and the
offline judge that built the pairs never had to retrieve against the result.

## Retrieval diagnostics

![Retrieval diagnostics](figures/eval-ablation-v1/retrieval_shift.png)

Measured on returned chunk ids alone, independent of any judge.

**How far each rung moved the top 5.** Persona prompting shares 0.77 of its top 5 with naive
RAG (the profile enters as the `Instruct:` prefix), the ROPG embedder shares 0.61 with the base
encoder, the base rewriter 0.77 with no rewriter, and the WPO rewriter 0.68 with the base
rewriter. Rewriting moves retrieval about as much as swapping the encoder did — but downward.

**Rows with no usable context.** The primary judge scored `context_utility` = 0 — retrieval
returned nothing the answer could use — on 18.4% of naive-RAG rows, 17.6% with persona
prompting, and **8.2%** with the ROPG embedder. Rewriting pushes it back up to 10.6% and 11.4%.
Halving the no-context rate is the clearest single statement of what Stage 1 bought, and
rewriting gives a third of it back.

**Persona divergence.** Without the profile, retrieval is identical across all four personas by
construction (1.00 distinct top-5 sets per question). Persona prompting produces 2.86 distinct
sets, the ROPG embedder **2.14**, the rewriting arms 3.06 and 3.26. So the trained encoder is
*less* persona-divergent than the base encoder with the same prefix, which agrees with the
Stage 1 finding that ROPG training flattened a pre-existing prefix artifact rather than adding
persona sensitivity ([ropg-runs-comparison-v1](ropg-runs-comparison-v1.md), persona-swap
control). Divergence and quality are anti-correlated across the five arms here: the most
persona-divergent retrieval is the least useful.

**No retrieval ground truth on this split.** The test questions carry no relevance labels, so
Recall@K and MRR are not computable here and `context_utility` is the judge's proxy for
retrieval quality. The labelled retrieval evaluation lives in the Stage 1 report on the
validation split.

## Per persona

![Per-persona scores](figures/eval-ablation-v1/per_persona.png)

`crammer`, `scholar` and `steady` appear in the DPO/ROPG training data; `newcomer` is held out.

Persona prompting's `persona_alignment` gain is concentrated where the baseline was weakest:
`crammer` $2.957 \to 3.479$ ($+0.52$) against `scholar` $3.606 \to 3.681$ ($+0.075$). The
naive-RAG answer was already close to what a `scholar` wants — terse, no hand-holding — so most
of rung 2's headline gain is the system learning to stop writing scholar answers for everyone.

The held-out persona is the one place the rewriter looks good: on `newcomer`,
`answer_correctness` goes $0.702 \to 0.777$ and `persona_alignment` $3.468 \to 3.521$ from rung
3 to rung 5, where all three training personas lose. **This does not survive testing.** Full
system minus rung 3, split by persona provenance, 10,000 resamples:

| Group | Metric | Delta | 95% CI | $p$ | $p_{\text{holm}}$ | $n$ |
|---|---|---:|---:|---:|---:|---:|
| newcomer | answer_correctness | +0.075 | $[-0.011, +0.160]$ | 0.125 | 1.00 | 94 |
| newcomer | persona_alignment | +0.053 | $[-0.106, +0.223]$ | 0.616 | 1.00 | 94 |
| newcomer | context_utility | −0.192 | $[-0.426, +0.021]$ | 0.108 | 1.00 | 94 |
| 3 training personas | persona_alignment | −0.092 | $[-0.177, -0.007]$ | 0.040 | 0.64 | 282 |
| 3 training personas | pedagogical_quality | −0.149 | $[-0.277, -0.021]$ | 0.029 | 0.55 | 282 |
| 3 training personas | context_utility (secondary) | −0.212 | $[-0.335, -0.097]$ | 0.0004 | 0.008 | 278 |

At $n = 94$ the newcomer intervals are roughly twice as wide as the pooled ones and every one
of them contains zero. The suggestive reading — that the rewriter generalizes to unseen
personas while overfitting the trained ones — is not supported; it is a direction, at one
quarter of the sample size, in a slice chosen after seeing the means.

## Judge cross-validation

![Judge cross-validation](figures/eval-ablation-v1/judge_agreement.png)

From `judge_agreement.csv`, pooled over all arms ($n = 1869$ rows scored by both):

| Metric | Exact agreement | Mean abs. diff | Pearson $r$ | Secondary − primary |
|---|---:|---:|---:|---:|
| answer_correctness | 0.961 | 0.039 | 0.904 | +0.02 |
| context_utility | 0.816 | 0.276 | 0.889 | −0.12 |
| faithfulness | 0.723 | 0.408 | 0.426 | +0.07 |
| persona_alignment | 0.716 | 0.355 | 0.648 | −0.06 |
| pedagogical_quality | 0.604 | 0.538 | 0.816 | +0.44 |

Two different things are visible, and conflating them would misread the run. **Absolute levels
disagree**: the secondary judge is 0.44 more generous on `pedagogical_quality` and 0.12 stricter
on `context_utility`, so no single-arm score here is a calibrated absolute quantity. **Deltas
agree**: across the 20 rung x metric pairs the two judges' estimates correlate at $r = 0.88$
and agree in sign on 75% of them, and both rank rung 3 first on every field except
`faithfulness`. Every claim in this report is a delta for that reason.

`faithfulness` is the weak field: $r = 0.43$, the lowest of the five, and the field where the
judges split on rung 2's direction. The faithfulness cost of persona prompting is therefore a
primary-judge finding, not a cross-validated one, and should be stated that way. It is also
consistent with [judge-agreement-luna-gemini-v1](judge-agreement-luna-gemini-v1.md), which
found no detectable difference between the flash and flash-lite secondary judges but did not
establish that either tracks faithfulness reliably.

## Threats to validity

- **One replicate, no seeds.** `replicates: [1]` means every number here has endpoint sampling
  variance folded into it and *no* variance component from training seeds, index rebuilds, or
  data-generation seeds. [experiment-design](../experiment-design.md) asks for ≥3 seeds; this
  run does not meet that. The paired intervals measure uncertainty across the 376 test rows only.
- **Held-out questions, overlapping sources.** The split is question-level (seed 42, 70/15/15):
  `question_ref` is disjoint across splits, but source files — and therefore corpus passages —
  are shared. A rung that benefits from passage familiarity is not fully isolated here.
- **The test set is mostly synthetic.** 76 of 94 test questions come from
  `ai_generated_questions`; 18 are real exam items. Persona profiles are synthetic throughout.
- **Judge-generator independence, partially.** The generator (`deepseek-v4-flash`) is
  independent of both judges, and the judge that scored these answers is not the prompt that
  built the DPO pairs, as the repo rules require. But the primary judge here (`gpt-5.6-luna`) is
  the same model family that labelled the ROPG and DPO training data, so rung 3's gain is
  measured partly by the teacher that produced its supervision.
- **Multiplicity is a judgement call.** The runner's 200-row Holm family is conservative to the
  point of hiding rung 4's regression; the per-panel 5-metric family is the one used for the
  starred figures. Both are published; the decision above holds under either.
- **`answer_correctness` is binary and single-rater per judge.** At 0.65–0.77 across arms it is
  the coarsest field, and small deltas on it should not be over-read.

## Figures

All eight figures live under `figures/eval-ablation-v1/` and regenerate from the run's
`results.jsonl` with:

```bash
python benchmarks/plot_eval_ablation.py \
    --run results/ablation/complete_eval_results/ablation \
    --out docs/results/figures/eval-ablation-v1
```

The script reuses `compare_runs.py`'s `paired_stats` and `holm` and the rubric fields from
`benchmarks/judge.py`, refuses to plot unless all rows share one config digest and every arm
covers the same key set, and writes every plotted number to `figures/eval-ablation-v1/stats.json`.
Bootstrap CIs and permutation $p$-values use 10,000 resamples at seed 42; the runner's own CSVs
use 5,000, so third-decimal differences between the two are expected.

| Figure | What it answers |
|---|---|
| `ladder.png` | Where every arm lands on every rubric field, per judge |
| `headline.png` | Full system against naive RAG, and against the best rung |
| `rung_deltas.png` | Which single-component effects survive paired CIs and Holm |
| `per_question.png` | Whether the mean shifts are broad or minority effects |
| `rewriter_mechanism.png` | Why rewriting costs context utility |
| `retrieval_shift.png` | What each component did to retrieval, judge-free |
| `per_persona.png` | Who the gains belong to, and how the held-out persona behaves |
| `judge_agreement.png` | Whether the second judge reproduces the deltas |

## Artifacts

The run directory holds `results.jsonl` (1880 rows, 24 MB), the two rank shards,
`summary.csv` (550 rows), `paired_deltas.csv` (200 rows), `judge_agreement.csv` (30 rows), and
`run_manifest.json` with input checksums, adapter paths, model settings, and library versions.
Those dumps are large prediction data and stay out of `docs/`; this document plus
`figures/eval-ablation-v1/` is the committed record.
