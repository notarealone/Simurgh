# Experiment Design

> Datasets, baselines, metrics, and evaluation protocol.

---

## Datasets

- **Domain** — 9th-grade Persian (فارسی نهم). Corpus: official textbook + gifted-schools edition + study guide (provenance in [data-extraction](data-extraction.md)).
- **Questions** — extracted from real exam papers into structured JSON (schema + provenance in [question-extraction](question-extraction.md)); ~618 questions across 7 sources: 6 real exam papers + `ai_generated_questions.json` (LLM-generated, curriculum-grounded, added to improve data quantity and lesson coverage).
- **Profiles** — synthetic learner personas (LLM-as-simulator; no real student data). A fixed set of 4 over 4 axes (comprehension, prior knowledge, learning goal, explanation style); 3 train + 1 test-holdout. Full schema in [personas](personas.md).
- **Grounding** — link each question to its answering corpus passage(s): gold passages for Recall@K and context for generation. Tag each question **grounded vs skill** and by **personalization headroom** — pure recall/grammar items carry little persona-fit signal; comprehension items carry the most.

Data is scarce, which constrains eval diversity and DPO volume. Mitigations to document and
pursue as needed (not core scope yet):

- [ ] Scrape more exam papers (real questions + gold answers via the [question-extraction](question-extraction.md) VLM pipeline) — highest ROI
- [x] Synthesize corpus-grounded questions — done: `data/questions/ai_generated_questions.json`
      (LLM-generated, 9th-grade Persian curriculum; enters the question-level split pool
      alongside real exam files)
- [ ] Persona-multiply for eval coverage; sample multiple DPO pairs per (question, persona)

## Baselines

The core ladder — each rung is a config over shared `src/` modules:

- [x] Rung 0 — naive RAG: BM25, no persona (`configs/phase0_naive.yaml`)
- [ ] Rung 1 — Qwen3-Embedding-0.6B frozen, no persona (`configs/phase1_dense.yaml`) — retriever swap baseline
- [ ] Rung 2 — Qwen3-Embedding-0.6B frozen, persona-prompted **untrained** rewriter (`configs/phase2_prompted_rewriter.yaml`) — baseline to beat
- [ ] Rung 3 — Qwen3-Embedding-0.6B + **ROPG-KD**, untrained rewriter (`configs/phase3_ropg_kd.yaml`) — retriever contribution
- [ ] Rung 4 — Qwen3-Embedding-0.6B + ROPG-KD, **DPO rewriter** (`configs/phase4_dpo_rewriter.yaml`) — full system

Rungs 3 vs 4 isolate the rewriter's marginal contribution on top of a trained retriever.
Rungs 1 vs 3 isolate the ROPG-KD retriever's contribution with a fixed (untrained) rewriter.

Future / out of scope:

- [ ] Generator-DPO on a small model
- [ ] Online ROPG-RL (online reward loop, higher compute)

## Metrics

Personalization is the thesis claim, so **persona alignment / pedagogical quality is the
primary metric**. EM/F1 measure answer-string correctness — two equally "correct" answers
can suit very different students — so report them as **secondary** evidence alongside
retrieval metrics.

**Primary — personalization quality**
- [ ] LLM-as-judge rubric score (persona fit + pedagogical quality + faithfulness) on a held-out test set
- [ ] Human evaluation on 50–100 samples to confirm judge scores track real pedagogical quality

**Secondary — correctness & retrieval**
- [ ] Generation correctness: Exact Match, F1 (and ROUGE/BLEU where a reference answer exists)
- [ ] Retrieval quality **per persona**: Recall@K, MRR — now a *diagnostic*, because the persona-shaped query can help or hurt recall while the gold passages stay persona-independent

**Significance** — every system answers the *same* (question, persona) rows, so use **paired**
significance tests and **bootstrap confidence intervals** over **≥3 seeds**. Paired
comparison extracts far more statistical power from a small eval set than unpaired tests —
essential given the data scarcity above. (Both methods explained where results are reported.)
For Stage-1 retrieval this is implemented: `benchmarks/compare_runs.py` consumes the
per-query vectors in `training_log.json` and reports bootstrap CIs plus Holm-corrected
sign-flip p-values, overall and per persona.

> **Judge independence:** the judge that *scores* final results must differ in family from
> the judge that *labels* the DPO preference pairs, or the numbers partly measure
> "optimizing to the judge." See [things-to-consider](things-to-consider.md) (Reward Signal).

## Evaluation Protocol

- [ ] Split by **source exam** (never by row), persona-balanced; personas = 3 train/val + 1 **test-holdout**
- [ ] Separate **validation** (tune checkpoints/hyperparameters) from **test** (touched once, at the end)
- [ ] **Freeze the test set, judge prompt, and seeds from day one and version them** — if any of these change between the Rung-0 and post-DPO runs, cross-time comparisons are invalid
- [ ] Report tables + plots with confidence intervals; state significance per claim

## Ablation Studies

- [ ] Persona removed (no profile anywhere) — drop profile from rewriter and retriever KD signal
- [ ] Generator persona-**blind** vs persona-**aware** — does a capable-enough generator make the rewriter redundant?
- [ ] Rewriter **untrained** vs **DPO** (Rung 3 vs 4 — the rewriter's marginal contribution)
- [ ] Retriever **frozen Qwen3-Embedding-0.6B** vs **ROPG-KD** (Rung 2 vs 3 — the retriever's marginal contribution)
- [ ] Retriever: BM25 vs frozen Qwen3-Embedding-0.6B vs ROPG-KD (full retriever ladder)
- [ ] On-policy vs off-policy DPO pairs (iterative-DPO study) — optional
- [ ] Document findings in [results](results/)

## Stage-1 retriever runs (ROPG `hard_neg`)

Four runs isolate each stage-1 change against the untrained encoder. Pairwise attribution
uses A vs B for anchoring, B vs C for filtering, and C vs D for `doc_frozen`. All share
`mode: hard_neg`, `format: triplets`, `lr: 5.0e-5`, `epochs: 3`, `max_negatives: 8`,
`seed: 42`.

| Run | `data.train_data` | `anchor.mode` | Isolates | Status |
|---|---|---|---|---|
| A | `data/ropg_kd` | `none` | plain MNRL at the corrected LR/epoch budget — the control | running |
| B | `data/ropg_kd` | `both` | base-model anchoring | **done** — see [results](results/stage1-ropg-runB.md) |
| C | `data/ropg_kd_filtered` | `both` | label filtering | next |
| D | `data/ropg_kd_filtered` | `doc_frozen` | the asymmetric (frozen document tower) arm | not run |

Both data directories are derived from the *same* judged `{train,val}.jsonl` and differ
only in whether the label filters ran — `configs/datagen_ropg.yaml` builds the
unfiltered one, `configs/datagen_ropg_filtered.yaml` the filtered one, and neither
re-judges anything. Selecting an arm is one key (`data.train_data`, or `ARM` in the
notebook). Every run logs the build it trained on
(`Triplets: max_negatives=8 | filters=True | groups 929/1296 retained`), which is the
after-the-fact check that the intended arm actually ran — `data.train_data` is exempt
from the notebook's config-drift check, because local and Kaggle roots legitimately
differ.

**Run order/status is B completed, A is running, and C follows.** A need not finish before
preparing C, but both are required for attribution.

**The arms are paired-comparable.** `src/rl/ropg_kd.py` loads its eval groups from
`{train_data}/val.jsonl` — the *scored* file, which `derive_triplets` never rewrites —
so filtering shrinks the training triplets while leaving all 276 val groups intact.
`val_loss` is the one exception: a filtered arm computes it over fewer triplets, so it
must not be compared across arms (`compare_runs.py` flags this).

**Negative selection is held fixed.** All A-C artifacts preserve the current
rank-13-through-rank-20 negative selection, despite the historical `hard_neg` name.
Changing that selection only for C would confound the B-vs-C comparison.

### Primary metric — revised after run B (2026-08-20)

**Headline pair: nDCG@1 and Recall@5**, both against the untrained
Qwen3-Embedding-0.6B baseline (nDCG@1 0.543, Recall@5 0.3961) — *not* against the
earlier trained checkpoints, none of which cleared it. Report the epoch-0 row in every
table.

The original criterion was nDCG@5 > 0.548 alone. Run B forced a revision, and the
reason is recorded here rather than in a footnote because changing a success criterion
after seeing a result is exactly the move that needs justifying:

- **nDCG@5 is over half noise at this label quality.** Val mean teacher score by rank
  runs 0.757 / 0.569 / 0.447 / 0.372 / 0.323, so the adjacent gaps at slots 3→4 and 4→5
  are 0.075 and 0.049 — at or below the single-label noise floor established in
  [methodology](methodology.md). **53.4% of ideal DCG@5's mass sits in slots 2–5.**
  The MNRL objective trains only the rank-1-vs-tail contrast, where the gap is 0.711.
  Grading a rank-1 objective mostly by slots 2–5 measures the labels' noise, not the
  encoder.
- **nDCG@1 grades the one slot where the teacher is reliable** — rank-1 mean 0.757,
  0.188 clear of rank 2 — and shares nDCG's ceiling of 1.0, so it is directly readable.
- **Recall@5 is already what the code selects on.** `is_better` in `src/rl/ropg_kd.py`
  ranks checkpoints by overall Recall@K and always has; the previous text calling nDCG
  "primary" contradicted the implementation, and that contradiction had already chosen
  run B's epoch-2 checkpoint. This resolves it in favour of the code.

**nDCG@1..5 is still reported in full, with the caveat above.** A drop in nDCG@5
alongside a rise in nDCG@1 is the expected signature of a rank-1 objective and is not
by itself a failure — but it is also not to be waved away, which is what `judged@5`
is for.

**`judged@5` is a required diagnostic column.** It is the fraction of the returned top-5
that the teacher actually judged. Only a group's ~20 judged chunks carry gain, out of a
171-chunk corpus, so a model that surfaces *unjudged but relevant* chunks is punished by
nDCG for improving. Recall@5 up with nDCG@5 down is consistent both with "the graded
middle got worse" and with "the retrieved set moved outside the judged pool";
`judged@5` is the only thing that separates them. Never report nDCG without it.

**`persona_swap` is a required control.** Personas reach the retriever only through the
`Instruct:` prefix, so an encoder that ignores that prefix improves every headline
metric while personalising nothing. Each epoch re-scores every val query under a rotated
persona; the matched-minus-swapped delta is the personalisation signal in isolation.
Ranking with the wrong persona costs 0.33 nDCG@5 under the val labels, so a
persona-sensitive encoder must degrade visibly. A null result here means the Stage-1
gain is generic retrieval quality and the personalisation claim rests entirely on
Stage 2.

**Ceiling, for calibration.** Computed from the val teacher scores with no model
involved: a perfect *persona-blind* ranker reaches nDCG@5 0.878; a perfect
persona-matched one reaches 1.000. The 0.122 between them is the entire personalisation
headroom at Stage 1, and it is small next to the 0.356 of generic-retrieval headroom
still open below 0.878.
- Filter thresholds for C and D: `min_positive_margin: 0.05`, `min_positive_score: 0.4`,
  `min_negative_margin: 0.25`. Record retained/total from `{split}_triplets_meta.json`
  alongside each result: train `929/1296`, val `206/276`.
- **D changes the serving contract.** `doc_frozen` trains the query tower against a
  document tower with the adapter off, so its index must be built the same way. Do not
  compare D against A–D's numbers without confirming the eval encoded the corpus
  base-only (`doc_base_only` in `evaluate_retrieval` handles this automatically).
- **No row is reportable until it has been paired-tested.** Run
  `benchmarks/compare_runs.py` on the downloaded `training_log.json`: it pairs the
  per-query vectors, gives a bootstrap CI and a sign-flip permutation p-value Holm-
  corrected across the metric family, and breaks down per persona. With *n* = 92 per
  persona the differences at stake are the size of one standard error, and the untrained
  baseline's own per-persona nDCG@5 spread (0.5368 / 0.5591) already sits under one.
  Cross-arm runs also get a **baseline-parity check**: two arms on the same val set must
  produce byte-identical epoch-0 vectors, since the adapter is the identity there. A
  mismatch means something leaked into the eval path and voids the comparison.
