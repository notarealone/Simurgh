# Experiment Design

> Datasets, baselines, metrics, and evaluation protocol.

---

## Datasets

- **Domain** — 9th-grade Persian (فارسی نهم). Corpus: official textbook + gifted-schools edition + study guide (provenance in [data-extraction](data-extraction.md)).
- **Questions** — extracted from real exam papers into structured JSON (schema + provenance in [question-extraction](question-extraction.md)); ~618 questions across 7 sources: 131 real exam questions plus 487 AI-generated (78.8% of the pool, present in all three splits). See [question-extraction](question-extraction.md), "AI-generated questions".
- **Profiles** — synthetic learner personas (LLM-as-simulator; no real student data). A fixed set of 4 over 4 axes (comprehension, prior knowledge, learning goal, explanation style); 3 train + 1 test-holdout. Full schema in [personas](personas.md).
- **Grounding** — link each question to its answering corpus passage(s): gold passages for Recall@K and context for generation. Tag each question **grounded vs skill** and by **personalization headroom** — pure recall/grammar items carry little persona-fit signal; comprehension items carry the most.

Data is scarce, which constrains eval diversity and DPO volume. Mitigations to document and
pursue as needed (not core scope yet):

- [ ] Scrape more exam papers (real questions + gold answers via the [question-extraction](question-extraction.md) VLM pipeline) — highest ROI
- [x] Synthesize exam-seeded questions — done: `data/questions/ai_generated_questions.json`
      (imitation of real exam items via GLM 5.2; enters the question-level split pool
      alongside real exam files)
- [ ] Synthesize genuinely corpus-grounded questions — keep this separate from the
      exam-seeded artifact if the distinction is needed later.
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

- [x] Current DPO/ROPG assets are split by question, not row: train and validation share
  no `question_ref`. Six source families do cross splits, so report the limitation as
  question-disjoint but source-overlapping.
- [ ] Use validation only for within-arm checkpoint selection. Keep any later held-out
  end-to-end test separate from the 272-prompt DPO validation tournament.
- [ ] Freeze prompts, pair files, judge model, candidate generation settings, and seeds.
  A hash mismatch invalidates comparison caches.
- [ ] Report paired confidence intervals and Holm-corrected significance values.

## Ablation Studies

- [ ] Persona removed (no profile anywhere) — drop profile from rewriter and retriever KD signal
- [ ] Generator persona-**blind** vs persona-**aware** — does a capable-enough generator make the rewriter redundant?
- [ ] Rewriter **untrained** vs **DPO** (Rung 3 vs 4 — the rewriter's marginal contribution)
- [ ] Retriever **frozen Qwen3-Embedding-0.6B** vs **ROPG-KD** (Rung 2 vs 3 — the retriever's marginal contribution)
- [ ] Retriever: BM25 vs frozen Qwen3-Embedding-0.6B vs ROPG-KD (full retriever ladder)
- [ ] On-policy vs off-policy DPO pairs (iterative-DPO study) — optional
- [ ] Document findings in [results](results/)

## Stage-2 DPO-family runs

This experiment selects a rewriter before Rung 4. No retriever checkpoint, corpus, index,
retrieval metric, or answer generator enters training or model comparison.

All runs use Qwen3-4B in 4-bit, LoRA rank 16 on attention and MLP projections, one epoch,
learning rate `1e-5`, target global batch 8, beta 0.1, and sequence caps
576 prompt / 224 completion / 768 full. A 2x T4 launch uses per-device batch 1 and
gradient accumulation 4. Native DDP changes throughput only.

| Arm | TRL loss | Weighting | Label smoothing | RPO alpha | Purpose |
|---|---|---:|---:|---:|---|
| `dpo` | sigmoid | no | 0.0 | 1.0 | required baseline |
| `wpo` | sigmoid | yes | 0.0 | 1.0 | address Grok-to-Qwen off-policy data |
| `robust_dpo` | robust | no | 0.1 | 1.0 | test uniform preference-label noise |

Each arm selects its own maximum `eval_rewards/accuracies` checkpoint. Objective losses are
on different scales and never rank arms, and `eval_loss` no longer selects within an arm
either: the robust objective is unbounded below and the WPO loss carries a policy-dependent
weight, so a falling loss can mean a less confident policy rather than a better one.

The epoch count, the selection metric, and the RPO supervised anchor all changed after the
seed-42 screening run, which produced no arm better than the untrained policy. The evidence
behind each change is in [dpo-arms-seed42-v1](results/dpo-arms-seed42-v1.md).

`benchmarks/judge_agreement.py` measures whether Luna's pair labels predict the tournament
judge's verdicts on a random sample of the pairs. If they do not, the training target and
the evaluation target are different quantities and no arm can win.

Cross-arm ranking uses a blind Gemini-family pairwise judge. The exact endpoint and model
remain runtime values in `DPO_EVAL_BASE_URL`, `DPO_EVAL_API_KEY`, and `DPO_EVAL_MODEL`.
The judge must differ from Luna, which labeled the pairs, and Grok, which generated the
candidates.

The judge does not share the rewriter's transport. Metis serves the Gemini family over
Google's native GenAI protocol, so `DPO_EVAL_BASE_URL` is a bare host with no path
(`https://api.metisai.ir`) and the calls go through `rag.llm.GeminiClient`, which posts to
`{base_url}/v1beta/models/{model}:generateContent`. The OpenAI-compatible route that serves
Luna and Grok has no Gemini models on it.

Thinking cannot be switched off. Gemini 3.5 and newer reject the older `thinking_budget`
field with a 400, and their `thinking_level` control has no *off* value, so the judge sets
the lowest level the model accepts through `comparison.judge_thinking_level` (`minimal`,
which is already the flash-lite default). Reasoning tokens are billed against the same
`comparison.judge_max_tokens` budget as the reply, which is why that cap is 512 rather than
the verdict's own size. The verdict shape is enforced server-side by a two-field response
schema instead of by prompt wording alone. A judge response that is not a rate limit,
timeout, or 5xx aborts the whole tournament instead of retrying: a wrong route, model name,
or request field fails identically for every remaining call.

Seed-42 screening covers all 272 validation question/persona prompts and five candidates:
DPO, WPO, robust DPO, base Qwen3-4B, and temperature-zero prompted Grok. Ten model pairs
produce 2,720 judge calls. Hidden A/B orientation is deterministic and balanced. Candidate
scores use win 1, tie 0.5, loss 0. Reports include overall and per-persona results,
5,000-sample paired bootstrap intervals, and sign-flip tests with Holm correction.

Select WPO or robust DPO by round-robin score, then their direct head-to-head score, then
WPO on an exact tie. Train standard DPO and the selected variant at seeds 43 and 44.
Each added seed judges five fixed pairings on all prompts, 1,360 calls per seed. Total
budget is 5,440 calls. Only the three-seed DPO-versus-variant comparison supports a thesis
claim.

If all trained arms lose to base Qwen or Grok, or validation loss improves while the
independent ranking worsens, record failed transfer. That result opens a separate decision
about Qwen on-policy candidates and persisted score/pair provenance. It does not trigger
automatic regeneration.

## Stage-1 retriever runs (ROPG `hard_neg`, Luna-era labels)

Four runs isolate each stage-1 change against the untrained encoder. Pairwise attribution
uses A vs B for anchoring, B vs C for filtering, and C vs D for `doc_frozen`. All share
`mode: hard_neg`, `format: triplets`, `lr: 5.0e-5`, `epochs: 3`, `max_negatives: 8`,
`seed: 42`.

| Run | `data.train_data` | `anchor.mode` | Isolates | Status |
|---|---|---|---|---|
| A | `data/ropg_kd` | `none` | plain MNRL at the corrected LR/epoch budget — the control | **done** — best epoch 2; `models/ropg/ropg_kd_runA_model/` |
| B | `data/ropg_kd` | `both` | base-model anchoring | **done** — best epoch 1; `models/ropg/ropg_kd_runB_model/` |
| C | `data/ropg_kd_filtered` | `both` | label filtering | **done** — best epoch 1; `models/ropg/ropg_kd_runC_model/` |
| D | `data/ropg_kd_filtered` | `doc_frozen` | the asymmetric (frozen document tower) arm | **done** — best epoch 1; `models/ropg/ropg_kd_runD_model/` |

Both data directories are derived from the *same* judged `{train,val}.jsonl` and differ
only in whether the label filters ran — `configs/datagen_ropg.yaml` builds the
unfiltered one, `configs/datagen_ropg_filtered.yaml` the filtered one, and neither
re-judges anything. Selecting an arm is one key (`data.train_data`, or `ARM` in the
notebook). Every run logs the build it trained on
(`Triplets: max_negatives=8 | filters=True | groups 929/1296 retained`), which is the
after-the-fact check that the intended arm actually ran — `data.train_data` is exempt
from the notebook's config-drift check, because local and Kaggle roots legitimately
differ.

**Run order/status: all four runs are complete.** Best epochs are A 2, B 1, C 1, and D 1.
Adapters and logs are under `models/ropg/ropg_kd_run{A,B,C,D}_model/`; pairwise and
persona-swap reports are under `models/ropg/comparisons/`. See
[ropg-runs-comparison-v1](results/ropg-runs-comparison-v1.md) for the completed set.

**The arms are paired-comparable.** `src/rl/ropg_kd.py` loads its eval groups from
`{train_data}/val.jsonl` — the *scored* file, which `derive_triplets` never rewrites —
so filtering shrinks the training triplets while leaving all 276 val groups intact.
`val_loss` is the one exception: a filtered arm computes it over fewer triplets, so it
must not be compared across arms (`compare_runs.py` flags this).

**Negative selection is held fixed.** All A-C artifacts preserve the current
rank-13-through-rank-20 negative selection, despite the historical `hard_neg` name.
Changing that selection only for C would confound the B-vs-C comparison.

### Primary metric — revised after run B (2026-08-20; Luna-era labels)

**Headline pair: nDCG@1 and Recall@5**, both against the Luna-era untrained
Qwen3-Embedding-0.6B baseline (nDCG@1 0.5895, Recall@5 0.5640) — not against the
earlier trained checkpoints. Report the epoch-0 row in every table.

The original criterion was nDCG@5 > 0.548 **on the nano-era labels**. Run B forced a revision,
and the reason is recorded here rather than in a footnote because changing a success criterion
after seeing a result is exactly the move that needs justifying:

- **nDCG@5 is over half noise at this label quality.** Under the Luna-era labels, val mean
  teacher score by rank runs 0.799 / 0.372 / 0.176 / 0.108 / 0.072, so the adjacent gaps at
  slots 3→4 and 4→5 are 0.068 and 0.036. **33.2% of ideal DCG@5's mass sits in slots 2–5.**
  The nano-era MNRL objective trains only the rank-1-vs-tail contrast, where the historical gap
  was 0.711. Grading a rank-1 objective mostly by slots 2–5 measures the labels' noise, not the
  encoder.
- **nDCG@1 grades the one slot where the teacher is most separated** — the Luna-era rank-1
  mean is 0.799, **0.427 clear of rank 2**, so it is directly readable.
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
nDCG for improving. The Luna-era epoch-0 baseline has `judged@5 = 1.0` by construction:
the candidate pool was mined with the base encoder, so trained runs necessarily give some
of that coverage back when they move outside the pool. Recall@5 up with nDCG@5 down is
consistent both with "the graded middle got worse" and with "the retrieved set moved outside
the judged pool"; `judged@5` is the only thing that separates them. Never report nDCG without it.

**`persona_swap` is a required control.** The evaluator re-scores each validation query
under a rotated, valid persona different from the group's `persona_id`. "Mismatched"
means that the substituted profile differs from the persona whose fixed,
persona-conditioned teacher labels remain in use. It does not mean that the substituted
persona is invalid. The raw query, corpus embedding, and labels stay fixed, so only the
`Instruct:` prefix changes. The matched-minus-swapped delta tests whether the encoder
uses that prefix. Under the Luna-era labels, this deliberate mismatch costs 0.1363
nDCG@5. All four executed runs returned a null swap effect (A/B/C/D: −0.0017 / −0.0043 /
−0.0053 / +0.0008, all p = 1.0000), so the Stage-1 gain is generic retrieval quality
and the personalisation claim rests entirely on Stage 2.

**Ceiling, for calibration.** Computed from the Luna-era val teacher scores with no model
involved: a perfect *persona-blind* ranker reaches nDCG@5 0.9573; a perfect
persona-matched one reaches 1.0000. The 0.0427 between them is the entire personalisation
headroom at Stage 1, and the generic-retrieval headroom below 0.9573 is 0.9573 − 0.5861 =
0.3712 from the Luna-era baseline.
- Filter thresholds for C and D: the live `configs/datagen_ropg_filtered.yaml` values are
  `min_positive_margin: 0.05`, `min_positive_score: 0.4`, and `min_negative_margin: 0.25`
  (`filters.enabled: true`), matching `{split}_triplets_meta.json`. The unfiltered
  `configs/datagen_ropg.yaml` carries `0.08 / 0.4 / 0.3` with filters disabled. Record
  retained/total alongside each result: train `929/1296`, val `206/276`.
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
