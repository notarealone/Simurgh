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

Four runs isolate each stage-1 change against the untrained encoder. Every row differs
from **A** by exactly one component, which is what makes the attribution valid. All share
`mode: hard_neg`, `format: triplets`, `lr: 5.0e-5`, `epochs: 3`, `max_negatives: 8`,
`seed: 42`.

| Run | `triplets.filters.enabled` | `anchor.mode` | Isolates |
|---|---|---|---|
| A | `false` | `none` | plain MNRL at the corrected LR/epoch budget — the control |
| B | `false` | `both` | base-model anchoring |
| C | `true` | `both` | label filtering |
| D | `true` | `doc_frozen` | the asymmetric (frozen document tower) arm |

- **Success criterion: nDCG@5 > 0.548**, the untrained Qwen3-Embedding-0.6B baseline —
  *not* beating the earlier trained checkpoints, none of which cleared it. Report the
  epoch-0 row in every table.
- Filter thresholds for C and D: `min_positive_margin: 0.08`, `min_positive_score: 0.4`,
  `min_negative_margin: 0.3`. Record retained/total from `{split}_triplets_meta.json`
  alongside each result — expect roughly 800 of 1296 train groups retained.
- **D changes the serving contract.** `doc_frozen` trains the query tower against a
  document tower with the adapter off, so its index must be built the same way. Do not
  compare D against A–D's numbers without confirming the eval encoded the corpus
  base-only (`doc_base_only` in `evaluate_retrieval` handles this automatically).
- No claim from this table is reportable until the paired bootstrap in
  [things-to-consider](things-to-consider.md) exists: with *n* = 92 per persona, the
  differences at stake are the size of one standard error.
