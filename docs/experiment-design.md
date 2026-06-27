# Experiment Design

> Datasets, baselines, metrics, and evaluation protocol.

---

## Datasets

- **Domain** — 9th-grade Persian (فارسی نهم). Corpus: official textbook + gifted-schools edition + study guide (provenance in [data-extraction](data-extraction.md)).
- **Questions** — extracted from real exam papers into structured JSON (schema + provenance in [question-extraction](question-extraction.md)); ~130 questions across 6 sources so far.
- **Profiles** — synthetic learner personas (LLM-as-simulator; no real student data). A fixed set of 4 over 4 axes (comprehension, prior knowledge, learning goal, explanation style); 3 train + 1 test-holdout. Full schema in [personas](personas.md).
- **Grounding** — link each question to its answering corpus passage(s): gold passages for Recall@K and context for generation. Tag each question **grounded vs skill** and by **personalization headroom** — pure recall/grammar items carry little persona-fit signal; comprehension items carry the most.

Data is scarce, which constrains eval diversity and DPO volume. Mitigations to document and
pursue as needed (not core scope yet):

- [ ] Scrape more exam papers (real questions + gold answers via the [question-extraction](question-extraction.md) VLM pipeline) — highest ROI
- [ ] Synthesize corpus-grounded questions with a big model (yields the gold-passage link for free)
- [ ] Persona-multiply for eval coverage; sample multiple DPO pairs per (question, persona)

## Baselines

The core ladder — each rung is a config over shared `src/` modules:

- [x] Rung 0 — naive RAG: lexical BM25, no persona (`configs/phase0_naive.yaml`, see [methodology](methodology.md))
- [ ] Rung 1 — persona-prompted generator + **untrained** rewriter (the baseline to beat)
- [ ] Rung 2 — persona-prompted generator + **persona-DPO** rewriter (the contribution)

Stretch / future (explicitly outside the RL-DPO core):

- [ ] Retriever adaptation via **REINFORCE/ROPG** (online RL, off the DPO-only constraint)
- [ ] Generator-DPO on a small model

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

- [ ] Persona removed (no profile anywhere)
- [ ] Generator persona-**blind** vs persona-**aware** — does a capable-enough generator make the rewriter redundant? (relate to generator size / the capability threshold)
- [ ] Rewriter **untrained** vs **DPO** (the rewriter's marginal contribution)
- [ ] On-policy vs off-policy DPO pairs (iterative-DPO study) — optional
- [ ] Retriever: no-retrieval floor / BM25 / BGE-M3
- [ ] Document findings in [results](results/)
