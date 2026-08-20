# Methodology

> Detailed description of the proposed approach.

---

## System Architecture

Simurgh personalizes a RAG pipeline for Persian educational text by training **two**
components: a persona-conditioned **query rewriter** (DPO) and a personalized **retriever**
(ROPG-KD). The generator is frozen; every measured gain traces to a specific trained
component.

Data flow for a single turn:

```
learner profile ─┐
                 ├──────────────────────────────────────────────────────┐
                 ▼                                                      ▼
   user query ─► [ query rewriter ] ─► persona-shaped query ─► [ retriever ] ─► context ─┐
                  (TRAINED, DPO)                                (TRAINED, KD)             ▼
                                                                             [ generator ] ─► answer
                                                                              (frozen)
```

- **Query rewriter** — trained with DPO (Qwen3-4B + LoRA). Reads the learner profile
  and the raw query and emits a reformulated, persona-conditioned query.
- **Retriever** — trained with ROPG-KD (Qwen3-Embedding-0.6B dense encoder, fine-tuned). An LLM judge
  scores each `(query, persona, document)` triple offline; those scores are distilled into
  the encoder so it ranks documents by persona-utility, not generic relevance. Trained
  before the rewriter (retriever is fixed when DPO pairs are built). Retrieval quality
  is reported **per persona** (Recall@K, MRR) as a diagnostic (see [experiment-design](experiment-design.md)).
- **Generator** — frozen, a light-but-big API model. Whether it *also* receives the profile
  is an **experimental axis**: a persona-*aware* generator personalizes the explanation
  directly; a persona-*blind* generator isolates the rewriter as the sole personalizer.

Every personalized path can switch back to a "no profile" path (the "profile is the
contract" rule).

## RAG Pipeline

The pipeline grows in rungs (the baseline ladder in [experiment-design](experiment-design.md)). Each rung is a
config over shared `src/` modules, not a separate codebase.

### Phase 0 — naive RAG (implemented)

Lexical retrieval, no personalization. Config: `configs/phase0_naive.yaml`. Demo: `notebooks/rag_demo.ipynb`.

- **Retriever** — SQLite FTS5 full-text index ranked by BM25 (`src/rag/store.py`). No embeddings; this baseline is simpler than DPR's dense retrieval on purpose.
- **Persian normalization** — the hazm normalizer plus Persian/Arabic digit folding, applied to both the indexed text and the query; ZWNJ is preserved (`src/data/persian.py`).
- **Corpus** — the raw markdown in `data/raw/` is OCR'd from source textbook PDFs and cleaned with an LLM; see [data-extraction](data-extraction.md) for the pipeline and per-file provenance.
- **Chunking** — fixed character windows with overlap (`src/data/chunking.py`); window size and overlap are config, not constants.
- **Generator** — one client for any OpenAI-compatible endpoint (`src/rag/llm.py`): OpenAI, Google AI Studio, LMStudio, or llama.cpp. The endpoint and key come from the `OPENAI_BASE_URL` and `OPENAI_API_KEY` environment variables (`.env`); the `model` stays in the config. The prompt tells the model to answer only from the retrieved context and to say when the answer is missing (`src/rag/prompts.py`). The prompt language is a config-selectable variant (`prompt_variant: en|fa`, both instruct a Persian answer) so the two can be compared — see [things-to-consider](things-to-consider.md).

### Retriever — trained with ROPG-KD (planned)

The retriever is a trained component (Stage 1, before the rewriter). Qwen3-Embedding-0.6B is
validated on Persian first, then fine-tuned with ROPG-KD.

- [ ] Measure retrieval quality (Recall@K, MRR) for BM25 vs Qwen3-Embedding-0.6B frozen — this is Rung 1 vs Rung 0
- [ ] Index Qwen3-Embedding-0.6B with FAISS; rebuild and version the index whenever embeddings or chunking change
- [ ] Validate frozen Qwen3-Embedding-0.6B on held-out Persian QA before committing to ROPG-KD fine-tuning
- [ ] Run ROPG-KD offline scoring pipeline (judge scores per triple), then KD training
- [ ] Report Recall@K/MRR per persona on val set after ROPG-KD to confirm retriever gains

## RL Formulation

Two components are trained, in order: the retriever (ROPG-KD) and then the query
rewriter (DPO). Fixing the retriever before building DPO pairs ensures preference
labels do not shift under the rewriter during training.

### Stage 1 — Retriever: ROPG-KD

ROPG-KD is the offline, knowledge-distillation variant of the ROPG-RL method (Salemi
et al., 2024). It trains the retriever without an online reward loop, fitting the
DPO-only compute constraint.

- **Encoder** — Qwen3-Embedding-0.6B, fine-tuned with a LoRA adapter.
- **Teacher signal (direct document scoring):** for each `(query, persona)` pair in the
  train set, retrieve top-K candidate documents and call the LLM judge once per
  `(query, persona, document)` triple. The judge scores how useful this document is for
  answering the question for a student with this profile, as a 0–1 utility score
  (persona fit + pedagogical value + relevance). Scores are stored offline in
  `data/ropg_kd/{train,val}.jsonl`.

  *Alternative considered:* generation-mediated scoring — generate a full answer using
  only this document as context, then score the answer. Rejected because it doubles API
  calls per triple (one generation + one judge call vs. one judge call) and adds
  generation noise that obscures the document's intrinsic utility. Direct scoring is
  simpler to implement, cheaper, and the rubric can directly target document-level
  pedagogical value.

- **KD loss:** the judge scores are softmaxed over the top-K documents per
  `(query, persona)` to form a soft target distribution; the encoder is trained to
  minimize KL divergence between its similarity distribution and the teacher's utility
  distribution. This steers the encoder toward ranking pedagogically useful documents
  first for each persona.
- **Training is fully offline and self-contained.** Once the teacher scores are stored,
  KD training involves *no other model*: not the rewriter (queries are the raw exam
  questions, the same ones the scoring pipeline retrieved with — the rewriter only
  enters in Stage 2, against this then-frozen retriever), not the generator, and not
  the judge itself — so a training run makes zero API calls and is re-runnable for
  free across seeds. One step processes one `(query, persona)` group: the query is
  embedded with the rendered persona profile as the Qwen3-Embedding instruction prefix
  (`Instruct: <profile>\nQuery: …`), the group's candidate documents are embedded with
  no instruction, and the KD loss above is applied to their cosine similarities. Only
  the LoRA adapter (`q_proj`/`k_proj`/`v_proj`/`o_proj`) receives gradients; the 0.6B base
  stays frozen.
  Because the persona conditions the *query side only*, documents are embedded
  persona-free — one shared FAISS index serves all personas at inference.
- **Encoder parity (train == serve):** the training encoder must be numerically identical
  to the serving one (`src/rag/embedder.py`) — Qwen3-Embedding's native **last-token**
  pooling, documents encoded bare, queries wrapped in the instruct prefix above. An
  earlier revision mean-pooled and used a raw `<profile>\n\n<query>` prefix; a LoRA
  adapter trained under one pooling head and read out through another loses most of what
  it learned, silently and with a healthy-looking loss curve. `benchmarks/check_encoder_parity.py`
  is the gate and asserts cosine ≥ 0.999 on both documents and queries; run it before
  trusting any ROPG-KD number.
- **KD temperatures:** student 0.05, teacher 0.2 — not 1.0. Student scores are cosine
  similarities in [-1, 1] and teacher scores lie in [0, 1], so at temperature 1.0 a softmax
  over the 20 candidates is near-uniform on *both* sides and the KD gradient is close to
  noise. Both values still need the ablation in [experiment-design](experiment-design.md)
  to confirm; they are a reasoned default, not a measured optimum.
- **Guard:** report Recall@K/MRR per persona on the val set throughout training to
  catch reward hacking (an encoder that scores well on the judge rubric but retrieves
  nothing useful).
- **Validation relevance definition:** two notions are reported side by side.
  - **nDCG@1–K (primary)** grades each judged doc by its raw teacher score (linear gain;
    scores already lie in [0,1], so exponential gain would only compress them). No cutoff
    is drawn, and the ceiling is 1.0 at every K. This is the honest reading of a graded
    judge: measured on `val.jsonl`, mean teacher score by rank runs
    0.757 / 0.569 / 0.447 / 0.372 / 0.323, so a binary top-3 set counts the 1st and 3rd
    doc identically despite a ~1.7× difference in judged utility — and the rank-3 vs
    rank-4 gap is **under 0.05 in 53% of groups**, meaning the boundary mostly separates
    near-ties.
  - **Recall@K / Hit@K / MRR** keep the binary set: the group's **top-3 docs by teacher
    score within the judged top-20 candidates**. Retained deliberately — nDCG ranges over
    the same graded distribution the KD loss is trained on, so a coarser, differently
    shaped metric belongs beside it as the reward-hacking guard. Note Recall@k divides by
    that set's size, so Recall@1 could never exceed 1/3; it is reported at K only.
  - *Alternatives considered:* a score threshold (e.g. ≥ 0.7) — semantically closer to
    "relevant", but measured on `val.jsonl` it leaves **79/276 groups with no relevant doc
    at all** (125/276 at 0.8), and such groups are skipped entirely, so any usable
    threshold silently discards 17–45% of the val set; and strict top-1 — simpler, but
    brittle under exactly the near-ties quantified above.
- **Significance:** per-query metric vectors are persisted to `training_log.json` each
  epoch so any two epochs can be compared with a **paired** bootstrap. This matters: with
  ~276 val queries a single epoch's marginal 95% CI is ≈ ±0.04, wide enough that two means
  differing by a real 0.05 still overlap. Because every epoch scores the same queries
  (per-query outcomes correlate ≈ 0.77), resampling the *difference* cancels shared query
  difficulty and is ~35% tighter. Epoch 0 — the untrained encoder, since a fresh LoRA
  adapter is exactly the identity — is the baseline every epoch is tested against.
- **Circular-dependency caveat:** Recall@K here measures whether the trained retriever
  agrees with the *same* LLM teacher that produced the training scores — not whether the
  retrieved documents actually contain the answer to the question. This is intentional for
  the KD objective (we want the retriever to internalise the teacher's preferences), but it
  means Recall@K is a training-time diagnostic, not an end-to-end quality guarantee.
  The true quality check is the ablation evaluation in [experiment-design](experiment-design.md):
  the judge there scores faithfulness to retrieved context and answer accuracy independently,
  which surfaces cases where the retriever found plausible-but-wrong documents.


#### Why `reader_kd` was retired as the primary arm

Both arms were trained to completion on 2×T4. **Neither beat the untrained
Qwen3-Embedding-0.6B baseline on nDCG@5**, the primary metric:

| Run | nDCG@5 | Hit@5 | Recall@5 | MRR |
|---|---|---|---|---|
| baseline (untrained, epoch 0) | **0.548** | **0.786** | 0.396 | 0.602 |
| `reader_kd`, best epoch (2/3) | 0.459 | 0.768 | 0.387 | 0.620 |
| `hard_neg`, best epoch (1/4) | 0.509 | 0.779 | **0.402** | **0.633** |

`hard_neg` epoch 1 dominates `reader_kd` epoch 2 on every metric, and is the only
trained checkpoint anywhere that exceeds the baseline on anything (Recall@5 +0.006,
MRR +0.031 — both inside noise). The two arms' **val losses are not comparable**:
KL over a 20-doc list and cross-entropy over 1 positive + 4 negatives are different
objectives on different scales, and reading the arms off those numbers inverts the
ranking the retrieval metrics give.

**Root cause — the teacher, not the objective.** In Salemi et al. the ROPG-KD target is
`Eval(y, M(φp(x,[d])))`: the frozen reader is run with only document *d* in context and
its output is scored against the **ground-truth label `y`**. Under greedy decoding that
is deterministic — re-run it and the number is identical, reliability 1.0. Our
`gen_ropg_data.py` substituted a subjective usefulness rating from `gpt-5.4-nano`,
sampled **once at `temperature: 1.0`**. There is no reader in the loop at all; the mode
name is aspirational. Measured on `train.jsonl` (1296 groups, 431 questions × 3 personas,
171-chunk corpus):

| Evidence | Value | Reading |
|---|---|---|
| split-half *r*, per-chunk mean teacher score (~10 queries/half) | 0.674 | single-label reliability ≈ **0.17** |
| split-half *r*, per-chunk (crammer − scholar) contrast | 0.420 | persona signal is real but faint |
| variance that is within-cell (across personas, same query+chunk) | 23.7% | most score variance is noise, not persona |
| groups where rank-1 − rank-2 < 0.05 | 34% | the triplet positive is a coin flip |
| groups where even the best doc scores < 0.5 | 10% | no useful doc exists; the positive is noise |
| mean positive − negative gap (rank-1 vs ranks 17–20) | **0.711** (sd 0.183) | the extremes are far above the noise floor |
| mean adjacent-rank gap, middle of the list | **0.02–0.03** | the graded middle is below it |

Roughly 83% of every individual `teacher_score` is noise. The score histogram also
shows the quantization this predicts, with mass piling on 0.05 / 0.12 / 0.15 / 0.18 /
0.62 / 0.78 / 0.85.

**Why that is fatal to KD but survivable for MNRL.** The last two rows are the whole
argument. `kd_loss` softmaxes the *entire* graded ranking, so most of its gradient is
spent asking the encoder to reproduce adjacent-rank differences smaller than the label
noise — coin flips. `mnrl_loss` reads only the rank-1 vs rank-17..20 contrast, where the
gap is 0.711. The same teacher supports one objective and not the other, which is why
`hard_neg` is now the primary arm and `reader_kd` is retained as a documented ablation
(`mode: reader_kd` still runs; `kd_loss` and `ScoredDataset` are untouched).

**A structural difference from LaMP, not just an implementation slip.** In LaMP each
user retrieves from **their own profile** — 55–205 personal documents, a different set
per user — so personalization is carried by the corpus and `Eval(y, ·)` never has to be
persona-aware. Simurgh shares **one 171-chunk corpus across all personas**, so every bit
of personalization must come from *reordering the same documents*. Note the corpus size
itself is in-distribution for LaMP; what is missing is the per-user corpus variation.
The principled fix, and the adaptation it requires, is recorded in
[things-to-consider](things-to-consider.md).

**What this stage now does about it.** Three changes, each switchable so its contribution
is separable (see [experiment-design](experiment-design.md) for the run table):

1. **Label filters** (`configs/datagen_ropg.yaml` → `triplets.filters`, default off).
   Drop groups whose rank-1/rank-2 margin is below `min_positive_margin` or whose best
   doc scores below `min_positive_score` — precisely the 34% and 10% rows above.
2. **Base-model anchoring** (`configs/train_ropg.yaml` → `anchor`). Every trained
   checkpoint scoring *below* an untrained baseline while train loss collapses to 0.058
   is the signature of destructive drift, not of underfitting, so the adapter is
   penalised for moving away from the pretrained embedding. Two lambdas: personalization
   is a property of the **query side only** — documents carry no persona — so the
   document tower is held hard (`lambda_doc: 0.5`) while the query tower is left free
   enough to learn persona conditioning (`lambda_query: 0.05`). `mode: doc_frozen` is the
   limit case, encoding documents with the adapter off entirely.
3. **Overfitting budget.** `lr` 2e-4 → 5e-5 and `epochs` → 3, since `hard_neg`'s best
   epoch was its first.

Reported honestly: as of this revision, **stage 1 has no result that beats the untrained
encoder**, and the success criterion for the runs above is beating nDCG@5 0.548 — not
beating the earlier trained checkpoints.

### Stage 2 — Rewriter: DPO

The trained rewriter policy is built on top of the **fixed** ROPG-KD retriever.

- **Policy** — Qwen3-4B with a LoRA adapter. A frozen copy is the DPO reference.
- **Action** — emit a reformulated, persona-conditioned query.
- **Preference-pair construction** — for each `(query, persona)` in the train split:
  sample N=6 rewrites from the current policy at varying temperatures (0.3–1.3) to
  ensure diversity → the labeling judge scores each rewrite *directly* on how well it
  would help retrieve the right study material for this learner (0–1 scale) →
  chosen = highest score, rejected = lowest. Cross-persona negatives are added for
  free: scholar's best rewrite becomes crammer's rejected (and vice versa), gated by
  a minimum score gap to keep the signal meaningful.

  *Alternative considered:* end-to-end scoring — retrieve + generate through the
  fixed ROPG-KD retriever and frozen generator, then judge the final answer for
  persona fit + pedagogical quality + faithfulness. Rejected because it triples the
  API cost per (query, persona): N generation calls at 800 tokens each (≈ 1,800 extra
  calls for ~100 questions × 3 personas × 6 rewrites) on a thesis budget with no
  batch discount. The proxy judge's predicted retrieval quality is a practical
  substitute: rewrite framing and vocabulary are the primary lever for which passage
  depth is retrieved, and the judge can evaluate this without running the full pipeline.

  *Rewriter model note:* The initial implementation generated rewrites with a local
  Gemma-4-E4B model (via Unsloth, 4-bit quantised) to avoid API costs. Rewrite
  quality was insufficient — the quantised model produced repetitive or poorly
  personalised rewrites — so the rewriter was replaced with a remote Grok model
  (`grok-4-1-fast`), which is a different model family from the judge. The judge
  independence rule (judge that labels pairs ≠ judge that scores evaluation results)
  still holds; the rewriter is not a judge.
- **Algorithm** — DPO over the LoRA adapter. Optional SFT warmup if DPO from the base
  policy proves unstable.
- **Judge independence:** the judge that labels DPO pairs must differ in family from the
  judge that scores evaluation results.

## Personalization Module

Personalization enters through **profile-conditioned query rewriting** (chosen from the
options in [things-to-consider](things-to-consider.md)).

- **Profile schema** — a small, fixed, versioned set of learner profiles: **4 axes**, each an ordinal level (**Bad / Average / Good / Excellent**). Stored canonically as the ordinal record (for ablation and split-balancing) and **rendered to a natural-language description** for the prompt — numbers in a prompt steer behavior unreliably. Full schema (axes, levels, the four personas) in [personas](personas.md). The four axes: comprehension level, prior knowledge, learning goal, explanation style — three of which shift *what is retrieved*, the rewriter's lever.
- **Persona set** — 4 personas; **3 used in train/val, 1 (`newcomer`) held out for test only**. The holdout is an unseen *recombination* of axis values present in training (high comprehension + low prior knowledge), so the claim is **compositional generalization**, not 4-way preset selection ([personas](personas.md)).
- **Where the profile is injected** — always into the rewriter; into the generator only on the persona-aware arm of the comparison. Every personalized path has a "no profile" switch.

## Training Procedure

**Stage 1 — ROPG-KD retriever**

- [ ] Validate frozen Qwen3-Embedding-0.6B on Persian (Recall@K/MRR vs BM25); commit to Qwen3-Embedding-0.6B if it matches or beats BM25
- [x] Run the offline scoring pipeline: for each `(query, persona, document)` triple in the train set, call the judge and store a utility score (`notebooks/gen_ropg_data.ipynb`, mirroring `configs/datagen_ropg.yaml` → `data/ropg_kd/`)
- [ ] KD-train the Qwen3-Embedding-0.6B LoRA adapter; select checkpoint on val Recall@K per persona
- [ ] Freeze the ROPG-KD retriever checkpoint before Stage 2

**Stage 2 — DPO rewriter**

- [ ] Smoke-test Qwen3-4B Persian output (5–10 sample rewrites)
- [ ] Build the persona-conditioned preference dataset from the **train split only** ([question-extraction](question-extraction.md) questions × personas; pairs labeled by the Stage 2 judge)
- [ ] DPO-train the rewriter LoRA against a frozen reference; low LR, 1–3 epochs
- [ ] Select checkpoints on **validation** persona-fit (not train loss); watch for length/repetition hacking and policy degeneration
- [ ] Log seed, config, and model + index versions per run (reproducible by construction)

Hardware: ROPG-KD trains on one GPU and scales to several via DDP, launched with
`torchrun` (2x T4 on Kaggle). A small model + LoRA fits Kaggle / limited university
GPUs; the generator and judges are API calls.

## Inference

User query + learner profile → the rewriter emits a persona-shaped query → the ROPG-KD
retriever returns context → the frozen generator produces the answer (profile in its prompt
on the persona-aware configuration). The same path serves every rung; rungs differ only by
config — untrained vs DPO rewriter, frozen vs ROPG-KD retriever, persona on/off, generator
persona-aware vs persona-blind.
