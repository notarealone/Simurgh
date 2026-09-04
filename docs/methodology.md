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
  the encoder so it ranks documents by persona utility. Retriever training and evaluation
  are separate from Stage 2. The DPO trainer never loads a retriever, index, corpus, or
  retrieval metric. Component integration belongs to the later Rung 4 experiment.
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

Two components are trained in separate stages: the retriever with ROPG-KD and the query
rewriter with DPO-family objectives. Neither stage loads or differentiates through the
other. The later Rung 4 experiment combines their selected checkpoints.

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
- **Temperature scope:** `student_temp` and `teacher_temp` feed `kd_loss` only. They are inert under the primary `hard_neg` (MNRL) arm; their ablation concerns only the retained `reader_kd` ablation.
- **Guard:** report Recall@K/MRR per persona on the val set throughout training to
  catch reward hacking (an encoder that scores well on the judge rubric but retrieves
  nothing useful).
- **Validation relevance definition:** two notions are reported side by side.
  - **nDCG@1–K** grades each judged doc by its raw teacher score (linear gain;
    scores already lie in [0,1], so exponential gain would only compress them). No cutoff
    is drawn, and the ceiling is 1.0 at every K. This is the honest reading of a graded
    judge: measured on the Luna-era `val.jsonl`, mean teacher score by rank runs
    0.799 / 0.372 / 0.176 / 0.108 / 0.072, so a binary top-3 set counts the 1st and 3rd
    doc identically despite a ~4.5× difference in judged utility (rank 1 / rank 3 =
    0.799 / 0.176 ≈ 4.5×) — and the rank-3 vs rank-4 gap is **under 0.05 in 61.6% of
    groups**, meaning the boundary mostly separates near-ties. **The same graded reading
    cuts the other way deeper in the list**, which is why the headline metric is nDCG@**1**,
    not nDCG@K: the rank-1 mean is 0.799, **0.427 clear of rank 2**, while adjacent teacher
    gaps at slots 3→4 and 4→5 are only 0.068 and 0.036, at the label noise floor, and
    33.2% of ideal DCG@5's mass sits in slots 2–5. See [experiment-design](experiment-design.md),
    "Primary metric — revised after run B".
  - **judged@K (diagnostic)** — the share of the returned top-K the teacher ever scored.
    Only ~20 of 171 corpus chunks are judged per group, so a model that surfaces
    unjudged-but-relevant chunks is penalised by nDCG for improving. Recall up with nDCG
    down fits both "the graded middle got worse" and "the ranking left the judged pool";
    this is what separates them. The untrained baseline's `judged@5 = 1.0` is structural:
    the 20-document candidate pool was mined with the base encoder, so trained runs
    necessarily give some of that coverage back when they move outside the pool.
  - **Persona-mismatch (counterfactual swap) control.** The evaluator re-scores every
    validation query under a rotated, valid persona different from the group's
    ``persona_id``. "Mismatched" does not mean that the substituted persona is invalid.
    It means that the query prefix no longer matches the persona whose fixed,
    persona-conditioned teacher labels are being evaluated. The raw query, corpus
    embedding, and labels stay fixed, so only the ``Instruct:`` prefix changes. A
    persona-sensitive encoder should therefore lose agreement with the original labels.
    Under the current Luna-era validation labels, this deliberate mismatch costs 0.1363
    nDCG@5. The implementation is enabled by ``eval.persona_swap``.
  - **Recall@K / Hit@K / MRR** keep the binary set: the group's **top-3 docs by teacher
    score within the judged top-20 candidates**. Retained deliberately — nDCG ranges over
    the same graded distribution the KD loss is trained on, so a coarser, differently
    shaped metric belongs beside it as the reward-hacking guard. Note Recall@k divides by
    that set's size, so Recall@1 could never exceed 1/3; it is reported at K only.
  - *Alternatives considered:* a score threshold (e.g. ≥ 0.7) — semantically closer to
    "relevant", but measured on the Luna-era `val.jsonl` it leaves **58/276 groups with no
    relevant doc at all** (63/276 at 0.8), and such groups are skipped entirely, so any
    usable threshold silently discards 21–23% of the val set; and strict top-1 — simpler,
    but brittle under exactly the near-ties quantified above.
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


#### Why `reader_kd` was retired as the primary arm — nano-era labels
**Nano-era evidence (`old_v2` labels, pre-`ff90cf6`).** The table below is retained as history; it must not be mixed with the current Luna-era run results.

Both arms were trained to completion on 2×T4. **Neither beat the untrained
Qwen3-Embedding-0.6B baseline on nDCG@5**, at the time the sole primary metric (run B
later forced that criterion to be revised — see
[experiment-design](experiment-design.md), "Primary metric — revised after run B").
Read under the revised headline pair, the `hard_neg` row below was already showing the
pattern run B later showed much more strongly — **Recall@5 and MRR above baseline while
nDCG@5 sits below it**. That was visible here and not acted on. nDCG@1 was not recorded
for these two runs, so the comparison cannot be completed retrospectively:

| Run | nDCG@5 | Hit@5 | Recall@5 | MRR |
|---|---|---|---|---|
| baseline (untrained, epoch 0) | **0.548** | **0.786** | 0.396 | 0.602 |
| `reader_kd`, best epoch (2/3) | 0.459 | 0.768 | 0.387 | 0.620 |
| `hard_neg`, best epoch (1/4) | 0.509 | 0.779 | **0.402** | **0.633** |

The objective conclusion still holds under the Luna labels: MNRL reads the rank-1-vs-tail contrast, while KD softmaxes the noisy graded middle; the Luna rank-1 gap is even wider.
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
`gen_ropg_data.py` substituted a subjective usefulness rating from `gpt-5.4-nano`
in the nano-era labels and sampled **once at `temperature: 1.0`**. There is no reader in the loop at all; the mode
name is aspirational. Measured on the nano-era `data/ropg_kd/old_v2/train.jsonl` (1296 groups,
432 questions × 3 personas, 171-chunk corpus):
The current teacher is `gpt-5.6-luna`, but its per-label reliability has not been re-measured. The Luna score distribution is much sparser: 68.9% of val document scores are exact zero across 20 documents per group. Therefore, the reliability and histogram values below describe the nano-era `old_v2` labels only.

| Evidence | Value | Reading |
|---|---|---|
| split-half *r*, per-chunk mean teacher score (~10 queries/half) | 0.674 | single-label reliability ≈ **0.17** |
| split-half *r*, per-chunk (crammer − scholar) contrast | 0.420 | persona signal is real but faint |
| variance that is within-cell (across personas, same query+chunk) | 23.7% | most score variance is noise, not persona |
| train groups dropped by margin < 0.05 | 258/1296 | the first sequential filter in the Luna-era filtered build removes ambiguous positives |
| train groups dropped by best score < 0.4 after the margin filter | another 109 | the second filter in the Luna-era build removes groups with no useful positive |
| train groups retained after both filters | **929/1296** | the current Luna-era filtered build |
| validation groups retained after both filters | **206/276** | the current Luna-era filtered build |
| historical mean positive − negative gap (rank-1 vs ranks 17–20) | **0.711** (sd 0.183) | the historical extremes are far above the noise floor |
| mean adjacent-rank gap, middle of the list | **0.02–0.03** | the graded middle is below it |

For these nano-era labels, roughly 83% of every individual `teacher_score` is noise. The score
histogram also shows the quantization this predicts, with mass piling on 0.05 / 0.12 / 0.15 /
0.18 / 0.62 / 0.78 / 0.85; these histogram values describe `data/ropg_kd/old_v2/` only.

**Why that is fatal to KD but survivable for MNRL.** The last two rows are the whole
argument. `kd_loss` softmaxes the *entire* graded ranking, so most of its gradient is
spent asking the encoder to reproduce adjacent-rank differences smaller than the label
noise — coin flips. `mnrl_loss` reads the rank-1 versus selected-negative contrast. The
0.711 gap is the historical rank-17..20 measurement; current 8-negative artifacts select
ranks 13-20 in rank order, and this selection is held fixed through Runs A-C for
comparability. The same teacher supports one objective and not the other, which is why
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

1. **Label filters** (`configs/datagen_ropg_filtered.yaml` → `filters`, enabled).
   The live thresholds are `min_positive_margin: 0.05`, `min_positive_score: 0.4`, and
   `min_negative_margin: 0.25`; they match `data/ropg_kd_filtered/train_triplets_meta.json`.
   The unfiltered `configs/datagen_ropg.yaml` carries `0.08 / 0.4 / 0.3` with filters disabled.
   At the live thresholds, 929/1296 train groups and 206/276 validation groups are retained.
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

Reported honestly: under the Luna labels all four runs clear the untrained baseline. Run B is
the selected checkpoint at nDCG@5 0.7307 / Recall@5 0.6461 / MRR 0.8538 against the Luna-era
baseline 0.5861 / 0.5640 / 0.7853 (+0.1446 / +0.0821 / +0.0685), with `judged@5` falling
1.0000 → 0.8688. The old success criterion, "beat nDCG@5 0.548", was a nano-era criterion
and is not what the current runs were scored against. The persona-swap control returned null
for all four runs (see [ropg-runs-comparison-v1](results/ropg-runs-comparison-v1.md)), so Stage 1
demonstrates generic retrieval improvement but not demonstrated persona use.

#### Label eras — nano (retired) vs Luna (current)

Stage-1 labels exist in two eras. The retired **nano era** was judged by `gpt-5.4-nano` and is archived at `data/ropg_kd/old_v2/`. The current **Luna era** is judged by `gpt-5.6-luna` and lives at `data/ropg_kd/`. Commit `ff90cf6` (2026-08-28) switched the pipeline from nano to Luna; `format_version: 4` marks Luna rows.

Commit `ff90cf6` had to regenerate the scored files anyway: the same commit replaced stem-only queries with the complete rendered question and added bounded judge retries with whole-group skips, which invalidated every previously judged row. The teacher was upgraded `gpt-5.4-nano` → `gpt-5.6-luna` inside that already-required pass, so the upgrade cost **zero extra passes**. Per-call cost stays small by construction: the teacher emits a single decimal, capped at `judge.max_completion_tokens: 16` with `reasoning_effort: none`, so no reasoning tokens are billed — 31,440 calls of ≤16 completion tokens each. The nano labels' measured single-label reliability ≈ 0.17 was the motivation.

| Statistic | Nano era (`old_v2`) | Luna era (current) |
|---|---:|---:|
| Mean teacher score, ranks 1–5 | 0.757 / 0.569 / 0.447 / 0.372 / 0.323 | 0.799 / 0.372 / 0.176 / 0.108 / 0.072 |
| Adjacent gaps, ranks 1→2 / 2→3 / 3→4 / 4→5 | not recorded / not recorded / 0.075 / 0.049 | 0.427 / 0.196 / 0.068 / 0.036 |
| Rank-3 vs rank-4 gap under 0.05 | 53.3% of groups | 61.6% of groups |
| Ideal DCG@5 mass in slots 2–5 | 53.4% | 33.2% |
| Groups with no doc ≥ 0.7 (≥ 0.8) | 79/276 (125/276) | 58/276 (63/276) |
| Persona-blind ceiling | 0.8781 | 0.9573 |
| Persona-matched ceiling | not separately recorded | 1.0000 |
| Personalisation headroom | 0.122 | 0.0427 |
| Cost of ranking with a mismatched persona | 0.3369 nDCG@5 | 0.1363 nDCG@5 |
| Untrained baseline | nDCG@1 0.543; nDCG@5 0.548; Recall@5 0.3961; MRR 0.602; Hit@5 0.786 | nDCG@1 0.5895; nDCG@2 0.5445; nDCG@3 0.5596; nDCG@4 0.5727; nDCG@5 0.5861; Hit@1–5 0.6920 / 0.7645 / 0.8623 / 0.9058 / 0.9275; Recall@5 0.5640; MRR 0.7853; judged@5 1.0000 |

Every Stage-1 number in any document must name its label era. Only Luna-era numbers may be compared with results under `models/ropg/*`.

### Stage 2 — Rewriter preference optimization

Stage 2 trains and evaluates the query rewriter alone. It does not backpropagate through,
load, or select against a retriever. This keeps the preference experiment attributable to
the rewriter. Rung 4 later combines the selected rewriter with a separately selected
retriever.

- **Policy and reference** — Qwen3-4B loaded in 4-bit with a LoRA adapter on all attention
  and MLP projections. TRL receives `ref_model=None` only after PEFT wraps the model, so
  the reference is the same frozen Qwen weights with the adapter disabled.
- **Action** — emit one persona-conditioned rewrite from the learner profile and complete
  rendered question. Training and inference use the same Qwen chat template with
  `enable_thinking=False`.
- **Data eras** — Stage-2 pair files exist in two formats, and every Stage-2 number must
  name which one it was measured on.

  *Format 1 (scalar era, retired).* 1,739 training pairs over 432 questions and 351
  validation pairs over 92 questions, no shared `question_ref`. Candidates from
  `grok-4-1-fast`; labels from Luna as a single bare decimal at `temperature: 1.0` with a
  16-token budget, using the gold answer and explanation as judge-only context. `chosen` =
  best of 3, `rejected` = worst of 3, no margin filter. Rows stored no scores or pair type,
  so the trainer could not infer confidence, margin, or cross-persona provenance, and row
  order is not provenance. The 92 validation questions × 3 personas give 276 possible keys
  but only 272 carry a pair — 4 keys had fewer than two scored candidates — and
  `benchmarks/compare_dpo_rewriters.py:156-198` asserts exactly 272. All results in
  [dpo-arms-seed42-v1](results/dpo-arms-seed42-v1.md) are format-1 numbers.

  *Format 2 (rubric era, current).* 4 candidates per persona at temperatures
  0.2/0.5/0.8/1.1. Luna scores a three-criterion rubric — `meaning_preservation` 0.40,
  `persona_fit` 0.35, `specificity` 0.25 — at `temperature: 0` under a strict
  `json_schema` response format, replacing the single scalar, and **three samples per
  candidate** are averaged because that judge is not deterministic at `temperature: 0`
  ([judge-noise-luna-v1](results/judge-noise-luna-v1.md)). Pairs must clear
  `min_chosen_score: 0.55`, `min_margin: 0.04` — calibrated against measured judge noise,
  not guessed — and a near-duplicate check; cross-persona negatives are re-scored under
  the target persona so both sides of the comparison share one rubric context. Rows carry
  `pair_type`, both scores, the margin, both sub-score dicts, both temperatures, and
  `rejected_persona_id`; every candidate is dumped separately with its scores, so filters
  can be retuned without regenerating. The shipped files are the fourth data generation
  (2,658 train / 534 val over the same 432 / 92 questions) under the second schema:
  audit in [dpo-data-v4-audit](results/dpo-data-v4-audit.md), which reports 0.08% modeled
  label error, a measured persona-discrimination effect of 0.180, and a `scholar`
  negative-side skew of 1.6× that per-persona eval reporting must not pool away.
  Regenerating the pair files changes the 272 constant above. The data remains off-policy:
  Grok generates, Luna ranks, Qwen3-4B is the policy. Full recipe in
  [data-design](data-design.md).

  The format-1 → format-2 change was triggered by the pre-registered failed-transfer rule
  in [experiment-design](experiment-design.md): the seed-42 screening found no arm
  distinguishable from the untrained base, which opens the decision on candidate sourcing
  and persisted provenance. `rl.dpo_train.load_pairs` refuses format 1, so the two eras
  cannot be mixed by accident.
- **Objectives** — three fixed arms share all optimizer and batch settings, and each adds
  TRL's RPO supervised NLL term on the chosen rewrite at `rpo_alpha: 1.0`:
  - `dpo`: sigmoid DPO;
  - `wpo`: sigmoid DPO with policy-probability weighting, which directly tests the known
    Grok-to-Qwen distribution gap;
  - `robust_dpo`: robust DPO with label smoothing 0.1, which tests uniform label noise.

  The anchor exists because the seed-42 screening run grew every arm's margin by pushing
  both log-probabilities down rather than by making the chosen rewrite more likely, and
  judged quality fell in proportion to that displacement
  ([results](results/dpo-arms-seed42-v1.md)). TRL applies the anchor before the WPO weight,
  so the same alpha anchors the `wpo` arm roughly nine times more weakly on this data.
- **Sequence contract** — prompt cap 576, completion cap 224 including EOS, and full cap
  768. Preflight tokenizes every row with the production template and refuses to train if
  any row would truncate.
- **Optimization** — one epoch, learning rate `1e-5`, target global batch 8, fp16,
  8-bit AdamW, linear schedule, gradient checkpointing, and precomputed reference log
  probabilities. On 2x T4, each process uses per-device batch 1 and accumulation 4.
  Native TRL/Accelerate DDP replicates one QLoRA model per GPU. DDP changes wall time, not
  model quality.
- **Execution** — `notebooks/train_dpo.ipynb`, generated by `notebooks/build_train_dpo.py`,
  runs unattended top to bottom: preflight and the two-step DDP smoke once on the first
  arm, then all three arms in sequence, then generation, then the tournament. Each stage
  records its own outcome, so one failed arm neither aborts the arms that finished nor
  discards their artifacts, and the tournament runs only once every arm has a promoted
  adapter. `run_status.json` in the run root reports what actually completed.
- **Checkpoint selection** — each arm restores its own highest `eval_rewards/accuracies`
  checkpoint. Validation loss selects nothing: the robust objective is unbounded below, and
  the WPO loss is scaled by a policy-dependent weight whose shrinkage tracks falling policy
  log-probabilities, so it fell monotonically while preference accuracy also fell. Loss
  scales also differ across the three objectives, so validation loss never ranks arms.
- **Cross-arm ranking** — an independent Gemini-family judge sees the learner profile,
  complete question, judge-only gold context, and two anonymous rewrites. Seed 42 compares
  DPO, WPO, robust DPO, base Qwen3-4B, and temperature-zero prompted Grok on all 272 unique
  validation `(question_ref, persona_id)` prompts. That is 10 pairs and 2,720 calls.
  Stable hidden A/B orientation is balanced per model pair. Reports include win/tie/loss
  rates, round-robin score, persona slices, 5,000-sample paired bootstrap intervals, and
  Holm-corrected sign-flip tests.
- **Seed protocol** — screen all three arms at seed 42, select WPO or robust DPO by
  round-robin score, then direct head-to-head, then WPO on an exact tie. Train standard DPO
  and that variant at seeds 43 and 44. The added seeds judge five fixed pairings each,
  1,360 calls per seed and 5,440 calls total. Seed-42 screening alone is not a result.
- **Failure rule** — if every trained arm loses to base Qwen or Grok, or lower validation
  loss accompanies worse independent-judge ranking, report failed transfer. Do not
  regenerate data automatically.

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

**Stage 2 — DPO-family rewriter**

- [x] Freeze and validate `data/dpo/{train,val}.jsonl`; require explicit validation data,
  question-disjoint splits, known personas, unequal completions, and no duplicates.
- [x] Run exact-tokenizer preflight. Every prompt, completion, and full sequence must fit
  576 / 224 / 768 tokens without trainer truncation.
- [x] Run two optimizer steps for `dpo`, `wpo`, and `robust_dpo` on 2x T4.
- [x] Screen all three seed-42 arms and judge them. Outcome: no arm beat the untrained
  policy ([results](results/dpo-arms-seed42-v1.md)).
- [ ] Measure Luna/Gemini label agreement with `benchmarks/judge_agreement.py`. A near-chance
  result means the training and evaluation targets differ and the arms cannot be fixed by
  tuning.
- [ ] Retrain all three seed-42 arms for one epoch with the RPO anchor and promote each arm's
  maximum `eval_rewards/accuracies` checkpoint.
- [ ] Generate 272 greedy rewrites per local candidate and cache 272 temperature-zero Grok
  rewrites.
- [ ] Run the 2,720-call Gemini screening tournament, then train and judge DPO plus the
  selected variant at seeds 43 and 44.
- [ ] Report the three-seed paired comparison. Keep retriever and Rung 4 results out of this
  stage.

Hardware: both stages launch with `torchrun`. Stage 2 targets 2x T4 and derives gradient
accumulation from world size to keep global batch 8. The generator and judges remain API
calls; the DPO training loop makes none.

## Inference

User query + learner profile → the rewriter emits a persona-shaped query → the ROPG-KD
retriever returns context → the frozen generator produces the answer (profile in its prompt
on the persona-aware configuration). The same path serves every rung; rungs differ only by
config — untrained vs DPO rewriter, frozen vs ROPG-KD retriever, persona on/off, generator
persona-aware vs persona-blind.
