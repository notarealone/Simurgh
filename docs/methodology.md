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
  the LoRA adapter (`q_proj`/`v_proj`) receives gradients; the 0.6B base stays frozen.
  Because the persona conditions the *query side only*, documents are embedded
  persona-free — one shared FAISS index serves all personas at inference.
- **Guard:** report Recall@K/MRR per persona on the val set throughout training to
  catch reward hacking (an encoder that scores well on the judge rubric but retrieves
  nothing useful).
- **Validation relevance definition:** for Recall@K/MRR, the relevant set for a
  `(query, persona)` group is its **top-3 docs by teacher score within the judged top-20
  candidates** — calibration-free (only the ranking among judged docs matters) and every
  group contributes equally regardless of how the judge's absolute scores are distributed.
  *Alternatives considered:* a score threshold (e.g. ≥ 0.7) with a top-1 fallback for
  groups with no doc above it — semantically closer to "relevant" but sensitive to judge
  calibration drift across groups; and strict top-1 — simpler, but brittle under near-ties
  between the best few candidates.
- **Circular-dependency caveat:** Recall@K here measures whether the trained retriever
  agrees with the *same* LLM teacher that produced the training scores — not whether the
  retrieved documents actually contain the answer to the question. This is intentional for
  the KD objective (we want the retriever to internalise the teacher's preferences), but it
  means Recall@K is a training-time diagnostic, not an end-to-end quality guarantee.
  The true quality check is the ablation evaluation in [experiment-design](experiment-design.md):
  the judge there scores faithfulness to retrieved context and answer accuracy independently,
  which surfaces cases where the retriever found plausible-but-wrong documents.

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

Hardware: a small model + LoRA fits Kaggle / limited university GPUs; the generator and
judges are API calls.

## Inference

User query + learner profile → the rewriter emits a persona-shaped query → the ROPG-KD
retriever returns context → the frozen generator produces the answer (profile in its prompt
on the persona-aware configuration). The same path serves every rung; rungs differ only by
config — untrained vs DPO rewriter, frozen vs ROPG-KD retriever, persona on/off, generator
persona-aware vs persona-blind.
