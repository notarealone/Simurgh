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

- **Query rewriter** — trained with DPO (Gemma-4-E4B + LoRA). Reads the learner profile
  and the raw query and emits a reformulated, persona-conditioned query.
- **Retriever** — trained with ROPG-KD (BGE-M3 dense encoder, fine-tuned). An LLM judge
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

The retriever is a trained component (Stage 1, before the rewriter). BGE-M3 is validated
on Persian first, then fine-tuned with ROPG-KD.

- [ ] Measure retrieval quality (Recall@K, MRR) for BM25 vs BGE-M3 frozen — this is Rung 1 vs Rung 0
- [ ] Index BGE-M3 with FAISS; rebuild and version the index whenever embeddings or chunking change
- [ ] Validate frozen BGE-M3 on held-out Persian QA before committing to ROPG-KD fine-tuning
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

- **Encoder** — BGE-M3, fine-tuned with a LoRA adapter.
- **Teacher signal (direct document scoring):** for each `(query, persona)` pair in the
  train set, retrieve top-K candidate documents and call the LLM judge once per
  `(query, persona, document)` triple. The judge scores how useful this document is for
  answering the question for a student with this profile, on a 1–10 rubric (persona fit +
  pedagogical value + relevance). Scores are stored offline.

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
- **Guard:** report Recall@K/MRR per persona on the val set throughout training to
  catch reward hacking (an encoder that scores well on the judge rubric but retrieves
  nothing useful).

### Stage 2 — Rewriter: DPO

The trained rewriter policy is built on top of the **fixed** ROPG-KD retriever.

- **Policy** — Gemma-4-E4B with a LoRA adapter. A frozen copy is the DPO reference.
  Qwen2.5-3B is the fallback if Persian output quality is insufficient (validated by
  smoke-testing rewrites before training).
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

- [ ] Validate frozen BGE-M3 on Persian (Recall@K/MRR vs BM25); commit to BGE-M3 if it matches or beats BM25
- [ ] Run the offline scoring pipeline: for each `(query, persona, document)` triple in the train set, call the judge and store a utility score (`src/rl/scorer.py`)
- [ ] KD-train the BGE-M3 LoRA adapter; select checkpoint on val Recall@K per persona
- [ ] Freeze the ROPG-KD retriever checkpoint before Stage 2

**Stage 2 — DPO rewriter**

- [ ] Smoke-test Gemma-4-E4B Persian output (5–10 sample rewrites); fall back to Qwen2.5-3B if quality is poor
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
