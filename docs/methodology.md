# Methodology

> Detailed description of the proposed approach.

---

## System Architecture

Simurgh personalizes a RAG pipeline for Persian educational text by training **one**
component with RL: a persona-conditioned **query rewriter**. Every other component is
frozen, so any measured gain traces to the rewriter rather than to the system at large.

Data flow for a single turn:

```
learner profile ─┐
                 ▼
   user query ─► [ query rewriter ] ─► persona-shaped query ─► [ retriever ] ─► context ─┐
                  (TRAINED, LoRA)                                (frozen)                 ▼
                                                                              [ generator ] ─► answer
                                                                               (frozen)
```

- **Query rewriter** — the only trained policy (small ≤7B model, LoRA, DPO). It reads the
  learner profile and the raw query and emits a reformulated, persona-conditioned query.
- **Retriever** — frozen (Phase 0: BM25; candidate: BGE-M3 dense). Because the rewriter
  shapes the query by persona, the retriever now sees a *persona-dependent* query, so
  retrieval quality is reported **per persona** as a diagnostic (see [experiment-design](experiment-design.md)).
- **Generator** — frozen, a light-but-big API model. Whether it *also* receives the profile
  is an **experimental axis**: a persona-*aware* generator personalizes the explanation
  directly, which may make the rewriter redundant above some generator-capability threshold;
  a persona-*blind* generator isolates the rewriter as the sole personalizer.

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

### Retriever — frozen backdrop (planned)

The retriever is shared infrastructure, not a trained component in the core. It is selected
once and held constant so the rewriter stays the only moving part.

- [ ] Measure retrieval quality (Recall@K, MRR) for the no-retrieval floor / BM25 / BGE-M3 dense, then fix one
- [ ] If BGE-M3: index with FAISS; rebuild and version the index whenever embeddings or chunking change
- [ ] Validate dense retrieval on held-out Persian QA before committing

## RL Formulation

The trained policy is the query rewriter; the learning signal is **offline preference
optimization (DPO)**, not online RL (PPO), per the compute constraints.

- **Policy** — the rewriter `π(rewrite | profile, query)`, a small ≤7B model with a LoRA adapter. A frozen copy is the DPO reference.
- **Action** — emit a reformulated, persona-conditioned query.
- **Reward (for building preferences)** — `retrieval_quality + λ · persona_fit`:
  - `persona_fit` — an LLM-judge score on the *final* answer (persona fit + pedagogical quality + faithfulness), credit-assigned end-to-end through the frozen retriever and generator.
  - `retrieval_quality` — an objective anchor (e.g. Recall@K vs. gold passages) that guards against reward hacking — a rewrite that games the judge but retrieves nothing useful.
  - The labeling judge that builds pairs must differ in family from the eval judge that scores results (judge independence — [experiment-design](experiment-design.md), [things-to-consider](things-to-consider.md)).
- **Preference-pair construction** — on-policy preferred: sample several rewrites from the current policy → each is retrieved + generated (frozen) → the labeling judge scores each final answer → chosen = highest, rejected = lowest. Iterating this (resample from the *updated* policy) is **iterative DPO**, the closest offline analogue to online RL. A cheaper off-policy start (pairs from a big model) is the fallback; on-policy vs off-policy is itself a reportable comparison.
- **Algorithm** — DPO over the LoRA adapter. Optional SFT warmup on the "chosen" rewrites only if DPO from the base policy proves unstable.

## Personalization Module

Personalization enters through **profile-conditioned query rewriting** (chosen from the
options in [things-to-consider](things-to-consider.md)).

- **Profile schema** — a small, fixed, versioned set of learner profiles: **4 axes**, each an ordinal level (**Bad / Average / Good / Excellent**). Stored canonically as the ordinal record (for ablation and split-balancing) and **rendered to a natural-language description** for the prompt — numbers in a prompt steer behavior unreliably. Full schema (axes, levels, the four personas) in [personas](personas.md). The four axes: comprehension level, prior knowledge, learning goal, explanation style — three of which shift *what is retrieved*, the rewriter's lever.
- **Persona set** — 4 personas; **3 used in train/val, 1 (`newcomer`) held out for test only**. The holdout is an unseen *recombination* of axis values present in training (high comprehension + low prior knowledge), so the claim is **compositional generalization**, not 4-way preset selection ([personas](personas.md)).
- **Where the profile is injected** — always into the rewriter; into the generator only on the persona-aware arm of the comparison. Every personalized path has a "no profile" switch.

## Training Procedure

- [ ] Build the persona-conditioned preference dataset from the **train split only** ([question-extraction](question-extraction.md) questions × personas; pairs labeled as above)
- [ ] DPO-train the rewriter LoRA against a frozen reference; low LR, 1–3 epochs
- [ ] Select checkpoints on **validation** persona-fit (not train loss); watch for length/repetition hacking and policy degeneration
- [ ] Log seed, config, and model + index versions per run (reproducible by construction)

Hardware: a small model + LoRA fits Kaggle / limited university GPUs; the generator and
judges are API calls.

## Inference

User query + learner profile → the rewriter emits a persona-shaped query → the frozen
retriever returns context → the frozen generator produces the answer (profile in its prompt
on the persona-aware configuration). The same path serves every rung; rungs differ only by
config — untrained vs DPO rewriter, persona on/off, generator persona-aware vs persona-blind.
