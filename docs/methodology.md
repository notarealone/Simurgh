# Methodology

> Detailed description of the proposed approach.

---

## System Architecture

<!-- High-level architecture diagram / description of the RAG + RL pipeline. -->

- [ ] Draw architecture diagram showing RAG pipeline + RL optimization loop
- [ ] Describe data flow from user query → personalized response

## RAG Pipeline

The pipeline grows in rungs (the baseline ladder in [[experiment-design]]). Each rung is a config over shared `src/` modules, not a separate codebase.

### Phase 0 — naive RAG (implemented)

Lexical retrieval, no personalization. Config: `configs/phase0_naive.yaml`. Demo: `notebooks/rag_demo.ipynb`.

- **Retriever** — SQLite FTS5 full-text index ranked by BM25 (`src/rag/store.py`). No embeddings; this baseline is simpler than DPR's dense retrieval on purpose.
- **Persian normalization** — the hazm normalizer plus Persian/Arabic digit folding, applied to both the indexed text and the query; ZWNJ is preserved (`src/data/persian.py`).
- **Corpus** — the raw markdown in `data/raw/` is OCR'd from source textbook PDFs and cleaned with an LLM; see [[data-extraction]] for the pipeline and per-file provenance.
- **Chunking** — fixed character windows with overlap (`src/data/chunking.py`); window size and overlap are config, not constants.
- **Generator** — one client for any OpenAI-compatible endpoint (`src/rag/llm.py`): OpenAI, Google AI Studio, LMStudio, or llama.cpp. The endpoint and key come from the `OPENAI_BASE_URL` and `OPENAI_API_KEY` environment variables (`.env`); the `model` stays in the config. The prompt tells the model to answer only from the retrieved context and to say when the answer is missing (`src/rag/prompts.py`). The prompt language is a config-selectable variant (`prompt_variant: en|fa`, both instruct a Persian answer) so the two can be compared — see [[things-to-consider]].

### Target — dense retrieval (planned)

- [ ] Replace BM25 with BGE-M3 dense (plus optional sparse) retrieval; index with FAISS
- [ ] Choose the generator for the trainable rungs (small ≤7B model, LoRA adapter)
- [ ] Rebuild and version the index whenever embeddings or chunking change

## RL Formulation

<!-- State space, action space, reward function, policy, training algorithm. -->

- [ ] Define state space (what the agent observes)
- [ ] Define action space (what the agent can change)
- [ ] Define reward function (how pedagogical quality is measured)
- [ ] Choose training algorithm (PPO, DPO, REINFORCE)

## Personalization Module

<!-- How user context/preferences are modeled and injected. -->

- [ ] Define user profile schema (grade level, learning style, etc.)
- [ ] Describe how profile influences query rewriting and retrieval
- [ ] Decide: profile-conditioned, persona embeddings, or per-user policy

## Training Procedure

<!-- Step-by-step training workflow. -->

- [ ] Write step-by-step training loop (data → preference pairs → DPO → RL)
- [ ] Specify training hyperparameters and hardware requirements
- [ ] Define stopping criteria and evaluation checkpoints

## Inference

<!-- How the trained system serves personalized responses. -->

- [ ] Describe inference pipeline: user query + profile → personalized response
- [ ] Specify latency and deployment considerations
