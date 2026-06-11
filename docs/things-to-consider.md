# Things to Consider & TODOs

> Updated as the project evolves.

---

## Reward Signal

- [ ] Define what the LLM-as-judge evaluates (relevance to persona, pedagogical quality, faithfulness — or all three?)
- [ ] Design the scoring rubric
- [ ] Decide: use judge only for preference pairs → train a separate reward model, or use raw judge scores as RL reward?
- [ ] Plan a human evaluation sample to validate that judge scores correlate with pedagogical quality

## Personalization Mechanism

- [ ] Pick a concrete approach:

  | Approach | How | Complexity |
  |---|---|---|
  | Profile-conditioned query rewriting | Prepend user profile to query before rewriting | Low |
  | Persona tokens in retriever | Add learned persona embeddings to query/doc embeddings | Medium |
  | Per-user retrieval policy | RL agent selects retrieval strategy based on user state | High |
  | Adaptive retrieval parameters | RL adjusts top-K, similarity threshold, chunk size per user | Medium |

- [ ] Document the choice in [[methodology]]

## Evaluation Plan

- [ ] Define the baseline ladder:
  1. Naive RAG (no personalization, no RL)
  2. RAG + persona prompting (prepend profile to query, no RL)
  3. RAG + DPO on rewriter only
  4. Full system (DPO rewriter + RL-optimized retriever)
- [ ] Add metrics beyond EM/F1:
  - LLM-as-judge score on held-out set
  - Human evaluation (50–100 samples)
  - Retrieval metrics (Recall@K, MRR) per persona

## Dataset Creation

- [ ] Set dataset size (questions × profiles per question)
- [ ] Decide: synthetic profiles (LLM-generated) or real student data?
- [ ] Decide: who writes ground-truth answers?
- [ ] Document in [[experiment-design]]

## Open Questions

- [ ] Which Persian textbooks and grade levels to target?
- [ ] Which RL algorithm for retriever? (PPO, DPO, REINFORCE, other)
- [ ] English vs Persian system prompt — which yields better Persian answers? Both exist as a config-selectable `prompt_variant` (`en` default); compare once the eval harness lands. Expectation: a wash on large API models, a model-specific tradeoff on small/local ones (English aids instruction-following; Persian reduces English leakage).

**Decided**

- *Retriever starting point.* Phase 0 uses lexical BM25 (SQLite FTS5), a baseline simpler than DPR. BGE-M3 dense retrieval stays the target for the trainable rungs; validate it on held-out Persian QA before committing.
- *Generator access.* One OpenAI-compatible client serves both API models (OpenAI, Google AI Studio) and local servers (LMStudio, llama.cpp); the endpoint and key come from the `OPENAI_BASE_URL` and `OPENAI_API_KEY` environment variables, the model from config. Still open: which small (≤7B) model to fine-tune for the trainable rungs.
