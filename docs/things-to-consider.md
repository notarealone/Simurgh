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
- [ ] Which retriever as starting point? (multilingual E5, BGE-M3, domain-fine-tuned)
- [ ] Which generator LLM? (local model vs. API)
- [ ] Which RL algorithm for retriever? (PPO, DPO, REINFORCE, other)
