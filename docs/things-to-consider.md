# Things to Consider & TODOs

> Updated as the project evolves. Open decisions live here with their rationale; settled
> ones move to **Decided**.

---

## Reward Signal

- [x] What the judge evaluates — persona fit + pedagogical quality + faithfulness (decomposed rubric)
- [x] How it is used — end-to-end **preference pairs** for DPO on the rewriter; reward `retrieval_quality + λ·persona_fit` (the retrieval term guards against judge-hacking)
- [x] Judge independence — the pair-labeling judge family ≠ the eval-scoring judge family
- [ ] Write the concrete scoring rubric (text, scale, decomposition)
- [ ] Human-validation sample (30–50) confirming judge scores track pedagogical quality
- [ ] Pick the λ weighting; decide on-policy vs off-policy pair construction (iterative DPO)

## Personalization Mechanism

- [x] **Decided — profile-conditioned query rewriting.** The rewriter is the single trained policy; persona conditions the rewrite (and, on one experimental arm, the generator prompt).
  - Caveat to test: on a 3-book corpus the rewriter's *marginal* uplift may be small, and a persona-aware generator may make it redundant above some capability threshold — framed as the persona-aware-vs-blind comparison in [experiment-design](experiment-design.md). A null/modest result here is still reportable.
- [x] Profile schema → [personas](personas.md): 4 axes (comprehension, prior knowledge, learning goal, explanation style), 4-level ordinal, 4 personas with `newcomer` as a recombination test-holdout. Big Five excluded (single-turn QA). Remaining: freeze exact prose wording + Persian rendering.

## Evaluation Plan

- [x] Baseline ladder (see [experiment-design](experiment-design.md)): naive RAG → persona-prompted + untrained rewriter → persona-prompted + DPO rewriter
- [x] Metrics beyond EM/F1: LLM-judge (primary), human sample, Recall@K/MRR per persona (diagnostic)
- [x] Significance: paired tests + bootstrap CIs over ≥3 seeds
- [ ] Build and **freeze** the eval set + judge prompt + seeds before any training run

## Dataset Creation

- [x] Synthetic personas (LLM-generated); no real student data
- [x] Domain — 9th-grade Persian; questions from real exams ([question-extraction](question-extraction.md))
- [ ] Ground questions to corpus passages; tag grounded-vs-skill and personalization-headroom
- [ ] Set dataset size; data-scarcity mitigations (scrape exams, synth corpus-grounded Qs) tracked in [experiment-design](experiment-design.md)

## Open Questions

- [ ] Which small (≤7B) model for the **rewriter** policy? (Gemma-class candidate)
- [ ] Generator model (light-but-big API) and the two judge models (disjoint families)
- [ ] Retriever final pick (BM25 vs BGE-M3), pending Recall@K
- [ ] English vs Persian system prompt — which yields better Persian answers? Both exist as a config-selectable `prompt_variant` (`en` default); compare once the eval harness lands. Expectation: a wash on large API models, a model-specific tradeoff on small/local ones (English aids instruction-following; Persian reduces English leakage).

**Decided**

- *Trained component.* The **query rewriter** (persona-conditioned, small ≤7B + LoRA, DPO) is the only trained policy; the generator is frozen (a light-but-big API model). Rationale: DPO needs a generative policy, and concentrating training on one component keeps gains attributable.
- *Retriever.* Not trained in the core — a frozen backdrop (BM25 now, BGE-M3 candidate), selected once via Recall@K and held constant. Optional stretch: REINFORCE/ROPG adaptation (online RL, off the DPO-only constraint).
- *Generator personalization.* Persona-blind vs persona-aware is an experimental axis, not a fixed choice (the rewriter-redundancy question).
- *Retriever starting point.* Phase 0 uses lexical BM25 (SQLite FTS5), a baseline simpler than DPR. BGE-M3 dense retrieval is the candidate for the frozen backdrop; validate it on held-out Persian QA before committing.
- *Generator access.* One OpenAI-compatible client serves both API models (OpenAI, Google AI Studio) and local servers (LMStudio, llama.cpp); the endpoint and key come from the `OPENAI_BASE_URL` and `OPENAI_API_KEY` environment variables, the model from config.
