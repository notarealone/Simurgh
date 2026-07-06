# Things to Consider & TODOs

> Updated as the project evolves. Open decisions live here with their rationale; settled
> ones move to **Decided**.

---

## Reward Signal

- [x] What the judge evaluates — persona fit + pedagogical quality + faithfulness (decomposed rubric)
- [x] How it is used — **proxy-judge preference pairs** for DPO on the rewriter; the judge scores each candidate rewrite directly on predicted retrieval fit (0–1 float) — no live retrieval or generation in the datagen loop
- [x] Judge independence — the pair-labeling judge family ≠ the eval-scoring judge family
- [ ] Write the concrete scoring rubric (text, scale, decomposition)
- [ ] Human-validation sample (30–50) confirming judge scores track pedagogical quality
- [ ] Pick the λ weighting; decide on-policy vs off-policy pair construction (iterative DPO)

## Personalization Mechanism

- [x] **Decided — profile-conditioned query rewriting + ROPG-KD retrieval.** Two trained components: the retriever (ROPG-KD) and the rewriter (DPO); persona conditions both. On one experimental arm, the generator prompt also receives the profile.
  - Caveat to test: on a 3-book corpus the rewriter's *marginal* uplift may be small, and a persona-aware generator may make it redundant above some capability threshold — framed as the persona-aware-vs-blind comparison in [experiment-design](experiment-design.md). A null/modest result here is still reportable.
- [x] Profile schema → [personas](personas.md): 4 axes (comprehension, prior knowledge, learning goal, explanation style), 4-level ordinal, 4 personas with `newcomer` as a recombination test-holdout. Big Five excluded (single-turn QA). Remaining: freeze exact prose wording + Persian rendering.

## Evaluation Plan

- [x] Baseline ladder (see [experiment-design](experiment-design.md)): 5 rungs — BM25 no-persona → Qwen3-Embedding no-persona → Qwen3-Embedding untrained rewriter → ROPG-KD untrained rewriter → ROPG-KD DPO rewriter
- [x] Metrics beyond EM/F1: LLM-judge (primary), human sample, Recall@K/MRR per persona (diagnostic)
- [x] Significance: paired tests + bootstrap CIs over ≥3 seeds
- [ ] Build and **freeze** the eval set + judge prompt + seeds before any training run

## Dataset Creation

- [x] Synthetic personas (LLM-generated); no real student data
- [x] Domain — 9th-grade Persian; questions from real exams ([question-extraction](question-extraction.md))
- [ ] Ground questions to corpus passages; tag grounded-vs-skill and personalization-headroom
- [ ] Set dataset size; data-scarcity mitigations (scrape exams, synth corpus-grounded Qs) tracked in [experiment-design](experiment-design.md)

## Open Questions

- [x] Which small (≤7B) model for the **rewriter** policy? → **Qwen3-4B + LoRA**
- [ ] Generator model (light-but-big API) and the **evaluation** judge model — the eval judge must come from a family disjoint from the data-generation judges below
- [x] Retriever final pick → **Qwen3-Embedding-0.6B** (Rung 1 dense base, fine-tuned via ROPG-KD in Rung 3); see Decided below.
- [ ] English vs Persian system prompt — which yields better Persian answers? Both exist as a config-selectable `prompt_variant` (`en` default); compare once the eval harness lands. Expectation: a wash on large API models, a model-specific tradeoff on small/local ones (English aids instruction-following; Persian reduces English leakage).

**Decided**

- *Trained components.* Two components are trained, in order: (1) the **retriever** (Qwen3-Embedding-0.6B + LoRA, ROPG-KD) and (2) the **query rewriter** (Qwen3-4B + LoRA, DPO). The generator remains frozen (a light-but-big API model). Training the retriever first ensures DPO preference pairs are built against a stable retriever.
- *Retriever.* Trained with **ROPG-KD** — now a core component, not a stretch. Qwen3-Embedding-0.6B fine-tuned offline via knowledge distillation from LLM judge scores (direct document scoring). ROPG-RL (online) remains out of scope. See Stage 1 in [methodology](methodology.md).
- *Generator personalization.* Persona-blind vs persona-aware is an experimental axis, not a fixed choice (the rewriter-redundancy question).
- *Retriever starting point.* Phase 0 uses lexical BM25 (SQLite FTS5). Qwen3-Embedding-0.6B is validated as the dense base (Rung 1) before ROPG-KD fine-tuning (Rung 3).
- *Rewriter policy model.* Qwen3-4B + LoRA.
- *ROPG-KD teacher signal.* Direct document scoring chosen over generation-mediated scoring. Rationale: direct scoring requires one judge call per `(query, persona, document)` triple vs. one generation + one judge call for generation-mediated, and avoids generation noise obscuring the document's intrinsic utility. The generation-mediated option is noted in [methodology](methodology.md) for completeness.
- *Data-generation judges.* **gpt-5.4-mini** for ROPG-KD teacher scoring, **gpt-5.4-nano** for DPO rewrite scoring; rewrite candidates are sampled from Gemma-4-E4B (the data-generation simulator); Qwen3-4B + LoRA is the policy being trained on those pairs. Rationale: price and generation time — the ROPG teacher scores 20 chunks per `(query, persona)` and warrants the stronger mini, while the DPO judge rates short rewrites where nano suffices. Per the guard-the-judge rule, the *evaluation* judge (still open above) must not reuse these models/prompts.
- *Generator access.* One OpenAI-compatible client serves both API models (OpenAI, Google AI Studio) and local servers (LMStudio, llama.cpp); the endpoint and key come from the `OPENAI_BASE_URL` and `OPENAI_API_KEY` environment variables, the model from config.
- *Checkpoint selection (ROPG-KD).* Best checkpoint = highest overall val Recall@K across personas (ties broken by lower val KD loss, then higher MRR) — not train/val KD loss alone. This guards against an encoder that minimises KL while degrading real retrieval (the reward-hacking check from Stage 1). Relevance for Recall@K/MRR is defined as the teacher's top-3 docs per `(query, persona)` group; alternatives considered are recorded in [methodology](methodology.md).
- *`src/rl/scorer.py` status.* Superseded by the datagen notebook (`notebooks/gen_ropg_data.ipynb`) and `configs/datagen_ropg.yaml`, which now produce `data/ropg_kd/{train,val}.jsonl` directly. The module is kept only pending cleanup and should not be treated as the source of truth for the ROPG-KD scoring pipeline.
