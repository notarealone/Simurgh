### Project Information

A – Problem Definition:
Retrieval-Augmented Generation (RAG) systems are the standard approach for document-based question answering. Yet conventional RAG pipelines treat every user the same — they cannot personalize responses to a user's profile, learning needs, or comprehension level (for example, a ninth-grade student). This limitation matters most in education and Persian textbook comprehension, where one student needs step-by-step explanations while another benefits from a direct answer or a challenging question that builds critical thinking.

The core problem: personalize a RAG pipeline to respond to both textbook content and the user's persona and educational needs. This requires personalizing pipeline components — the Query Rewriter and Document Retriever — so the output matches the educational goals of each user.

B – Project Goal and Significance:
Design and implement a Reinforcement Learning (RL)-based optimization framework to personalize the RAG pipeline for Persian textbooks. The goal: improve educational effectiveness and reading comprehension for users with diverse learning profiles.

Secondary objectives:
* Use reinforcement learning to optimize the pipeline (query rewriting through retrieval) for pedagogical quality.
* Focus on Persian educational resources to build infrastructure for localized educational systems.
* Extend prior DPO-based work by adding RL optimization for the Retriever model.

C – Project Implementation Methodology:

1. Development and Improvement of the Query Rewriter
   * *Iterative DPO Training:* Use "LLM as a Judge" to generate preference pairs. Train the rewriter with iterative Direct Preference Optimization (DPO).

2. Reward Definition
   * Use the LLM-as-Judge output (e.g., a 1–10 score) as the reward signal for training an RL policy model.

3. Optimizing the Retriever Model with Personalization
   * Personalizing Embeddings
   * Using Reward-based Fine-tuning

4. Evaluation and Validation
   * Create a dataset of Persian textbook questions with diverse user profiles (beginner, advanced, challenging).
   * *Primary metric — persona alignment / pedagogical quality:* scored by an LLM-as-judge rubric and validated against a human-evaluated sample. EM and F1 measure answer-string correctness, not learner fit — two equally correct answers can serve very different students.
   * *Secondary metrics — correctness and retrieval:* Exact Match and F1 against standard RAG baselines, plus retrieval quality (Recall@K, MRR) per persona.

---

## Revision history

The section above is the original proposal and is kept verbatim. Subsequent design
decisions are recorded here as dated revisions rather than by editing the text above, so the
proposal's evolution stays auditable. The live, detailed plan lives in [methodology](methodology.md),
[experiment-design](experiment-design.md), and [things-to-consider](things-to-consider.md).

### v1.1 — 2026-06-26 — scope sharpened to a single RL-trained component

Refines, and in one place narrows, the original Implementation Methodology (section C) after
weighing it against the project constraints (≈4 weeks, solo, minimal compute, DPO over PPO).

- **Trained component narrowed to the query rewriter.** The persona-conditioned **query
  rewriter** (original step C.1) is now the *only* RL-trained policy — a small ≤7B model with
  a LoRA adapter, optimized with **DPO** (the iterative/on-policy variant when affordable).
  Concentrating training on one component keeps every measured gain attributable.
- **Retriever optimization (original step C.3) demoted to an optional stretch.** DPO needs a
  generative policy; an embedding retriever has no token log-probs, so optimizing it requires
  REINFORCE/ROPG (online RL) or contrastive fine-tuning — both outside the DPO-only core.
  The retriever is therefore a **frozen backdrop** (BM25 now, BGE-M3 candidate), selected
  once via Recall@K. Retriever adaptation via REINFORCE/ROPG remains a labelled stretch goal,
  pursued only if time allows.
- **Generator is frozen.** A light-but-big API model serves as the (frozen) generator.
  Whether it *also* receives the learner profile is an **experimental axis** (persona-aware
  vs persona-blind): a capable persona-aware generator may make the rewriter redundant above
  some capability threshold — itself a question the experiments probe. Training a *small*
  generator with DPO is recorded as future work, not core scope.
- **Reward (step C.2) made concrete.** `retrieval_quality + λ · persona_fit`, where
  `persona_fit` is an LLM-judge score on the final answer (persona fit + pedagogy +
  faithfulness) credit-assigned end-to-end, and `retrieval_quality` is an objective anchor
  guarding against reward hacking. **Judge independence:** the judge that *labels* DPO pairs
  must differ in family from the judge that *scores* evaluation.
- **Personalization mechanism decided.** Profile-conditioned query rewriting. A fixed set of
  **4 personas** over **4 ordinal axes** (Bad / Average / Good / Excellent), stored as the
  ordinal record but rendered to natural language for the prompt; **3 personas train, 1 is
  held out for test only** (a generalization claim to an unseen profile).
- **Evaluation hardened.** Baseline ladder (naive RAG → persona-prompted + untrained
  rewriter → persona-prompted + DPO rewriter); **paired** significance tests + **bootstrap
  CIs** over **≥3 seeds**; the test set, judge prompt, and seeds are **frozen and versioned
  from day one**.

### v1.2 — 2026-07-03 — retriever promoted to core; 5-rung baseline ladder

Supersedes v1.1's decision to freeze the retriever. Two trained components are now in
scope:

- **Retriever trained with ROPG-KD (core, not stretch).** The retriever is no longer
  a frozen backdrop. It is fine-tuned from BGE-M3 using offline knowledge distillation
  (ROPG-KD): an LLM judge scores each `(query, persona, document)` triple for
  pedagogical utility, and those scores are distilled into the encoder via KL divergence.
  This is the offline variant of ROPG-RL (Salemi et al., 2024) — same personalization
  objective, no online reward loop, compatible with the DPO-only compute constraint.
- **Baseline ladder expanded to 5 rungs** (was 3):
  - Rung 0: BM25, no persona (done)
  - Rung 1: BGE-M3 frozen, no persona
  - Rung 2: BGE-M3 frozen, persona-prompted untrained rewriter
  - Rung 3: BGE-M3 + ROPG-KD, untrained rewriter
  - Rung 4: BGE-M3 + ROPG-KD, DPO rewriter (full system)
  Rungs 3 vs 4 isolate the rewriter's marginal contribution on top of a trained retriever.
- **DPO rewriter policy model decided: Gemma-4-E4B + LoRA.** Qwen2.5-3B is the
  fallback if Persian output quality is insufficient (validated by smoke-testing rewrites
  before committing to full DPO training).
- **Training order:** ROPG-KD retriever first (retriever fixed), then DPO rewriter on
  top — ensures preference pairs for DPO are built against a stable retriever.
