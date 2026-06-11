# Optimization Methods for Personalizing Large Language Models through Retrieval Augmentation

- **Authors:** Alireza Salemi, Surya Kallumadi, Hamed Zamani
- **Year:** 2024
- **Venue:** SIGIR 2024 (arXiv:2404.05970)
- **Link:** https://arxiv.org/abs/2404.05970

---

## Summary

This is the closest prior work to Simurgh's core thesis: it is the first study to **optimize the retriever** that feeds personal documents to an LLM, rather than treating retrieval as a fixed, off-the-shelf component. Building on the RAG-style personalization pipeline from the LaMP benchmark, the authors introduce a family of methods (named **ROPG** for retriever optimization and **RSPG** for retriever selection) that train or select the retriever using feedback from the downstream personalized-generation task. They show that aligning retrieval with the personalization objective — not generic relevance — produces statistically significant gains on six of seven LaMP datasets.

## Key Contributions

- **First retriever optimization specifically for LLM personalization**, moving beyond frozen retrievers in personalized RAG.
- **ROPG — two optimization signals:** a *reinforcement-learning* variant (ROPG-RL) that uses the reward from the personalized-generation task to train the retriever with arbitrary, possibly non-differentiable metrics, and a *knowledge-distillation* variant (ROPG-KD) that distills the downstream LLM's preferences back into the retriever.
- **RSPG — retriever selection:** a pre- and post-generation model that picks the best retriever (or retrieval strategy) per input, rather than committing to one retriever globally.
- **Empirical validation on LaMP** (seven personalization tasks), with significant improvements on six of seven datasets over a strong RAG baseline.
- Establishes that **retrieval feedback for personalization should come from the personalized objective**, since generic relevance and personal usefulness diverge.

## Relevance to This Thesis

This paper is Simurgh's primary positioning anchor and the most direct precedent for the retriever-optimization rung of the [baseline ladder](../things-to-consider.md). The proposal's stated goal — "extend prior DPO-based work by adding RL optimization for the Retriever" — is essentially a Persian, educationally-framed extension of ROPG-RL. The reward-from-downstream-task framing maps onto Simurgh's LLM-as-judge reward (persona alignment / pedagogical quality) driving the retriever, and RSPG's per-input retriever selection is a close analogue to choosing a retrieval strategy *per learner profile* (cf. [[adaptive-rag-learning-to-adapt-retrieval-augmented-large-language-models-through-question-complexity]]). Where Simurgh differs: LaMP personalizes from a *user's own document history*, while Simurgh personalizes from a *declared learner profile* (grade, learning style, goal) against a shared Persian textbook index — the personalization signal is the profile, not a personal corpus.

## Notes

- LaMP tasks are English and personalize from per-user histories; Simurgh has no real user data and uses synthetic profiles over a shared corpus, so the *method* transfers but the *benchmark* does not — Simurgh must build its own persona-conditioned evaluation (cf. [[experiment-design]]).
- ROPG-RL uses online reward signals to train the retriever. Under Simurgh's compute budget, weigh this against the offline DPO path used for the rewriter; ROPG-KD (distillation) may be the cheaper retriever-side option. See open question "Which RL algorithm for retriever?" in [[things-to-consider]].
- The same authors' follow-ups (Stochastic RAG; LaMP-QA) extend this line — worth tracking for the retriever-optimization section of the [[literature-review]].
- Reward-hacking caveat applies symmetrically to the retriever: an RL-optimized retriever can learn to satisfy the judge's surface preferences instead of fetching pedagogically useful passages. Guard with retrieval metrics per persona (Recall@K / MRR), not judge score alone.