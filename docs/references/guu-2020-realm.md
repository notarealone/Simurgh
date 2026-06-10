# REALM: Retrieval-Augmented Language Model Pre-Training

- **Authors:** Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, Ming-Wei Chang
- **Year:** 2020
- **Venue:** ICML 2020
- **Link:** https://arxiv.org/abs/2002.08909

---

## Summary

Guu et al. propose REALM, which integrates a latent knowledge retriever directly into the language model pre-training loop. Instead of storing all world knowledge in model parameters, REALM learns a retriever jointly with a masked-language-model objective by backpropagating through retrieval over millions of Wikipedia documents. At inference the model retrieves relevant passages and conditions on them, achieving 4–16 absolute accuracy gains over prior methods on open-domain QA while being more interpretable and modular.

## Key Contributions

- First system to jointly pre-train a retriever and a language model end-to-end using only an unsupervised (masked LM) signal — no labeled QA pairs needed at pre-training.
- Introduces an asynchronous MIPS index refresh strategy that keeps the document embeddings approximately up to date during training without prohibitive compute overhead.
- Demonstrates that retrieved passages serve as interpretable intermediate "reasoning steps," unlike knowledge stored implicitly in weights.
- Shows that retrieval-augmented pre-training transfers well to downstream open-domain QA fine-tuning, outperforming much larger parametric models.
- Frames retrieval as a latent variable with a tractable marginal likelihood, providing a principled probabilistic foundation for retrieval-augmented models.

## Relevance to This Thesis

REALM establishes the theoretical grounding for treating retrieval as a differentiable latent variable, which is the conceptual basis for learning a personalized retriever through gradient-based signals (including reward-based fine-tuning). The asynchronous index refresh strategy is directly relevant when Simurgh's personalized embeddings need to be updated without rebuilding the entire FAISS index on each training step. REALM's modular design — retriever and reader can be swapped independently — mirrors the ablation strategy the thesis requires (each component removed in turn).

## Notes

- REALM's joint pre-training requires substantial compute (TPU days); for this thesis only the architectural insight is relevant, not replication of pre-training.
- The retriever here is encoder-only and document-agnostic; personalizing it to a learner profile is not addressed and is a gap this thesis can fill.
- Persian language application would require a Persian or multilingual masked-LM backbone for the retriever encoder (e.g., ParsBERT or XLM-R).
- Open question: can a frozen REALM-style retriever be steered toward persona-appropriate passages by fine-tuning only a lightweight adapter on the query side?
