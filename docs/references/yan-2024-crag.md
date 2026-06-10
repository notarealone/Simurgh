# Corrective Retrieval Augmented Generation

- **Authors:** Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling
- **Year:** 2024
- **Venue:** arXiv 2024
- **Link:** https://arxiv.org/abs/2401.15884

---

## Summary

CRAG (Corrective Retrieval Augmented Generation) addresses the brittleness of standard RAG when the static corpus returns low-quality or irrelevant documents. A lightweight **retrieval evaluator** scores the relevance of retrieved passages and triggers one of three actions: use the passages directly (high confidence), discard and fall back to a web search (low confidence), or combine both (ambiguous). A **decompose-then-recompose algorithm** then strips noise from whichever documents are used, extracting only the relevant knowledge strips before passing them to the generator. CRAG integrates as a plug-in corrective step on top of existing RAG pipelines and shows consistent improvements across short- and long-form generation tasks.

## Key Contributions

- Introduces a **retrieval evaluator** that assigns a confidence score to retrieved documents and triggers corrective actions (use / web-search / combine) accordingly, improving robustness to retrieval failures.
- Proposes a **decompose-then-recompose algorithm** that segments documents into fine-grained knowledge strips, scores each for relevance, and discards irrelevant strips — reducing context noise for the generator.
- Frames retrieval correction as a lightweight plug-in, enabling drop-in integration with any RAG system without end-to-end retraining.
- Demonstrates gains on four benchmarks spanning factoid QA, multi-hop QA, and long-form generation, including settings where the static corpus is noisy or incomplete.
- Shows that web-search fallback meaningfully compensates for static-corpus gaps, relevant when the knowledge base is incomplete.

## Relevance to This Thesis

CRAG's retrieval evaluator is a direct inspiration for adding a **retrieval-quality gate** in Simurgh: before passing retrieved passages to the generator, a lightweight scorer could check whether they actually match the learner's profile and the query, triggering a rewrite or re-retrieval if not. The decompose-then-recompose idea maps onto persona-conditioned context filtering — stripping out passages at the wrong difficulty level or wrong topic depth before generation. Both mechanisms are plug-in by design, which fits Simurgh's ablation strategy (each component must be switchable off independently).

## Notes

- The web-search fallback is not applicable in Simurgh's setting (closed Persian educational corpus, no live web access); the confidence-score-based gating and document filtering ideas transfer, but the fallback must be replaced by re-retrieval with a rewritten query.
- The retrieval evaluator in CRAG is trained on English data; adapting it for Persian requires either a multilingual evaluator or fine-tuning on Persian relevance labels.
- CRAG does not consider user profiles; the evaluator scores document-query relevance, not document-query-persona relevance. Extending the evaluator to condition on persona is a natural research contribution.
- Open question: how much does the decompose-then-recompose step help specifically in Persian text, where chunk boundaries are harder to define cleanly (ZWNJ, long nominal phrases)?
