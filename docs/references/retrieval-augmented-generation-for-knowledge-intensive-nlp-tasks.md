# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

- **Authors:** Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela
- **Year:** 2020
- **Venue:** NeurIPS 2020
- **Link:** https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html

---

## Summary

Lewis et al. define retrieval-augmented generation as a combination of parametric memory in a pretrained sequence-to-sequence model and non-parametric memory in a dense document index. Their two formulations condition generation either on one shared set of retrieved passages or on passages that may vary by generated token.

## Key Contributions

- Introduces a general-purpose RAG fine-tuning recipe for knowledge-intensive NLP tasks.
- Couples a neural retriever over a Wikipedia index with a pretrained sequence-to-sequence generator.
- Reports state-of-the-art results on three open-domain question-answering tasks in the evaluated setup.
- Shows how explicit retrieved evidence can improve provenance and make the external knowledge store updateable without changing all model parameters.

## Relevance to This Thesis

This paper provides the foundational RAG decomposition used throughout the thesis: retrieval selects external evidence before generation. Simurgh keeps that decomposition but asks a different question: when multiple passages are relevant to the same educational query, which passages are most useful for a particular learner profile?

## Notes

The paper studies task relevance and factual generation, not learner-conditioned utility. It therefore motivates the pipeline but does not establish that a standard RAG retriever personalizes educational evidence.