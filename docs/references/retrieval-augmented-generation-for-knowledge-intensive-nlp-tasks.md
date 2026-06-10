# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

- **Authors:** Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, et al.
- **Year:** 2020
- **Venue:** NeurIPS 2020
- **Link:** https://arxiv.org/abs/2005.11401

---

## Summary

Lewis et al. introduce Retrieval-Augmented Generation (RAG), a general framework that couples a pre-trained sequence-to-sequence generator with a dense neural retriever over a Wikipedia index. At inference time the retriever fetches relevant passages and the generator conditions on them, combining parametric knowledge (weights) with non-parametric memory (the index). The model achieves state-of-the-art results on several open-domain QA and knowledge-intensive benchmarks while producing more specific and factually grounded text than parametric-only baselines.

## Key Contributions

- Defines the RAG architecture: DPR retriever + BART seq2seq generator, end-to-end trainable via marginalization over retrieved documents.
- Proposes two variants: RAG-Sequence (one retrieval per answer) and RAG-Token (retrieval can vary per token), giving flexibility in how retrieved passages are fused.
- Shows that non-parametric memory is more easily updated than model weights, enabling knowledge updates without full retraining.
- Demonstrates stronger factual grounding and output diversity compared to purely parametric generation on open-domain QA, fact verification, and generation tasks.
- Establishes a practical recipe for combining off-the-shelf dense retrieval with generation, which became the de facto RAG baseline in subsequent work.

## Relevance to This Thesis

This paper defines the RAG architecture that Simurgh is built on; every component in the project (retriever, generator, query rewriter) maps directly onto the RAG-Sequence/RAG-Token pipeline introduced here. It serves as the foundational baseline rung ("naive RAG") in the baseline ladder and provides the retriever–generator interaction that personalization layers will wrap. Understanding how the generator marginalizes over passages informs how persona-conditioned rewriting will change what is retrieved and ultimately generated.

## Notes

- The original RAG system is not personalized — it has no user-profile conditioning, making it the natural starting point for ablations.
- Retrieval is over English Wikipedia; applying the same pattern to a Persian educational corpus requires replacing the index and verifying that the DPR-style encoder transfers to Persian text (or fine-tuning a multilingual/Persian encoder).
- The seq2seq backbone (BART) is large; for low-compute settings consider smaller alternatives (e.g., mT5-small) with LoRA fine-tuning.
- Open question: does RAG-Token vs. RAG-Sequence matter when the generator is a ≤7B causal LM rather than BART?
