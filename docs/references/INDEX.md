# References Index

> One file per paper. Copy `TEMPLATE.md` to start a new entry.

| Paper | Year | Topic | Relevance |
|---|---|---|---|
| [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](references/lewis-2020-rag.md) | 2020 | RAG architecture | Foundational RAG pipeline (naive-RAG baseline rung) that all Simurgh components wrap |
| [Dense Passage Retrieval for Open-Domain Question Answering](references/karpukhin-2020-dpr.md) | 2020 | Dense retrieval / dual-encoder | Default retriever inside the RAG pipeline; starting point for persona-conditioned query-encoder fine-tuning |
| [REALM: Retrieval-Augmented Language Model Pre-Training](references/guu-2020-realm.md) | 2020 | Retrieval-augmented pre-training | Differentiable latent-variable retrieval framing and async index refresh — conceptual basis for reward-based retriever training |
| [Leveraging Passage Retrieval with Generative Models for Open Domain QA](references/izacard-2021-fid.md) | 2021 | Fusion-in-Decoder / generative reader | Multi-passage fusion architecture for the generator; motivates investing in personalized retrieval quality |
| [BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation](references/chen-2024-bge-m3.md) | 2024 | Multilingual retrieval / embeddings | Primary retriever backbone for Persian RAG; supports dense+sparse+multi-vector in one model |
| [Query Rewriting for Retrieval-Augmented Large Language Models](references/ma-2023-query-rewriting.md) | 2023 | Query rewriting / RAG | Direct architectural precedent for Simurgh's persona-conditioned Query Rewriter trained with DPO |
| [Retrieval-Augmented Generation for LLMs: A Survey](references/gao-2024-rag-survey.md) | 2024 | RAG survey | Naive/Advanced/Modular taxonomy scaffolds the baseline ladder and evaluation plan |
| [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](references/asai-2024-self-rag.md) | 2024 | Adaptive retrieval / self-critique | Teacher-generated critique recipe and inference-time control inform DPO preference-pair generation |
| [Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity](references/jeong-2024-adaptive-rag.md) | 2024 | Query routing / adaptivity | Complexity-based routing is a direct analogue for persona-based retrieval strategy selection |
| [Corrective Retrieval Augmented Generation (CRAG)](references/yan-2024-crag.md) | 2024 | Retrieval correction | Retrieval evaluator and noise-filtering algorithm inform a persona-aware retrieval-quality gate |
