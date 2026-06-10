# Retrieval-Augmented Generation for Large Language Models: A Survey

- **Authors:** Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang
- **Year:** 2024
- **Venue:** arXiv (cs.CL)
- **Link:** https://arxiv.org/abs/2312.10997

---

## Summary

Gao et al. provide a comprehensive survey of Retrieval-Augmented Generation systems for LLMs, organizing the landscape into three paradigms: Naive RAG (basic retrieve-then-read), Advanced RAG (pre- and post-retrieval optimizations such as query rewriting, re-ranking, and context compression), and Modular RAG (flexible, composable pipelines where modules such as search, memory, and fusion can be added or replaced). The survey analyzes each paradigm's core retrieval, generation, and augmentation techniques, reviews evaluation frameworks and benchmarks, and maps out open challenges and future directions.

## Key Contributions

- Introduces the **Naive / Advanced / Modular RAG taxonomy**, the most widely adopted framework for situating RAG variants relative to one another.
- Surveys query-transformation techniques (query rewriting, step-back prompting, HyDE) as part of Advanced RAG's pre-retrieval stage — directly relevant to personalized query rewriting.
- Reviews augmentation granularities (token, sentence, document) and fusion strategies (RAG-Sequence, RAG-Token, late interaction), clarifying design choices for the generator.
- Covers retrieval-quality improvements: hybrid sparse-dense retrieval, re-ranking, iterative / recursive retrieval, and context compression.
- Discusses evaluation metrics (EM/F1, faithfulness, answer relevance, context relevance) and benchmarks (RAGAS, RGB, TruLens), useful for structuring Simurgh's own evaluation plan.

## Relevance to This Thesis

The Naive / Advanced / Modular taxonomy provides the conceptual scaffold for Simurgh's baseline ladder: "naive RAG" maps to Naive RAG, "persona prompting" is an Advanced RAG pre-retrieval modification, and "DPO rewriter + full system" is a Modular RAG configuration. The survey's treatment of query rewriting as a first-class Advanced RAG technique grounds the design choice of making the query rewriter the primary personalization point. Its evaluation framework (faithfulness, context relevance, answer relevance) translates directly into the judge rubric for LLM-as-judge preference-pair generation.

## Notes

- The survey is intentionally broad; it does not address personalization or user-profile conditioning as a first-class concern — this is precisely the gap Simurgh fills.
- No Persian / multilingual considerations; the pipeline components (chunking, embedding, retrieval) will need explicit adaptation for Persian text (ZWNJ normalization, Persian-capable encoders such as ParsBERT or multilingual-e5).
- Modular RAG's plug-and-play framing justifies swapping out components during ablations without redesigning the whole system.
- Open question: which Advanced RAG pre-retrieval technique (query rewriting vs. HyDE vs. step-back) transfers best to Persian educational QA under the low-compute constraint?
