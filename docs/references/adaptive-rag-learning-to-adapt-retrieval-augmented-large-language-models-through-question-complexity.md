# Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity

- **Authors:** Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, Jong C. Park
- **Year:** 2024
- **Venue:** NAACL 2024
- **Link:** https://arxiv.org/abs/2403.14403

---

## Summary

Adaptive-RAG trains a small classifier to predict the complexity of an incoming query and then routes it to the most appropriate retrieval strategy: no retrieval (for simple factoid queries the LLM can answer directly), single-step retrieval (standard RAG), or iterative multi-step retrieval (for complex multi-hop questions). The classifier is trained on automatically generated complexity labels derived from the predictions of existing retrieval-augmented and non-retrieval models, requiring no manual annotation. This routing mechanism improves both efficiency and accuracy on open-domain QA benchmarks compared to always applying the most expensive strategy.

## Key Contributions

- Proposes a **query-complexity classifier** that dynamically selects among no-retrieval, single-step retrieval, and iterative retrieval, framing adaptivity as a learned routing problem.
- Generates training labels automatically by comparing predictions from simpler and more complex models, avoiding costly human annotation of difficulty.
- Demonstrates that matching retrieval strategy to question complexity strictly dominates both always-retrieve and always-iterate baselines in accuracy and latency.
- Shows that a **smaller model** can serve as an effective router, keeping the added compute cost low — relevant to resource-constrained deployments.
- Validates on multiple open-domain QA datasets; code is publicly released.

## Relevance to This Thesis

Adaptive-RAG's routing-by-complexity framing is a direct analogue to routing-by-persona in Simurgh: just as Adaptive-RAG selects a retrieval strategy based on query difficulty, Simurgh selects (or rewrites) a query strategy based on the learner's profile (grade level, learning style). The automatic label-generation technique — inferring signal from model prediction discrepancies — is methodologically similar to Simurgh's LLM-as-simulator approach for generating preference pairs without human annotations. The lightweight classifier design also supports the low-compute constraint, suggesting that a small LoRA-adapted router could handle persona-conditioned strategy selection without a large dedicated model.

## Notes

- Complexity and persona-fit are orthogonal axes; Simurgh may need to condition jointly on both (a simple question still needs to be answered at the right level for a 7th-grader vs. a PhD student).
- The classifier is trained on English QA datasets; transfer to Persian educational content would require either Persian-language training data or a multilingual backbone.
- The paper focuses on retrieval strategy selection, not on *what* is retrieved or *how* the answer is generated — personalization in Simurgh extends further into the generation stage.
- Open question: does routing complexity correlate with routing persona difficulty, or are they independent — empirically testable as a cheap ablation in Simurgh's evaluation.
