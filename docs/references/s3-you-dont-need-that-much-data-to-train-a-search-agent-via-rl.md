# s3: You Don't Need That Much Data to Train a Search Agent via RL

- **Authors:** Pengcheng Jiang, Xueqiang Xu, Jiacheng Lin, Jinfeng Xiao, Zifeng Wang, Jimeng Sun, Jiawei Han
- **Year:** 2025
- **Venue:** EMNLP 2025 (arXiv:2505.14146)
- **Link:** https://arxiv.org/abs/2505.14146

---

## Summary

s3 is a lightweight, model-agnostic RAG framework that **freezes the generator and trains only the searcher with RL**. It targets two flaws in prior RL-for-RAG work: optimizing search-only metrics (NDCG) that ignore downstream answer quality, and end-to-end fine-tuning that entangles reasoning and retrieval and breaks compatibility with frozen/proprietary LLMs. The reward is **Gain Beyond RAG (GBR)** — how much the searcher improves generation accuracy *over a naive-RAG baseline*. Trained on just 2.4k examples (≈70× less data than comparable methods), s3 beats stronger-data baselines across six general-QA and five medical-QA benchmarks.

## Key Contributions

- **Decouples search from generation**: only the searcher is trained; the generator stays frozen, so any (even proprietary) LLM can be the reader.
- **Gain Beyond RAG (GBR) reward**: rewards retrieval *for its downstream answer-quality lift over naive RAG*, directly tying the searcher's objective to end-task utility instead of intrinsic ranking metrics.
- **Extreme data efficiency**: ~2.4k training samples vs ~70× more for baselines, while still improving downstream QA.
- **Strong generalization**: consistent gains across general and medical QA without retraining the generator.

## Relevance to This Thesis

s3 is arguably the best-fit *training recipe* for Simurgh's retriever-optimization rung under a 4-week, low-compute, solo budget. Three properties line up with the project constraints. (1) **Frozen generator, train only the searcher** matches Simurgh's setup, where the generator may be an API model (allowed as judge/reader, not as the trainable policy) — s3 shows you can still optimize retrieval around it. (2) **GBR is the personalized-reward pattern Simurgh needs**: redefine "gain" as the LLM-judge's *persona-alignment / pedagogical* score relative to naive RAG, and GBR becomes a persona-conditioned retriever reward — the per-learner analogue of "gain beyond generic RAG." (3) **2.4k-sample efficiency** makes RL feasible on synthetic persona data without a large preference corpus. It pairs naturally with [[optimization-methods-for-personalizing-large-language-models-through-retrieval-augmentation]] (what to optimize: personalized retrieval) and [[deepretrieval-hacking-real-search-engines-and-retrievers-with-large-language-models-via-reinforcement-learning]] (how: small-model RL on retrieval outcome).

## Relevance caveat / scope

s3 trains an *agentic, multi-turn* searcher (query → search → refine). For a 4-week thesis, the transferable core is the **GBR reward and the frozen-generator decoupling**, not necessarily the full multi-step search loop — a single-shot persona-conditioned retriever trained with a GBR-style reward is the realistic first target. Treat the agentic loop as future work (cf. scope notes in [[methodology]]).

## Notes

- GBR measured against a *naive-RAG* baseline fits Simurgh's baseline-ladder discipline perfectly: the reward literally encodes "better than the rung below."
- Reward needs a reference naive-RAG answer per query to compute the gain — doubles generation cost at training time; cheap with a cached baseline, but budget it.
- Evaluated on English general/medical QA; no educational-personalization or Persian evaluation. The *reward design* transfers; results do not.
- Same lead author as [[deepretrieval-hacking-real-search-engines-and-retrievers-with-large-language-models-via-reinforcement-learning]] — read the two together for the "train the search side, freeze the reader" design pattern.
