# Personalization of Large Language Models: A Survey

- **Authors:** Zhehao Zhang, Ryan A. Rossi, Branislav Kveton, Yijia Shao, Diyi Yang, Hamed Zamani, Franck Dernoncourt, Joe Barrow, Tong Yu, Sungchul Kim, Ruiyi Zhang, Jiuxiang Gu, Tyler Derr, Hongjie Chen, Junda Wu, Xiang Chen, Zichao Wang, Subrata Mitra, Nedim Lipka, Nesreen Ahmed, Yu Wang
- **Year:** 2024 (rev. 2025)
- **Venue:** Transactions on Machine Learning Research (TMLR) (arXiv:2411.00027)
- **Link:** https://arxiv.org/abs/2411.00027

---

## Summary

A broad survey of LLM personalization that organizes the field into a taxonomy spanning
personalization *granularity*, the *signals* used (user history, profiles, preferences), the
*techniques* (retrieval of user context, lightweight adapters, fine-tuning to user-specific
objectives), and *evaluation*. It provides the general map for locating Simurgh's specific choices.

## Key Contributions

- A unified **taxonomy** of personalization signals, techniques, and evaluation for LLMs.
- A consolidated view of **personalized RAG** (retrieving user context at inference) alongside parametric approaches (adapters / fine-tuning).
- A survey of **benchmarks and metrics**, surfacing the field's evaluation weaknesses.

## Relevance to This Thesis

This is Simurgh's **general positioning anchor**. It situates the project as **profile-based**
personalization (a *declared* learner profile) rather than the more common **history-based**
personalization (mining a user's past behavior) — the same distinction drawn against
[optimization-methods-for-personalizing-large-language-models-through-retrieval-augmentation](optimization-methods-for-personalizing-large-language-models-through-retrieval-augmentation.md) and
[personalize-before-retrieve-llm-based-personalized-query-expansion-for-user-centric-retrieval](personalize-before-retrieve-llm-based-personalized-query-expansion-for-user-centric-retrieval.md).
Notably, co-author **Hamed Zamani** also authored the ROPG retriever-personalization paper already
in the library, so this survey is the broader frame around Simurgh's nearest prior work. Its
evaluation section reinforces Simurgh's insistence on a held-out test set, judge independence, and
significance testing ([experiment-design](../experiment-design.md)).

## Notes

- The taxonomy helps phrase Simurgh's novelty precisely: *profile-conditioned* personalization of a *query rewriter* for *Persian educational* RAG — a cell the survey's map leaves largely empty.
- Use it for the literature-review framing paragraph, not for a specific method; the concrete recipes live in the focused references.
