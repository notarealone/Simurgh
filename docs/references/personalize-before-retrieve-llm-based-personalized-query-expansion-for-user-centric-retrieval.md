# Personalize Before Retrieve: LLM-based Personalized Query Expansion for User-Centric Retrieval

- **Authors:** Yingyi Zhang, Pengyue Jia, Derong Xu, Yi Wen, Xianneng Li, Yichao Wang, Wenlin Zhang, Xiaopeng Li, Weinan Gan, Huifeng Guo, Yong Liu, Xiangyu Zhao
- **Year:** 2025
- **Venue:** arXiv:2510.08935 (cs.IR)
- **Link:** https://arxiv.org/abs/2510.08935

---

## Summary

This paper argues that standard query expansion applies a *uniform* strategy to every user, ignoring individual expression style, preferences, and history — which limits retrieval for user-centric tasks. It proposes **PBR (Personalize Before Retrieve)**, which personalizes the query *before* it ever reaches the retriever via two components: **P-PRF**, which generates stylistically aligned pseudo-relevance-feedback terms by conditioning on the user's history to mimic how that user phrases things; and **P-Anchor**, which uses graph-based structural alignment over the user's corpus to anchor the expanded query in the user's semantic space. PBR reports up to ~10% retrieval gains on personalized benchmarks (e.g. PersonaBench) and works across multiple retrievers.

## Key Contributions

- **Formalizes the personalization gap in query expansion**: uniform expansion erases user-specific semantics (style, preferences, context).
- **P-PRF**: user-history-conditioned pseudo-feedback generation that preserves the individual's expression style during expansion.
- **P-Anchor**: graph-based structural alignment that grounds the personalized query in the heterogeneous structure of the user's own corpus.
- **Retriever-agnostic gains**: consistent improvements (up to ~10% on PersonaBench) across diverse retrieval architectures, since personalization happens before retrieval.

## Relevance to This Thesis

PBR is a direct, recent analogue to Simurgh's **persona-conditioned Query Rewriter** ([query-rewriting-for-retrieval-augmented-large-language-models](query-rewriting-for-retrieval-augmented-large-language-models.md)): both intervene *before* retrieval and condition the query transformation on the user. The "personalize before retrieve" placement validates Simurgh's architectural choice to put personalization in the rewriter, upstream of a general-purpose Persian retriever, rather than baking it into the embeddings. P-PRF's idea of preserving the *user's expression style* maps onto adapting a query to a learner's level (e.g. simplifying vocabulary for a struggling ninth-grader vs. adding challenge framing for an advanced student). It also offers a profile-free ablation by design: turn off P-PRF/P-Anchor and the pipeline reverts to generic expansion — exactly the "switchable back to no profile" contract Simurgh requires of every personalized component.

## Notes

- Personalization signal here is *user history / personal corpus*; Simurgh's is a *declared learner profile* over a shared textbook index. P-PRF's history-conditioning would need re-grounding on the profile schema (grade / learning style / goal) rather than past queries.
- PBR is training-free (prompting + graph alignment), whereas Simurgh's rewriter is trained with DPO. PBR is therefore a strong **persona-prompting baseline rung** (ladder step 2) to beat with the trained rewriter — useful for isolating whether DPO training actually adds value over clever prompting.
- P-Anchor's graph alignment over a per-user corpus may not transfer cleanly to a single shared Persian corpus; the more portable idea is P-PRF's style-conditioned expansion.
- Evaluated in English on PersonaBench; Persian-specific expansion must handle ZWNJ / ye-ke normalization so expanded terms actually match the index (cf. [methodology](../methodology.md)).
