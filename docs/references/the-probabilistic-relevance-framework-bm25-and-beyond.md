# The Probabilistic Relevance Framework: BM25 and Beyond

- **Authors:** Stephen Robertson, Hugo Zaragoza
- **Year:** 2009
- **Venue:** Foundations and Trends in Information Retrieval, 3(4), 333–389
- **Link:** https://doi.org/10.1561/1500000019

---

## Summary

Robertson and Zaragoza present the probabilistic relevance framework and derive BM25 as a practical term-weighting and document-ranking function. The model combines inverse document frequency, saturating term frequency, and document-length normalization.

## Key Contributions

- Connects BM25 to the probability ranking principle and the probabilistic relevance framework.
- Explains the assumptions that reduce document ranking to a sum of query-term contributions.
- Derives the roles of term-frequency saturation and document-length normalization.
- Reviews extensions including relevance feedback, query expansion, BM25F, and non-textual features.

## Relevance to This Thesis

BM25 is the lexical retrieval baseline for the RAG pipeline. It provides a cheap, transparent comparison point for dense retrieval and makes Persian normalization effects directly observable because its ranking depends on token overlap.

## Notes

BM25 cannot match semantically related expressions that share no indexed terms. Its scores also encode query-document relevance rather than learner-conditioned utility, so adding a profile to the query does not by itself establish personalization.
