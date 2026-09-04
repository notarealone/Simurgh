# Dense Passage Retrieval for Open-Domain Question Answering

- **Authors:** Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih
- **Year:** 2020
- **Venue:** EMNLP 2020
- **Link:** https://aclanthology.org/2020.emnlp-main.550/

---

## Summary

Karpukhin et al. formulate passage retrieval with a dual encoder that maps questions and passages to dense vectors and ranks passages by vector similarity. The retriever is trained from question-passage examples and can search a large corpus through a precomputed passage index.

## Key Contributions

- Demonstrates that dense representations alone can support practical open-domain passage retrieval.
- Uses separate BERT encoders for questions and passages with an inner-product scoring function.
- Trains with positive passages and in-batch or BM25-derived negative passages.
- Reports 9–19 percentage-point absolute gains over a strong Lucene-BM25 system in top-20 passage retrieval accuracy on the evaluated open-domain QA datasets.

## Relevance to This Thesis

DPR supplies the supervised dual-encoder pattern used to explain dense retrieval: encode the query and corpus independently, precompute corpus embeddings, and train similarity scores with contrastive negatives. This pattern motivates the trainable retriever component in Simurgh.

## Notes

The reported gains are tied to English open-domain QA datasets with relevance supervision. They do not imply that a dense retriever transfers to Persian educational text or learns which passage is useful for a particular learner profile.
