# Dense Passage Retrieval for Open-Domain Question Answering

- **Authors:** Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih
- **Year:** 2020
- **Venue:** EMNLP 2020
- **Link:** https://arxiv.org/abs/2004.04906

---

## Summary

Karpukhin et al. show that dense retrieval via a lightweight dual-encoder framework — two independent BERT encoders, one for questions and one for passages, trained with in-batch negatives — substantially outperforms BM25 for open-domain question answering. The dual-encoder is trained on a small number of (question, positive passage, negative passage) triples and produces embeddings indexed with FAISS for fast approximate nearest-neighbour search. The resulting retriever improves top-20 passage accuracy by 9–19 absolute points over BM25 across several QA datasets.

## Key Contributions

- Demonstrates that a simple dual-encoder trained on QA pairs beats BM25 without requiring any extra pre-training signals beyond standard supervised contrastive learning.
- Introduces the in-batch negative training strategy, which scales efficiently and avoids the need for explicit hard negative mining in the basic setup.
- Provides the retriever backbone that was directly adopted in the RAG framework ([[lewis-2020-rag]]).
- Shows that the gap between dense and sparse retrieval is largest when queries are paraphrases or when lexical overlap between question and answer passage is low — motivating learned query representations.
- End-to-end QA using DPR + a reader achieves strong results on Natural Questions, TriviaQA, WebQuestions, and CuratedTREC.

## Relevance to This Thesis

DPR is the default retriever inside the RAG pipeline Simurgh extends, so understanding its dual-encoder training is prerequisite knowledge for the personalized retriever component. The thesis will explore whether persona-conditioned query embeddings (or reward-based fine-tuning of the question encoder) can steer retrieval toward passages appropriate for a given learner profile; DPR's training objective and FAISS indexing are the starting point for that work. The in-batch negative strategy is also directly applicable when constructing persona-conditioned contrastive fine-tuning data.

## Notes

- DPR encodes questions and passages independently, which enables fast FAISS lookup but prevents cross-attention between query and passage at retrieval time — a known expressiveness trade-off.
- The original model is English-only (BERT-base); for Persian text a multilingual encoder (e.g., mBERT, XLM-R) or a Persian-specific encoder must be substituted and ideally fine-tuned on Persian QA pairs.
- Low-compute concern: fine-tuning a dual-encoder on synthetic Persian preference data should fit on a single GPU with PEFT/LoRA on the question encoder only; passage encoder can be frozen.
- Open question: does personalizing the *question* encoder alone suffice, or does the passage encoder also need persona-aware fine-tuning?
