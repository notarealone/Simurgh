# Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models

- **Authors:** Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, Jingren Zhou
- **Year:** 2025
- **Venue:** arXiv preprint
- **Link:** https://arxiv.org/abs/2506.05176

---

## Summary

Zhang et al. introduce the Qwen3 Embedding family for text embedding and reranking, built from Qwen3 foundation models. Its multi-stage training combines large-scale unsupervised pretraining, supervised fine-tuning on multilingual data, model-generated training data, and model merging.

## Key Contributions

- Provides embedding and reranking models at 0.6B, 4B, and 8B parameter scales.
- Supports multilingual, cross-lingual, general text, and code retrieval tasks.
- Uses task instructions on the query side and supports flexible output dimensions through Matryoshka representation learning.
- Releases the model family under the Apache 2.0 license.

## Relevance to This Thesis

Qwen3-Embedding-0.6B is the concrete dense-retrieval backbone used by Simurgh. The 0.6B variant fits the project's compute constraint while providing multilingual support, instruction-aware query encoding, a context length of 32K tokens, and embeddings of up to 1024 dimensions.

## Notes

The paper reports aggregate multilingual benchmark results, not Persian educational retrieval results. Its broad pretrained capability motivates the starting checkpoint but cannot substitute for in-domain retrieval evaluation. Fine-tuning can also change its general ranking behavior without making that behavior profile-dependent, so the persona-swap test remains necessary.
