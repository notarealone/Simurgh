# LoRA: Low-Rank Adaptation of Large Language Models

- **Authors:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
- **Year:** 2022
- **Venue:** ICLR 2022 (arXiv:2106.09685)
- **Link:** https://arxiv.org/abs/2106.09685

---

## Summary

LoRA freezes the pretrained weights and learns low-rank update matrices in selected Transformer layers. This reduces trainable parameters and optimizer state while keeping the base model unchanged.

## Key Contributions

- Replaces a full weight update with a low-rank factorization.
- Reduces trainable parameters and training memory.
- Adds no inference latency after the update matrices are merged into the base weights.

## Relevance to This Thesis

Both trainable components use LoRA adapters: Qwen3-Embedding-0.6B for retrieval and Qwen3-4B for query rewriting. Separate adapters also preserve a clean frozen reference for DPO.

## Notes

LoRA reduces adaptation cost, but it does not by itself reduce the memory required to store the frozen base weights.
