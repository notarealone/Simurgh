# BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation

- **Authors:** Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, Zheng Liu
- **Year:** 2024
- **Venue:** arXiv 2024 (ACL Findings 2024)
- **Link:** https://arxiv.org/abs/2402.03216

---

## Summary

BGE M3-Embedding is a text embedding model covering more than 100 languages that unifies three retrieval functions — dense, sparse (lexical), and multi-vector (ColBERT-style) — in a single encoder. Training uses self-knowledge distillation: relevance scores produced by each retrieval head are blended into a unified teacher signal that supervises the other heads, so the three functions reinforce one another without requiring separate models. The authors also introduce a large-batch training strategy to improve embedding discriminativeness across long documents up to 8,192 tokens.

## Key Contributions

- Single model supporting dense, sparse, and multi-vector retrieval simultaneously across 100+ languages.
- Self-knowledge distillation: uses each retrieval method's own scores as soft labels to train the others, avoiding the need for a separate teacher model.
- Effective handling of long documents (up to 8,192 tokens), extending beyond typical 512-token limits.
- Optimized batching strategy that improves in-batch negative diversity for better discriminative training.
- State-of-the-art results on multilingual, cross-lingual, and long-document retrieval benchmarks (MIRACL, BEIR, MLDR).

## Relevance to This Thesis

BGE M3-Embedding is the primary candidate retriever backbone for Simurgh: its multilingual training covers Persian, making it directly applicable without language-specific fine-tuning from scratch. The three retrieval modes (dense + sparse + multi-vector) can be combined via hybrid scoring, which is useful when persona-conditioned re-ranking is layered on top. Persona embedding experiments (injecting learner profile signals into the query representation) can be prototyped as lightweight adapters over the frozen M3 encoder, keeping compute within Kaggle/LoRA constraints.

## Notes

- Persian is included in the 100+ language coverage but benchmark numbers are reported mainly for MIRACL languages; Persian-specific retrieval quality should be validated on a held-out Persian QA set before committing to this backbone.
- The multi-vector (ColBERT) mode is more expressive but significantly more expensive at inference time — the simpler dense mode is the practical default under minimal-compute constraints.
- Self-knowledge distillation makes training self-contained, but the released checkpoint weights are already fine-tuned; for this thesis the pretrained checkpoint is used directly with PEFT rather than re-trained.
- Open question: how well does the sparse head handle Persian morphology and ZWNJ tokenization artifacts? May need normalization pre-processing before indexing.
