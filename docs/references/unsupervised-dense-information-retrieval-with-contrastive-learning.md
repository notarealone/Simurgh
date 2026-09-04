# Unsupervised Dense Information Retrieval with Contrastive Learning

- **Authors:** Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, Edouard Grave
- **Year:** 2022
- **Venue:** Transactions on Machine Learning Research
- **Link:** https://openreview.net/forum?id=jKN1pXi7b0

---

## Summary

Izacard et al. introduce Contriever, a dense retriever pretrained without relevance labels through contrastive learning. Random cropping creates paired views of a document, while in-batch negatives teach the encoder to organize texts in a shared representation space.

## Key Contributions

- Develops an unsupervised contrastive pretraining procedure for dense retrieval.
- Evaluates zero-shot, few-shot, supervised, multilingual, and cross-lingual retrieval settings.
- Reports higher Recall@100 than BM25 on 11 of 15 BEIR datasets in its unsupervised evaluation.
- Shows that unsupervised pretraining can improve later in-domain or MS MARCO fine-tuning.

## Relevance to This Thesis

Contriever provides the main contrast to supervised DPR: dense retrieval can begin from general contrastive structure when task-specific relevance labels are scarce. It also appears in the retrieval pool of the ROPG/RSPG base work reviewed later in the chapter.

## Notes

Unsupervised semantic similarity is not the same as educational usefulness. The paper does not condition retrieval on a learner profile, and its cross-lingual results do not remove the need to evaluate Persian normalization and in-domain retrieval directly.
