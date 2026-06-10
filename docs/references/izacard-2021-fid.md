# Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering

- **Authors:** Gautier Izacard, Edouard Grave
- **Year:** 2021
- **Venue:** EACL 2021
- **Link:** https://arxiv.org/abs/2007.01282

---

## Summary

Izacard and Grave propose Fusion-in-Decoder (FiD), a generative reader architecture that encodes each retrieved passage independently with a T5 encoder and then fuses all encoded representations in the decoder via cross-attention. This contrasts with prior approaches that re-rank or select a single passage before generation. FiD achieves top results on Natural Questions and TriviaQA, and crucially its performance improves monotonically as the number of retrieved passages grows — demonstrating that generative models can effectively aggregate evidence across dozens of passages without relying on billion-parameter models.

## Key Contributions

- Introduces the Fusion-in-Decoder architecture: separate encoding of each (question, passage) pair followed by joint decoding over all encoded representations.
- Shows a clear scaling relationship between the number of retrieved passages and answer accuracy, motivating retrieval of more candidates rather than aggressive re-ranking.
- Demonstrates competitive performance without the large model scale previously assumed necessary, pointing toward efficient retrieval-augmented generation.
- Establishes a strong generative reader baseline that outperforms extractive readers and single-passage generative models on standard open-domain QA benchmarks.
- Decouples retriever quality from reader capacity: a better retriever directly translates to a better answer, enabling modular system improvement.

## Relevance to This Thesis

FiD's architecture directly informs the generator component of Simurgh's RAG pipeline: encoding retrieved passages independently and fusing at decoding time is a practical pattern for a ≤7B generative model reading multiple Persian passages. The observed scaling with passage count motivates investing in retrieval quality (and thus personalized retrieval), since better persona-matched passages will translate into higher-quality personalized answers. The clean modular separation between retriever and reader also aligns with the thesis's ablation design, where the retriever is the primary variable being personalized.

## Notes

- FiD encodes passages independently, so the encoder does not see cross-passage interactions; this is a compute-friendly design well suited to low-resource settings with LoRA on the encoder.
- The original model uses T5; for Persian, switching to mT5 or a Persian-capable seq2seq model is the natural adaptation.
- A potential concern: when retrieving persona-conditioned passages, the fused decoder may average out persona signals across irrelevant passages included in the top-K — motivating careful choice of K and passage re-ranking by persona relevance.
- Open question: does persona conditioning in the decoder prefix (via a learned persona token) help FiD maintain persona-specific generation even when some retrieved passages are off-profile?
