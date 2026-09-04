# Precise Zero-Shot Dense Retrieval without Relevance Labels

- **Authors:** Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan
- **Year:** 2023
- **Venue:** ACL 2023
- **Link:** https://aclanthology.org/2023.acl-long.99/

---

## Summary

Gao et al. propose Hypothetical Document Embeddings (HyDE) for zero-shot dense retrieval without relevance labels. An instruction-following language model generates a hypothetical document from the query, and an unsupervised dense encoder uses that document's embedding to retrieve real documents.

## Key Contributions

- Replaces direct query encoding with a generated hypothetical document as an intermediate representation.
- Combines an instruction-following generator with an unsupervised contrastive encoder such as Contriever.
- Uses dense embedding space to filter factual errors in the generated document rather than treating the hypothetical text as evidence.
- Evaluates the method across web search, question answering, fact verification, and multilingual retrieval tasks.

## Relevance to This Thesis

HyDE demonstrates that generated text can bridge a short query and document-style corpus passages before retrieval. It is therefore a useful alternative to direct query rewriting when explaining why a learned rewriter may change retrieval quality.

## Notes

The hypothetical document can hallucinate, and success depends on the encoder mapping its useful semantic pattern near real evidence. HyDE does not define how a learner profile should alter the generated representation or how to distinguish profile sensitivity from a generic expansion gain.
