# Literature Review

> A comprehensive survey of related work. One reference note per paper lives in
> [`references/`](references/INDEX.md); this file synthesizes them into an argument.

---

## RAG Foundations

Retrieval-Augmented Generation pairs a parametric generator with a non-parametric
index so the model can ground its output in retrieved evidence instead of relying on
memorized weights alone. The pattern is the backbone of Simurgh, and the foundational
work below defines the "naive RAG" rung of our [baseline ladder](things-to-consider.md).

- **The original RAG framework.** Lewis et al. (2020) couple a DPR retriever with a
  BART generator that marginalizes over retrieved passages, introducing the
  RAG-Sequence and RAG-Token variants and showing that non-parametric memory is easier
  to update than weights. This is the architecture every Simurgh component wraps.
  → [`retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks.md`](references/retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks.md)
- **Dense retrieval.** Karpukhin et al. (2020) introduce Dense Passage Retrieval (DPR),
  a dual-encoder trained with contrastive learning that was the first fully neural
  retriever to beat BM25 on open-domain QA. DPR is the conceptual starting point for the
  persona-conditioned query encoder we want to fine-tune.
  → [`dense-passage-retrieval-for-open-domain-question-answering.md`](references/dense-passage-retrieval-for-open-domain-question-answering.md)
- **Retrieval-augmented pre-training.** Guu et al. (2020, REALM) treat retrieval as a
  latent variable trained end-to-end during pre-training, with an asynchronously
  refreshed index. The differentiable-retrieval framing informs how a reward signal
  could shape the retriever rather than only the generator.
  → [`realm-retrieval-augmented-language-model-pre-training.md`](references/realm-retrieval-augmented-language-model-pre-training.md)
- **Fusing many passages.** Izacard & Grave (2021, Fusion-in-Decoder) encode retrieved
  passages independently and fuse them in the decoder, scaling answer quality with the
  number of passages. This motivates investing in *retrieval quality* — better passages
  translate directly into better generations.
  → [`leveraging-passage-retrieval-with-generative-models-for-open-domain-question-answering.md`](references/leveraging-passage-retrieval-with-generative-models-for-open-domain-question-answering.md)
- **A map of the field.** Gao et al. (2024) survey RAG for LLMs and give the
  Naive → Advanced → Modular RAG taxonomy that scaffolds our baseline ladder and frames
  query rewriting / retriever tuning as "Advanced RAG" modules.
  → [`retrieval-augmented-generation-for-large-language-models-a-survey.md`](references/retrieval-augmented-generation-for-large-language-models-a-survey.md)

**Limitations that motivate this thesis.** None of the foundational systems condition
on *who is asking*. Retrieval and generation are identical for a struggling ninth-grader
and an advanced student. Simurgh's premise is that for educational text the same correct
answer can serve learners very differently, so the pipeline must be made
profile-aware — which the foundations leave entirely open.

### Retriever & query-rewriter building blocks

Two component-level papers seed the parts of the pipeline Simurgh actually modifies:

- **Multilingual retriever.** Chen et al. (2024, BGE-M3) provide a single model doing
  dense, sparse, and multi-vector retrieval across 100+ languages up to 8k tokens — the
  leading candidate backbone for a Persian index. Persian-specific quality still needs
  verifying (MIRACL is multilingual but not Persian-focused), and the multi-vector mode
  may be too heavy for Kaggle. → [`bge-m3-embedding-multi-lingual-multi-functionality-multi-granularity-text-embeddings-through-self-knowledge-distillation.md`](references/bge-m3-embedding-multi-lingual-multi-functionality-multi-granularity-text-embeddings-through-self-knowledge-distillation.md)
- **Query rewriting.** Ma et al. (2023) propose Rewrite-Retrieve-Read, a small trainable
  rewriter optimized with RL from reader feedback — the direct architectural precedent
  for Simurgh's Query Rewriter. We extend it by conditioning rewrites on a learner
  profile and swapping online RL for offline DPO over LLM-judge preference pairs.
  → [`query-rewriting-for-retrieval-augmented-large-language-models.md`](references/query-rewriting-for-retrieval-augmented-large-language-models.md)

- [x] Summarize original RAG paper (Lewis et al., 2020) and key follow-ups
- [x] Cover standard pipeline components: retriever, generator, how they connect
- [x] Identify limitations of vanilla RAG that motivate personalization

## RL for Information Retrieval

<!-- RL applied to retrieval, ranking, and query reformulation. -->
Seeded by Ma et al. (2023), whose rewriter is trained with RL on reader feedback. Next
batch should add RL-based retriever/reranker optimization and learning-to-rank.

- [ ] Survey RL-based retrieval optimization methods
- [ ] Cover query reformulation with RL (have: Ma et al. 2023 — extend with more)
- [ ] Cover learning-to-rank approaches relevant to retrieval

## RL for Text Generation

<!-- RLHF, DPO, PPO-based optimization of LLM outputs. -->
Not yet covered. Priority for the next batch — the core training method (DPO) lives here.

- [ ] Summarize RLHF and PPO-based training for LLMs (Ouyang et al. 2022; Schulman et al. 2017)
- [ ] Cover DPO and why it's preferred for some use cases (Rafailov et al. 2023)
- [ ] Review preference pair generation strategies (LLM-as-judge, RLAIF; Zheng et al. 2023)

## Personalized Retrieval and Generation

<!-- User-aware retrieval, preference-tuned generation, personalization in LLMs. -->
Not yet covered. Next batch — the differentiating contribution of the thesis.

- [ ] Survey user-aware retrieval methods (profile-conditioned, persona embeddings)
- [ ] Cover preference-tuned generation approaches (e.g., LaMP, personalized DPO)
- [ ] Review personalization in educational / tutoring systems

## Adaptive / Self-Reflective RAG

These papers make RAG *decide how to retrieve* based on the input — a mechanism that
transfers cleanly to deciding how to retrieve based on the *learner*.

- **Self-RAG** (Asai et al., 2024, ICLR) trains one LM to retrieve on demand and critique
  its own output via reflection tokens. The teacher-generated critique recipe is a model
  for how Simurgh's LLM-as-simulator can generate persona-fit preference pairs.
  → [`self-rag-learning-to-retrieve-generate-and-critique-through-self-reflection.md`](references/self-rag-learning-to-retrieve-generate-and-critique-through-self-reflection.md)
- **Adaptive-RAG** (Jeong et al., 2024, NAACL) routes queries to no/single/multi-step
  retrieval by predicted complexity. Swap "complexity" for "persona" and it becomes a
  per-learner retrieval-strategy selector — a direct analogue for Simurgh.
  → [`adaptive-rag-learning-to-adapt-retrieval-augmented-large-language-models-through-question-complexity.md`](references/adaptive-rag-learning-to-adapt-retrieval-augmented-large-language-models-through-question-complexity.md)
- **CRAG** (Yan et al., 2024) adds a lightweight retrieval evaluator that triggers
  correction/filtering when evidence is weak. The evaluator pattern suggests a
  persona-aware retrieval-quality gate. → [`corrective-retrieval-augmented-generation.md`](references/corrective-retrieval-augmented-generation.md)

- [x] Summarize Self-RAG, Adaptive-RAG, CRAG (FLARE still TODO)
- [x] Identify which ideas transfer to a personalization setting
- [ ] Add FLARE (Jiang et al., 2023) for active/forward-looking retrieval

## Open Questions & Gap Analysis

<!-- What's missing in existing work that this thesis addresses? -->
Provisional gap (firms up as RL/personalization sections fill in): foundational RAG is
user-agnostic; adaptive RAG conditions on *question* properties, not *user* properties;
query rewriting is trained for generic retrieval utility, not for a learner's pedagogical
needs. Simurgh's contribution is to make the rewriter and retriever **persona-conditioned**
and to optimize them for **pedagogical quality** (LLM-judge + human-validated) rather than
answer-string correctness — in **Persian**, a low-resource educational setting absent from
all the work above.

- [x] Draft initial gap statement from foundations (refine after RL + personalization batches)
- [ ] Synthesize gaps across all sections once RL and personalization sections are written
- [ ] State clearly what this thesis contributes that prior work does not
