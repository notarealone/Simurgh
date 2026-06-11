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
Seeded by Ma et al. (2023), whose rewriter is trained with RL on reader feedback. The
2025 wave of RL-with-verifiable-rewards revives that idea with small models and
retrieval-grounded rewards — the most direct novelty injection for Simurgh's trainable
rungs.

- **RL query rewriting, no gold rewrites.** Jiang et al. (2025, DeepRetrieval, COLM)
  train a 3B policy with PPO to emit a `<think>` reasoning trace and then an `<answer>`
  query, rewarded purely by retrieval outcome (Recall@3K, NDCG@10, or SQL execution
  accuracy) plus a format term — no supervised rewrites. The 3B model more than doubles
  prior SOTA recall on literature search (65.1% vs 24.7%) and matches GPT-4o/Claude-3.5
  on evidence-seeking; the reasoning trace is load-bearing (recall falls to 51.9%
  without it). This is Ma et al. modernized and the closest precedent for Simurgh's
  rewriter — with the caveat that its reward is retrieval-only, not persona/pedagogy, so
  it trains the *retrieval-utility* half of the rewriter, not persona fit.
  → [`deepretrieval-hacking-real-search-engines-and-retrievers-with-large-language-models-via-reinforcement-learning.md`](references/deepretrieval-hacking-real-search-engines-and-retrievers-with-large-language-models-via-reinforcement-learning.md)
- **Train the searcher, freeze the generator.** Jiang et al. (2025, s3, EMNLP) train a
  7B search agent with PPO while leaving the generator frozen, rewarded by *Gain Beyond
  RAG* — the answer-accuracy lift of the agent's context over naive top-k retrieval
  (GBR = Acc(G(Q, D_s3)) − Acc(G(Q, D_RAG))). Filtering to questions naive RAG already
  fails, it reaches Search-R1-level accuracy with 2.4k samples (≈70× less data) and ~33×
  less compute. The frozen-generator decoupling and the "beat the rung below" reward map
  almost one-to-one onto Simurgh's constraints (API generator, synthetic data, baseline
  ladder).
  → [`s3-you-dont-need-that-much-data-to-train-a-search-agent-via-rl.md`](references/s3-you-dont-need-that-much-data-to-train-a-search-agent-via-rl.md)

- [x] Survey RL-based retrieval optimization methods (DeepRetrieval, s3; retriever-side
      ROPG under Personalized Retrieval below)
- [x] Cover query reformulation with RL (Ma et al. 2023; DeepRetrieval 2025)
- [ ] Cover learning-to-rank approaches relevant to retrieval

## RL for Text Generation

<!-- RLHF, DPO, PPO-based optimization of LLM outputs. -->
The foundational machinery (RLHF/PPO, DPO, RLAIF) is still to be summarized here — it is
the core training method for Simurgh's rewriter. This batch contributes one applied
exemplar: Dinucu-Jianu et al. (2025) align a 7B *generator* for tutoring with GRPO and a
decomposed LLM-judge reward (detailed under Personalized Retrieval and Generation). It
shows the end-to-end shape of RL-for-generation that Simurgh's preference loop mirrors —
synthetic interactions, a quality judge, a small policy — and a concrete reason to weigh
GRPO against DPO for the trainable rungs.

- [ ] Summarize RLHF and PPO-based training for LLMs (Ouyang et al. 2022; Schulman et al. 2017)
- [ ] Cover DPO and why it's preferred for some use cases (Rafailov et al. 2023)
- [ ] Review preference pair generation strategies (LLM-as-judge, RLAIF; Zheng et al. 2023)

## Personalized Retrieval and Generation

<!-- User-aware retrieval, preference-tuned generation, personalization in LLMs. -->
The differentiating contribution of the thesis, and its closest prior art. Three recent
papers each personalize one of the three places Simurgh intervenes — the retriever, the
query, and the generator — but none combine them, and none target Persian educational
text.

- **Optimizing the retriever for personalization.** Salemi et al. (2024, SIGIR) are the
  closest precedent: they train a retriever (Contriever) to feed a *frozen* FlanT5-XXL
  personalized documents, with **ROPG-RL** (REINFORCE; reward = the generated answer's
  task metric minus a baseline document's), **ROPG-KD** (distil the LLM's per-document
  utility into the retriever via KL), and **RSPG** (a learned selector that picks the
  best retriever per query from a pool). RSPG-Post wins on 6/7 LaMP tasks (avg +5.5% over
  SOTA, +15.3% over a non-personalized LLM). This is the blueprint for Simurgh's
  RL-optimized retriever — swap the generic task-metric reward for an LLM-judge
  pedagogical-fit score. Key divergence: LaMP personalizes from each user's *own document
  history*, whereas Simurgh personalizes from a *declared learner profile* over a shared
  Persian corpus, and RSPG's per-query selection is a natural analogue to choosing a
  retrieval strategy per *learner*. → [`optimization-methods-for-personalizing-large-language-models-through-retrieval-augmentation.md`](references/optimization-methods-for-personalizing-large-language-models-through-retrieval-augmentation.md)
- **Personalizing the query before retrieval.** Zhang et al. (2025, PBR) personalize
  *before* retrieving, training-free: **P-PRF** prompts an LLM to generate pseudo-feedback
  in the user's own expression style, and **P-Anchor** runs Personalized PageRank over a
  graph of the user's corpus to anchor the query in their semantics. It lifts Recall@5
  ~10% on PersonaBench, and ablations show P-PRF carries most of the gain. Because it
  needs no training, PBR doubles as a recipe for the persona-conditioned rewriter *and* a
  ready **persona-prompting baseline rung** (ladder step 2) to beat with DPO — though its
  history/corpus conditioning must be re-grounded on Simurgh's profile schema and made
  Persian-aware (ZWNJ, ye/ke). → [`personalize-before-retrieve-llm-based-personalized-query-expansion-for-user-centric-retrieval.md`](references/personalize-before-retrieve-llm-based-personalized-query-expansion-for-user-centric-retrieval.md)
- **Aligning the generator to pedagogy.** Dinucu-Jianu et al. (2025, EMNLP) turn a 7B LLM
  into a tutor with multi-turn GRPO over *simulated* student–tutor dialogues — no human
  annotation. The reward `r_sol + λ·(r_ped − 1)` adds a post-dialogue student solve-rate
  term to an LLM-judge pedagogical term (answer-leakage + helpfulness), with λ tracing an
  explicit teach-vs-solve Pareto frontier. The 7B tutor matches LearnLM on teaching
  quality while *preserving* reasoning, where SFT degrades it. Every ingredient — small
  model, synthetic learners, a decomposed pedagogical reward, an explicit leakage guard —
  is a template for Simurgh's judge rubric and simulator in the adjacent generation task.
  → [`from-problem-solving-to-teaching-problem-solving-aligning-llms-with-pedagogy-using-reinforcement-learning.md`](references/from-problem-solving-to-teaching-problem-solving-aligning-llms-with-pedagogy-using-reinforcement-learning.md)

- [x] Survey user-aware retrieval methods (ROPG/RSPG retriever optimization; PBR query
      personalization)
- [x] Cover preference-tuned generation approaches (LaMP via Salemi et al.; pedagogical
      GRPO via Dinucu-Jianu et al.)
- [x] Review personalization in educational / tutoring systems (Dinucu-Jianu et al. 2025)

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
With the RL and personalization sections filled, the gap sharpens. Foundational RAG is
user-agnostic; adaptive RAG (Self-RAG, Adaptive-RAG, CRAG) conditions on *question*
properties, not *user* properties. Each piece of the closest prior art covers one of
Simurgh's three intervention points but stops short: Salemi et al. optimize the
**retriever**, but for a user's *document history* and a generic task metric
(accuracy/ROUGE), not a *declared learner profile* scored for pedagogy; DeepRetrieval and
s3 train the **rewriter/searcher** with RL, but reward *retrieval or answer-correctness
utility*, not persona fit; PBR personalizes the **query** yet is training-free and
English; the pedagogy work aligns the **generator** for teaching but leaves *retrieval*
untouched and is English math, not Persian text. None combine persona-conditioned
**retrieval + rewriting** optimized for **pedagogical quality** (LLM-judge +
human-validated) over **Persian** educational text — which is precisely Simurgh's
contribution. The transferable mechanism is a reward in the spirit of s3's
Gain-Beyond-RAG and the pedagogy paper's `r_sol + λ·r_ped`, but keyed to a learner
profile and validated rung-by-rung against the baseline ladder.

- [x] Draft initial gap statement from foundations (refine after RL + personalization batches)
- [x] Synthesize gaps across all sections once RL and personalization sections are written
- [x] State clearly what this thesis contributes that prior work does not
