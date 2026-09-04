# Literature Review

> A focused review of the methods that determine this thesis's pipeline: foundational
> RAG, lexical and dense retrieval, query rewriting, DPO, and personalized retriever
> optimization. One reference note per paper lives in
> [`references/`](references/INDEX.md). The review is intentionally selective rather than
> a general RAG or reinforcement-learning survey.

The thesis optimizes a retrieval-augmented generation (RAG) pipeline for Persian
educational personalization. Three lines of prior work define the design:

1. **Retrieving external evidence** — foundational RAG and the lexical/dense methods that
   select evidence before generation.
2. **Transforming the query** — rewriting methods that bridge the user's wording and the
   representation that retrieves useful documents.
3. **Learning from downstream feedback** — DPO for offline preference learning and
   ROPG/RSPG for optimizing a retriever against a personalized generation objective.

These lines play different roles. RAG defines the decomposition; BM25, DPR, and
Contriever provide retrieval baselines and training patterns; Rewrite-Retrieve-Read and
HyDE show how generated text can intervene before retrieval; DPO and ROPG explain how
feedback can train the two components selected in this thesis. The review asks of each
method what it optimizes, how it works, and why its evidence does not directly establish
learner-conditioned retrieval over a shared Persian corpus.

---
## Retrieval-Augmented Generation and Retrieval Foundations

### Pipeline decomposition

Lewis et al. (2020) define RAG as the combination of parametric memory in a
sequence-to-sequence generator and non-parametric memory in a dense document index. A
retriever assigns probabilities to passages for an input, and the generator conditions
its output on the input and retrieved passage. RAG-Sequence uses one latent passage for
an entire output sequence, whereas RAG-Token permits the latent passage to vary across
generated tokens. Both marginalize over a truncated top-*K* set instead of treating the
single first-ranked passage as certain.

This decomposition makes the evidence store explicit and updateable, but the original
objective concerns task relevance and factual generation. It does not distinguish two
passages that answer the same question but differ in pedagogical suitability. Simurgh
therefore adopts the retrieval-before-generation boundary while treating
learner-conditioned document utility as a separate quantity.
→ [`retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks`](references/retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks.md)

### Lexical and dense retrieval

Robertson and Zaragoza (2009) place BM25 in the probabilistic relevance framework. BM25
scores exact query-term matches using inverse document frequency, saturating term
frequency, and document-length normalization. It is efficient over an inverted index and
transparent about why a document receives a score. Those same properties expose its
boundary: a relevant passage with no matching indexed term receives no semantic credit,
and orthographic variation changes the observed matches. This makes BM25 both a useful
baseline and a direct diagnostic for Persian character and token normalization.
→ [`the-probabilistic-relevance-framework-bm25-and-beyond`](references/the-probabilistic-relevance-framework-bm25-and-beyond.md)

Karpukhin et al. (2020) replace sparse term matching with DPR's supervised dual encoder.
Question and passage encoders produce dense vectors, their inner product gives the
retrieval score, and positive passages are contrasted with in-batch or BM25-derived
negatives. Passage vectors can be precomputed, preserving efficient corpus search. The
paper reports 9–19 percentage-point absolute gains over a strong Lucene-BM25 baseline in
top-20 passage retrieval accuracy across its open-domain QA datasets. The evidence shows
the value of learned semantic matching, but it depends on English QA relevance labels and
does not demonstrate learner-conditioned utility.
→ [`dense-passage-retrieval-for-open-domain-question-answering`](references/dense-passage-retrieval-for-open-domain-question-answering.md)

Izacard et al. (2022) address the label requirement with Contriever, which learns dense
representations through unsupervised contrastive training on paired crops and in-batch
negatives. In its unsupervised BEIR evaluation, Contriever exceeds BM25 on Recall@100 for
11 of 15 datasets and also supports multilingual and cross-lingual transfer. Contriever
therefore provides a stronger zero-shot starting point than assuming a dense retriever
must be trained in-domain. Its limitation remains the objective: general semantic
similarity is neither evidence of pedagogical usefulness nor evidence that a profile
changes the ranking.
→ [`unsupervised-dense-information-retrieval-with-contrastive-learning`](references/unsupervised-dense-information-retrieval-with-contrastive-learning.md)

Zhang et al. (2025) provide the concrete backbone used in this thesis through the Qwen3
Embedding family. The series derives embedding and reranking models at 0.6B, 4B, and 8B
scales from Qwen3 foundation models and combines unsupervised pretraining with supervised
multilingual fine-tuning. Simurgh uses Qwen3-Embedding-0.6B because it preserves the
family's multilingual and instruction-aware retrieval interface at the smallest released
scale. This is an implementation choice, not evidence of Persian educational quality:
the checkpoint must still be evaluated before and after in-domain training, and profile
sensitivity must be tested separately.
→ [`qwen3-embedding-advancing-text-embedding-and-reranking-through-foundation-models`](references/qwen3-embedding-advancing-text-embedding-and-reranking-through-foundation-models.md)

| Method | Representation | Supervision | Main role here | Unresolved boundary |
|---|---|---|---|---|
| BM25 | Sparse terms | None | Transparent lexical baseline | No semantic match without shared terms |
| DPR | Dense vectors | Labeled question-passage pairs | Supervised dual-encoder pattern | Labels encode relevance, not learner utility |
| Contriever | Dense vectors | Unsupervised contrastive pairs | Zero-shot dense starting point | Similarity does not establish personalization |
| Qwen3-Embedding-0.6B | Dense vectors | Multistage multilingual training | Thesis retrieval backbone | Benchmark capability does not establish in-domain or profile-dependent quality |

### Query rewriting before retrieval

Ma et al. (2023) make the query an explicit trainable interface in
Rewrite-Retrieve-Read. A language model first rewrites the input, a search system
retrieves evidence for that rewrite, and a frozen reader produces the answer. Their
trainable variant transfers reader feedback to a smaller rewriter through reinforcement
learning. The method changes retrieval without modifying the retriever or reader;
however, its reward measures downstream QA performance rather than learner fit, and its
evidence comes from English QA with web search.
→ [`query-rewriting-in-retrieval-augmented-large-language-models`](references/query-rewriting-in-retrieval-augmented-large-language-models.md)

Gao et al. (2023) propose a different bridge in HyDE. Instead of producing a search-style
query, an instruction-following model generates a hypothetical relevant document;
Contriever embeds that generated document and retrieves nearby real passages. The
hypothetical text may contain false statements, so it is used only as a representation,
not as evidence supplied directly to the answer. HyDE shows that document-shaped
generation can reduce the query-document representation gap without relevance labels,
but supplies no rule for how learner characteristics should change that representation.
→ [`precise-zero-shot-dense-retrieval-without-relevance-labels`](references/precise-zero-shot-dense-retrieval-without-relevance-labels.md)

Together, these works justify separating retrieval, rewriting, and generation, but none
tests whether a profile-dependent rewrite or score changes document utility for learners
who all search the same corpus. That distinction between generic retrieval improvement
and personalization is the boundary carried into the thesis experiments.


### RLHF and PPO

Ouyang et al. (2022) provide the practical RLHF lineage used here: supervised fine-tuning,
reward-model training from ranked responses, and PPO optimization of the policy under a
penalty to a frozen reference. This pipeline can align generation with preferences, but it
requires online policy samples, a separate reward model, and a value model. Those costs
motivate the offline alternative below.
→ [`training-language-models-to-follow-instructions-with-human-feedback`](references/training-language-models-to-follow-instructions-with-human-feedback.md)

## Preference Optimization from Pairwise Data — DPO

Rafailov et al. (2023) target the operational cost of reinforcement learning from human
feedback (RLHF). Standard RLHF is a three-stage pipeline: supervised fine-tuning (SFT);
fitting a separate reward model on preference pairs under the Bradley-Terry model; and
then RL fine-tuning of the policy against that reward with a KL penalty to a frozen
reference, optimized with PPO. The RL stage samples from the policy during training,
needs a value baseline, and is sensitive to hyperparameters.

**Method.** DPO's key observation is that the KL-constrained reward-maximization objective
has a closed-form optimal policy, and the latent reward can be reparameterized in terms of
that optimal policy and the reference policy —
`r(x,y) = β·log(π(y|x)/π_ref(y|x)) + β·log Z(x)`. Because the Bradley-Terry preference
model depends only on the *difference* of rewards between two completions, the intractable
partition function `Z(x)` cancels, and maximum likelihood on the preference data reduces to
a single binary-cross-entropy loss applied directly to the policy:

`L_DPO = − E[ log σ( β·log(π_θ(y_w|x)/π_ref(y_w|x)) − β·log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]`.

The gradient increases the likelihood of the preferred completion and decreases the
dispreferred one, weighted by how wrongly the policy's *implicit* reward currently ranks
the pair. There is no explicit reward model, no sampling during training, and no RL loop.
The paper proves the reparameterization is fully general (any Bradley-Terry / Plackett-Luce
reward class is representable), so no expressivity is lost versus explicit reward modelling.

**What it offers / what we take from it.** DPO shows that, given offline `(prompt, chosen,
rejected)` triples and a frozen reference model, preference alignment can be done with a
supervised-style classification loss — no online rollouts. The reference policy and the
scalar `β` together set the implicit KL budget (how far the policy may drift). The
per-example importance weight is load-bearing: a naive "raise chosen, lower rejected"
objective (Unlikelihood) degenerates, so the σ-weighting that down-weights already-correct
pairs is what keeps training stable. Empirically DPO matches or beats PPO-based RLHF on
controlled sentiment, summarization (TL;DR), and single-turn dialogue (Anthropic HH), while
being simpler and more robust to sampling temperature. A useful by-product: the trained
policy exposes an implicit reward `β·log(π_θ/π_ref)` that can be read off for ranking or
filtering. The main caveats are off-policy/distribution-shift risk (the preference data may
not match the live policy) and sensitivity to the choice of reference policy.
→ [`direct-preference-optimization-your-language-model-is-secretly-a-reward-model`](references/direct-preference-optimization-your-language-model-is-secretly-a-reward-model.md)

### DPO variants used in the thesis

Three extensions address distinct DPO failure modes. WPO reweights offline pairs by their
probability under the current policy, targeting the mismatch between the data-generating
model and the trained policy (Zhou et al., 2024). Robust DPO corrects the loss under an
assumed uniform preference-label flip rate (Ray Chowdhury et al., 2024). RPO adds supervised
negative log-likelihood on preferred responses, which prevents a model from improving the
preference margin solely by making both responses less likely (Liu et al., 2024).

These changes are alternatives only in part. WPO changes pair weights, robust DPO changes
the noise model, and RPO adds an anchor; the thesis therefore applies the RPO anchor to
each objective arm rather than treating it as a fourth competing arm.
→ [`wpo-enhancing-rlhf-with-weighted-preference-optimization`](references/wpo-enhancing-rlhf-with-weighted-preference-optimization.md)
→ [`provably-robust-dpo-aligning-language-models-with-noisy-feedback`](references/provably-robust-dpo-aligning-language-models-with-noisy-feedback.md)
→ [`provably-mitigating-overoptimization-in-rlhf`](references/provably-mitigating-overoptimization-in-rlhf-your-sft-loss-is-implicitly-an-adversarial-regularizer.md)

### Parameter-efficient adaptation

LoRA freezes the pretrained model and learns low-rank updates in selected Transformer
layers (Hu et al., 2022). QLoRA retains that update rule while storing the frozen base model
in four-bit form and backpropagating into the adapters (Dettmers et al., 2023). The thesis
uses LoRA for both trainable components and the QLoRA form for the Qwen3-4B rewriter; this
choice reduces memory use but does not itself establish equal quality to full fine-tuning.
→ [`lora-low-rank-adaptation-of-large-language-models`](references/lora-low-rank-adaptation-of-large-language-models.md)
→ [`qlora-efficient-finetuning-of-quantized-llms`](references/qlora-efficient-finetuning-of-quantized-llms.md)

## Retriever Optimization for Personalization — ROPG / LaMP

Salemi et al. (2024) study the retriever in a personalized RAG pipeline. On the LaMP
benchmark, each user carries a profile of their own prior `(input, output)` documents; a
retriever selects a few of them to condition a **frozen** generator (FlanT5-XXL). Prior
work left the retriever off-the-shelf; this paper is the first to *train* it, and its
central claim is that retrieval feedback for personalization should come from the
downstream personalized objective, because generic relevance and personal usefulness
diverge.

**Method.** The paper contributes two families of methods on top of the LaMP pipeline:

- **ROPG — retriever optimization.** Two training signals, both derived from the frozen
  LLM's behaviour on the task.
  - *ROPG-RL:* the retriever is a policy (a softmax over its scores for the top-16 profile
    documents) trained with REINFORCE. The reward for a sampled document is its downstream
    task metric minus the metric of the un-fine-tuned retriever's top-1 document (a
    variance-reduction baseline). Because the metric can be non-differentiable, no gradient
    flows through the LLM — only the policy-gradient on the retriever's distribution.
  - *ROPG-KD:* a supervised alternative. The frozen LLM scores each candidate document's
    per-document usefulness; those scores form a softmax *teacher* distribution, and the
    retriever's softmax is trained to match it via KL divergence. The authors report RL as
    less stable and more overfit-prone, and KD as the more robust option (at the cost of one
    extra teacher pass per document).
- **RSPG — retriever selection.** Because no single retriever is best across all inputs, a
  Longformer-based selection model picks, per input, among a pool
  `{no-personalization, recency, BM25, Contriever, ROPG-RL, ROPG-KD}`. *RSPG-Pre* scores
  each candidate's prompt before generation; *RSPG-Post* scores each `(prompt, LLM output)`
  pair after generation. Both are trained with the same KL-distillation recipe as ROPG-KD,
  but over retrievers rather than documents.

**What it offers / what we take from it.** On LaMP, RSPG-Post is best on 6 of 7 tasks
(average +5.5% over prior SOTA; +15.3% average over a non-personalized LLM), with an
ablation showing the ROPG-trained retrievers are necessary — the gains are not from the
selector alone. The transferable lessons are method-level: (1) train the retriever against
the downstream personalized signal, not generic relevance, when the only available feedback
is the output text rather than per-document labels; (2) distillation from a frozen LLM's
per-document utility is a cheaper, more stable route than online RL for a retriever; (3)
per-input selection over a pool of retrieval strategies can beat committing to one retriever
globally; (4) a frozen generator is enough to make retriever optimization meaningful, which
bounds cost. The reward-hacking caveat applies to the retriever too — the metric that trains
it is also the metric it is judged on — so retrieval-side diagnostics (Recall@K, MRR,
selection-success rate) are the guard. Scope note: LaMP personalizes from each user's own
document *history* and is English and short-output; the method transfers even where the
benchmark does not.
→ [`optimization-methods-for-personalizing-large-language-models-through-retrieval-augmentation`](references/optimization-methods-for-personalizing-large-language-models-through-retrieval-augmentation.md)

## Positioning

These two papers sit on opposite sides of a RAG pipeline and answer different questions.
DPO is a general recipe for converting preference pairs into an aligned generation policy
without an explicit reward model or an online RL loop. ROPG is a general recipe for
converting a downstream task signal into a personalized retriever, with a selection layer on
top. The thesis draws on both: the preference-optimization machinery for training from
pairwise feedback, and the downstream-objective framing for making retrieval personalized.
The specific way these are combined and evaluated is developed in
[`methodology`](methodology.md) and [`experiment-design`](experiment-design.md); this review
records what each method is, what it demonstrates, and the lessons carried forward.
