# Literature Review

> A focused review of the two works this thesis builds on directly. One reference note
> per paper lives in [`references/`](references/INDEX.md); this file synthesizes them into
> an argument. The review is deliberately narrowed to these two methods — the ones the
> thesis extends — rather than a broad RAG/RL survey. Persona-modelling references are
> kept separately for [`personas.md`](personas.md) and are out of scope here.

The thesis optimizes a retrieval-augmented generation (RAG) pipeline with reinforcement
learning for personalization. Two lines of prior work anchor that goal, one on each side
of the pipeline:

1. **Aligning a model from preferences** — *Direct Preference Optimization* (DPO), a
   reward-model-free way to train a policy directly from preference pairs.
2. **Optimizing the retriever for personalization** — the *ROPG / RSPG* methods on the
   LaMP benchmark, the first work to train a retriever from the downstream
   personalized-generation objective rather than from generic relevance.

The two are complementary: DPO is a method for turning preference data into an aligned
generation policy; ROPG is a method for turning a downstream task signal into a better
retriever. Together they cover the "how to train from preferences/feedback" and "how to
personalize retrieval" questions that a personalized RAG system has to answer.

---

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
