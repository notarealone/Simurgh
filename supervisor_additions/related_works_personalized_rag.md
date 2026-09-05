

Yes—but with an important distinction: **the original ROPG paper does not train a decoder-only retriever or rewriter**. It trains a dense **encoder-only bi-encoder** retriever, while the downstream generator is FlanT5, an encoder–decoder model. The closest decoder-only work I found is on **RL-trained query rewriting**, rather than ROPG itself.

## Most relevant papers

| Paper | Relevance to your project |
|---|---|
| [Optimization Methods for Personalizing Large Language Models through Retrieval Augmentation](https://www.alphaxiv.org/abs/2404.05970) | The source paper for **ROPG-RL** and **ROPG-KD**. It optimizes retrieval using downstream personalization quality rather than document-relevance labels. |
| [LaMP: When Large Language Models Meet Personalization](https://www.alphaxiv.org/abs/2304.11406) | Benchmark and task framework for personalized generation; useful for positioning your persona-conditioned Persian education setting. |
| [Direct Preference Optimization](https://www.alphaxiv.org/abs/2305.18290) | Relevant to your query-rewriter training with chosen/rejected rewrites. |
| [Rethinking On-policy Optimization for Query Augmentation](https://www.alphaxiv.org/abs/2510.17139) | The closest match to your **decoder-only query rewriter** idea: it trains an autoregressive LLM with retrieval rewards using PPO, and introduces OPQE. |
| [Reinforcement Fine-Tuning for History-Aware Dense Retriever in RAG](https://www.alphaxiv.org/abs/2602.03645) | A newer end-to-end RL retriever paper. It optimizes document selection against final QA reward, but its retriever is still an embedding model rather than a decoder-only LM. |
| [Optimizing RAG Rerankers with LLM Feedback via Reinforcement Learning](https://www.alphaxiv.org/abs/2604.02091) | Relevant if you later replace the embedding retriever with an LLM-based reranker trained from judge feedback. |

## Is there an ROPG paper for a decoder-only model?

### 1. Decoder-only **query rewriter**: yes, closely related

[Rethinking On-policy Optimization for Query Augmentation](https://www.alphaxiv.org/abs/2510.17139) treats an autoregressive language model as a policy:

$$
\pi_\theta(q' \mid q)
$$

The model generates an augmented query, retrieves documents with a fixed retriever, receives a retrieval reward such as Recall or NDCG, and is optimized with PPO. The paper explicitly describes each generated token or sub-query as an action in an RL process. [RL Query Augmentation](https://www.alphaxiv.org/abs/2510.17139?page=5)

This is very close to your **DPO-trained rewriter**, although the optimization method differs:

- Their method: **on-policy PPO**, with fresh rollouts and retrieval rewards.
- Your method: **offline preference optimization**, where a judge ranks several rewrites and DPO learns from chosen/rejected pairs.
- Their retriever is generally frozen.
- Your system trains both the retriever and the rewriter.

The paper also proposes **OPQE**, where the decoder-only policy generates a pseudo-document rather than a short rewritten query. This is especially relevant because dense retrievers often respond better to passage-like text than to terse query reformulations. [OPQE](https://www.alphaxiv.org/abs/2510.17139?page=11)

For your project, OPQE suggests a useful ablation:

1. original question;
2. short persona-aware rewritten query;
3. pseudo-document-style persona-aware query;
4. concatenation of original query and pseudo-document.

### 2. Decoder-only **retriever**: I did not find a direct ROPG implementation

The original ROPG work defines the retriever as a dense bi-encoder. The query and document are encoded separately, and retrieval uses dot-product similarity followed by nearest-neighbor search. [Retriever Architecture](https://www.alphaxiv.org/abs/2404.05970?page=5)

Its ROPG-RL objective samples a document according to the retriever policy and rewards it according to downstream personalized-generation quality. [ROPG-RL Objective](https://www.alphaxiv.org/abs/2404.05970?page=4)

ROPG-KD similarly converts downstream utility scores into a target document distribution and trains the retriever with KL divergence. [ROPG-KD Objective](https://www.alphaxiv.org/abs/2404.05970?page=4)

So the answer is:

- **ROPG applied to an encoder retriever:** yes—the original paper.
- **ROPG applied to a decoder-only query generator:** not under the ROPG name, but the PPO query-augmentation literature does this.
- **ROPG applied to a decoder-only document retriever:** I did not find a direct paper among the relevant results.

## Why the original ROPG paper is not decoder-only

The original architecture is designed for efficient indexing:

```text
query ──> encoder ──┐
                    ├── dot product ──> ranked documents
document ─> encoder ┘
```

A decoder-only LM would normally score a document conditionally:

$$
s(q,d)=\log p_\theta(d\mid q)
$$

or perhaps score the query conditioned on the document:

$$
s(q,d)=\log p_\theta(q\mid d)
$$

That is much more expensive because documents cannot be independently pre-embedded in the same simple way as a bi-encoder. For a corpus of 171 chunks, as in your project, this is feasible; for a large corpus, it becomes impractical unless you use:

- cached document representations,
- a decoder-only model converted into a dual encoder,
- late-interaction representations,
- a small candidate set from an embedding retriever,
- or decoder-only scoring only as a reranker.

## How the papers map onto your design

Your system can be described as combining three lines of work:

### Personalized retrieval optimization

Your retriever stage is closest to **ROPG-KD**, but with a persona-conditioned utility function:

$$
u(q,p,d)
$$

where $q$ is the question, $p$ is the learner persona, and $d$ is a corpus chunk. The original ROPG paper also argues that the retrieval model should be optimized using downstream personalized-generation feedback rather than generic relevance labels. Its experiments show that ROPG-RL and ROPG-KD improve different LaMP tasks, with no universally dominant method. [ROPG Results](https://www.alphaxiv.org/abs/2404.05970?page=8)

### Decoder-only query rewriting

Your rewriter is closer to the query-augmentation formulation in [Rethinking On-policy Optimization for Query Augmentation](https://www.alphaxiv.org/abs/2510.17139) than to the original ROPG retriever. That paper explicitly distinguishes online PPO from offline preference tuning such as DPO: PPO optimizes live retrieval rewards, whereas DPO learns from a fixed preference dataset and therefore optimizes a surrogate preference objective. [PPO versus Preference Tuning](https://www.alphaxiv.org/abs/2510.17139?page=4)

This distinction is important for interpreting your negative first DPO run: the rewriter may learn to suppress rejected outputs without increasing the probability of chosen outputs. That failure mode is not evidence that query rewriting cannot work; it indicates that the preference objective and checkpoint selection did not sufficiently enforce chosen-output imitation.

### End-to-end retriever optimization

[Reinforcement Fine-Tuning for History-Aware Dense Retriever in RAG](https://www.alphaxiv.org/abs/2602.03645) is a useful newer comparison. It models retrieval as a stochastic policy and optimizes the final answer F1 reward. However, it uses Qwen embedding models as retrievers, not a decoder-only language model. [Retriever Training Setup](https://www.alphaxiv.org/abs/2602.03645?page=6)

## What I would cite as the closest prior work

For your thesis, I would make the positioning explicit:

> To our knowledge, this is an application of downstream-utility retrieval optimization and preference-trained query rewriting to persona-conditioned educational retrieval. The retriever component is inspired by ROPG-KD, while the decoder-only rewriter is related to RL-based query augmentation and DPO-based preference tuning, but the complete combination has not been directly evaluated in the ROPG literature.

The strongest related-work sequence is:

1. [ROPG](https://www.alphaxiv.org/abs/2404.05970) — downstream optimization of personalized retrievers.
2. [LaMP](https://www.alphaxiv.org/abs/2304.11406) — personalization benchmark.
3. [DPO](https://www.alphaxiv.org/abs/2305.18290) — offline preference optimization.
4. [Rethinking On-policy Optimization for Query Augmentation](https://www.alphaxiv.org/abs/2510.17139) — decoder-only retrieval-aware query generation.
5. [HARR](https://www.alphaxiv.org/abs/2602.03645) — end-to-end RL optimization of dense retrieval.

The main novelty in your setup is therefore not simply “using ROPG with a decoder-only model.” It is the **combination of persona-conditioned utility, learned dense retrieval, and preference-trained query rewriting in a low-resource educational domain**.

## User

add below link to related works and add more papers for related works.
Retrieval-Augmented Tutoring for Algorithm Tracing and Problem-Solvingin AI Educationhttps://arxiv.org/pdf/2605.12988
i want list of thesis contribution.

i want table for related works papers and its relttion to thesis.

write a paragraph about novelty and contribution of work not seen in previos papers

## Aurelle

## Related work and positioning

Your thesis is positioned at the intersection of **personalized retrieval**, **retrieval-augmented education**, **retrieval-aware query rewriting**, and **preference optimization of language models**. The closest prior work is the ROPG paper, but your system extends its idea from general personalized generation to **persona-conditioned educational retrieval in Persian**, with a separately trained query rewriter.

### Related works

| Paper | Main idea | Relation to this thesis |
|---|---|---|
| [LaMP: When Large Language Models Meet Personalization](https://www.alphaxiv.org/abs/2304.11406) | Introduces a benchmark for personalized language-model tasks using user profiles and personal information. | Provides the broader personalization motivation. Your thesis focuses on educational personas rather than historical user profiles, and evaluates personalization at the retrieval and query-rewriting levels. |
| [Optimization Methods for Personalizing Large Language Models through Retrieval Augmentation](https://www.alphaxiv.org/abs/2404.05970) | Introduces ROPG-RL and ROPG-KD for optimizing retrievers using downstream personalized-generation utility. The retriever is a dense encoder-based bi-encoder. | This is the primary methodological foundation for your retriever. Your work adapts the KD idea to learner personas, educational documents, and LLM-judge utility scores. The original paper uses a generic personalized-generation setting rather than persona-conditioned tutoring. The authors optimize dense retrieval models using downstream task feedback. [ROPG Framework](https://www.alphaxiv.org/abs/2404.05970?page=1) |
| [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://www.alphaxiv.org/abs/2305.18290) | Trains a language model directly from chosen/rejected preference pairs without an explicit reward model or PPO loop. | Provides the learning objective for your query rewriter. Your chosen and rejected rewrites are generated and judged according to persona-specific retrieval usefulness. |
| [Rethinking On-policy Optimization for Query Augmentation](https://www.alphaxiv.org/abs/2510.17139) | Optimizes autoregressive decoder-only query generators with retrieval rewards using PPO. It also proposes pseudo-document generation through OPQE. | This is the closest work to your decoder-only rewriter. It demonstrates that an autoregressive model can be trained to produce retrieval-effective queries. Your thesis differs by using offline preference optimization with DPO rather than on-policy PPO, and by conditioning the rewrite on learner persona. The paper explicitly contrasts online PPO with offline preference tuning such as DPO. [Query Optimization](https://www.alphaxiv.org/abs/2510.17139?page=4) |
| [Reinforcement Fine-Tuning for History-Aware Dense Retriever in RAG](https://www.alphaxiv.org/abs/2602.03645) | Treats retrieval as a stochastic policy and optimizes document selection using downstream answer reward, including multi-hop retrieval history. | Supports the idea that retrievers should be optimized against end-task performance rather than generic relevance. However, it trains Qwen embedding retrievers, not a persona-conditioned educational retriever, and does not address learner profiles. |
| [Retrieval-Augmented Tutoring for Algorithm Tracing and Problem-Solving in AI Education](https://www.alphaxiv.org/abs/2605.12988) | Presents KITE, a retrieval-augmented tutoring system with multimodal course retrieval, intent-aware responses, Socratic feedback, and simulated-student evaluation. | This is the most relevant educational RAG paper. It demonstrates that retrieval-grounded tutoring can support algorithmic reasoning and scaffolding. However, KITE uses a fixed multi-stage retrieval pipeline and does not train retrieval or query rewriting according to individual learner personas. KITE reports strong grounding and context-relevance scores, but its main contribution is tutoring architecture and evaluation rather than learned personalization. [KITE Contributions](https://www.alphaxiv.org/abs/2605.12988?page=2) |
| [DeepTutor: Towards Agentic Personalized Tutoring](https://www.alphaxiv.org/abs/2604.26962) | Develops an agentic tutoring approach that adapts educational interactions to individual learners. | Provides a broader reference for personalized and agentic tutoring. Your thesis is more retrieval-centric: it operationalizes personalization through explicit learner personas, persona-conditioned utility judgments, and trained retrieval components. |
| [Think-to-Personalize: Unifying Reasoning and Retrieval for User-Centric Personalized Dense Retrieval](https://www.alphaxiv.org/abs/2608.18855) | Combines reasoning and retrieval for user-aware dense retrieval. | Relevant to the idea that a retriever should model user-specific usefulness rather than only query-document similarity. Your thesis applies this principle to learner profiles and educational content, with a much smaller and controlled domain. |
| [Retrieval-Augmented Generation for Educational Application: A Systematic Survey](https://arxiv.org/abs/2504.08745) | Surveys educational applications of RAG, including grounding, tutoring, and learning support. | Useful for the general educational-RAG background section. It does not provide the specific retriever-training or persona-conditioned query-rewriting method proposed in your thesis. |
| RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | Establishes the general RAG paradigm by combining parametric generation with non-parametric retrieval. | Provides the foundational architecture underlying your tutoring system: retrieve educational passages and generate a grounded answer. Your contribution is to make both retrieval and query formulation learner-aware. |

### More direct comparison with the closest papers

The most important distinction is between **using retrieval in education** and **learning how retrieval should change for different learners**.

The KITE paper uses retrieval to ground an educational tutor and adapts the response strategy to pedagogical intent. Its pipeline includes dense retrieval, BM25, MMR, cross-encoder reranking, and source boosting. [KITE Retrieval Pipeline](https://www.alphaxiv.org/abs/2605.12988?page=4) Your thesis instead asks a different question: *which type of educational content is most useful for this particular learner persona?* For example, a crammer may benefit from vocabulary and simple verse explanations, while a scholar may benefit from literary analysis. This learner-specific utility signal is used to train the retriever itself.

The ROPG paper directly optimizes retrieval using downstream utility, but its retriever is a shared-parameter dense encoder and its personalization setting is based on user profiles. [Dense Retriever](https://www.alphaxiv.org/abs/2404.05970?page=5) Your thesis introduces a different conditioning structure: the persona is added to the query side, while documents are embedded once, allowing one shared index to serve multiple learner types.

The query-augmentation literature trains decoder-only models with retrieval-derived rewards. In particular, [Rethinking On-policy Optimization for Query Augmentation](https://www.alphaxiv.org/abs/2510.17139) models the rewriter as an autoregressive policy and optimizes retrieval effectiveness through PPO. [Decoder-only Rewriter](https://www.alphaxiv.org/abs/2510.17139?page=5) Your thesis investigates whether a smaller decoder-only model can acquire this behavior through offline DPO pairs generated by a stronger teacher and evaluated by a judge.

## Proposed thesis contributions

The following list is suitable for a thesis proposal or contribution section.

1. **A persona-conditioned Persian educational retrieval framework.**  
   The thesis develops a retrieval-augmented tutoring system for Persian ninth-grade literature, where retrieved content is selected according to both the question and the learner’s pedagogical profile.

2. **A structured Persian educational corpus for personalized retrieval.**  
   The corpus combines the official ninth-grade Persian textbook, advanced-student material, and a step-by-step explanatory guide. The sources are divided into semantically meaningful sections such as poems, literary notes, vocabulary, verse explanations, and deep literary analysis.

3. **A compositional learner-persona benchmark.**  
   The thesis defines learner personas along multiple dimensions, including prior knowledge, learning objective, and preferred explanation style. The held-out `newcomer` persona tests compositional generalization because its combination of attributes is not observed during training.

4. **Persona-aware utility supervision for retriever training.**  
   A judge model assigns utility scores to document–question–persona triples. These scores represent usefulness for a particular learner rather than generic topical relevance. This enables the retriever to learn that the same question may require different evidence for different learners.

5. **An adapted ROPG-KD retriever-training method.**  
   The thesis adapts the ROPG-KD principle to educational personalization. Because the judge scores are noisy, the method uses the reliable contrast between the best document and hard negatives rather than attempting to reproduce the entire noisy ranking distribution.

6. **A hard-negative and anchoring strategy for low-resource retriever adaptation.**  
   The retriever is trained using the highest-utility document against low-utility hard negatives, while an anchoring penalty prevents the LoRA adapter from moving too far from the strong pretrained embedding model.

7. **A persona-aware decoder-only query rewriter.**  
   The thesis trains a local decoder-only language model to rewrite questions in a way that improves retrieval for a specified learner profile. This separates query formulation from final answer generation.

8. **Preference optimization using retrieval usefulness.**  
   Multiple candidate rewrites are generated, judged according to their ability to retrieve useful educational passages, and converted into chosen/rejected pairs for DPO training. Cross-persona negatives are also used, such as treating a scholar-oriented rewrite as a negative example for a crammer persona.

9. **A controlled component-level evaluation.**  
   The thesis evaluates five configurations: naive retrieval, persona prompting, trained retrieval, trained retrieval with an untrained rewriter, and the full trained system. This isolates the contributions of persona prompting, retriever training, and query-rewriter training.

10. **A diagnostic evaluation of genuine personalization.**  
    In addition to standard retrieval metrics such as Recall@5, MRR, and nDCG, the thesis uses a persona-swap test. The same question is evaluated with an incorrect persona to determine whether the retriever actually uses learner information rather than merely improving generic retrieval.

11. **An empirical analysis of preference-transfer failure.**  
    The thesis documents when DPO fails to improve a small query rewriter, including the failure mode where training suppresses rejected rewrites without sufficiently increasing the likelihood of chosen rewrites. This is useful negative evidence for low-resource retrieval-aware preference training.

12. **A reproducible evaluation protocol for personalized educational RAG.**  
    The complete framework combines retrieval metrics, LLM-based educational-quality evaluation, persona fit, faithfulness, reference-answer metrics, preference accuracy, and ablations. This provides a more complete evaluation than measuring either retrieval quality or final answer quality alone.

## Novelty and contribution paragraph

This thesis proposes a form of personalization that is not present in the previous papers considered here: **personalization is learned inside both the retriever and the query-rewriting model, rather than being implemented only through prompting or response-generation strategies**. Existing educational RAG systems such as KITE focus on grounding and pedagogical response behavior, but use a fixed retrieval pipeline and do not learn which evidence is useful for different learner profiles. ROPG optimizes personalized retrieval using downstream utility, but does not study persona-conditioned Persian educational retrieval or a decoder-only preference-trained query rewriter. Conversely, decoder-only query-augmentation work optimizes retrieval-aware generation, primarily with on-policy PPO, but does not address learner personas or educational content selection. The proposed thesis combines these directions into one controlled framework: a persona-conditioned retriever trained from judge-based educational utility, a decoder-only query rewriter trained from DPO preference pairs, cross-persona hard negatives, and a compositional held-out learner persona. The central novelty is therefore the joint study of **learner-specific evidence selection and learner-specific query formulation** in a low-resource Persian educational RAG setting, with explicit experiments designed to distinguish genuine personalization from general retrieval improvement.