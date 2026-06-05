# Experiment Design

> Datasets, baselines, metrics, and evaluation protocol.

---

## Datasets

<!-- What datasets will you use? (e.g., MS MARCO, Natural Questions, custom) -->

- [ ] Choose source textbooks and grade levels
- [ ] Decide dataset size (questions × profiles per question)
- [ ] Decide: synthetic profiles (LLM-generated) or real student data
- [ ] Decide: who writes ground-truth answers
- [ ] Build the dataset

## Baselines

<!-- What systems will you compare against? -->

- [ ] Define baseline ladder: naive RAG → persona prompting → DPO rewriter → full system
- [ ] Implement or configure each baseline

## Metrics

<!-- Retrieval: Recall@K, MRR, NDCG -->
<!-- Generation: ROUGE, BLEU, faithfulness, relevance -->
<!-- Personalization: user satisfaction, preference alignment -->

- [ ] Define retrieval metrics (Recall@K, MRR) per persona
- [ ] Define generation metrics (Exact Match, F1, ROUGE/BLEU)
- [ ] Define personalization metrics (LLM-as-judge, human evaluation)
- [ ] Decide primary vs. secondary metrics

## Evaluation Protocol

<!-- How will experiments be run and results reported? -->

- [ ] Define train/val/test split strategy
- [ ] Specify number of runs and statistical significance testing
- [ ] Define how results are reported (tables, plots, significance)

## Ablation Studies

<!-- Which components will you ablate to understand their contribution? -->

- [ ] Ablation 1: RL optimization removed (DPO rewriter only)
- [ ] Ablation 2: personalization removed (no user profile)
- [ ] Ablation 3: query rewriting removed (raw query → retriever)
- [ ] Document findings in [[results]]
