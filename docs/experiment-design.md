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

Personalization is the thesis claim, so **persona alignment / pedagogical quality is the primary metric**. EM/F1 measure answer-string correctness — two equally "correct" answers can suit very different students — so report them as **secondary** evidence alongside retrieval metrics.

**Primary — personalization quality**
- [ ] LLM-as-judge rubric score (persona fit + pedagogical quality + faithfulness) on a held-out set
- [ ] Human evaluation on a 50–100 sample to confirm judge scores track real pedagogical quality

**Secondary — correctness & retrieval**
- [ ] Generation correctness: Exact Match, F1 (and ROUGE/BLEU where a reference answer exists)
- [ ] Retrieval quality per persona: Recall@K, MRR

> **Judge independence:** the judge that *scores* final results must not be the same prompt/model that *generated* the DPO preference pairs, or the numbers partly measure "optimizing to the judge." See [[things-to-consider]] (Reward Signal).

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
