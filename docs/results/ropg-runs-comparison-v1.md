# ROPG runs comparison v1

## Decision

Select **Run B** as the Stage 1 ROPG checkpoint.

Run B has the highest Recall@5, the metric used by `src/rl/ropg_kd.py::is_better` to select checkpoints. It also has the highest nDCG@5 and Hit@5. Run C is slightly better at the first few ranking positions, but none of its advantages over Run B remains statistically significant after Holm correction. Run B's Recall@5 advantage over Run C does remain significant.

Run D is the weakest arm. Freezing the document tower reduced graded ranking quality and recall. None of the four runs produced a statistically significant persona-swap effect, so these results support generic retrieval improvement, not Stage 1 personalization.

## Experiment inputs

The analysis uses each run's `training_log.json` under `models/ropg/`:
The labels are the current Luna era: `gpt-5.6-luna` judged `data/ropg_kd/*.jsonl`, with
`format_version: 4`, generated in commit `ff90cf6`. See [methodology](../methodology.md),
"Label eras — nano (retired) vs Luna (current)".

| Run | Training data | Anchor mode | Best epoch |
|---|---|---|---:|
| A | `ropg_kd`, filters off | `none` | 2 |
| B | `ropg_kd`, filters off | `both` | 1 |
| C | `ropg_kd_filtered`, filters on | `both` | 1 |
| D | `ropg_kd_filtered`, filters on | `doc_frozen` | 1 |

All runs use seed 42, `hard_neg` triplets, three epochs, learning rate $5\times10^{-5}$, and eight negatives. Each evaluation contains the same 276 query and persona rows. Their epoch-0 per-query vectors are identical, which establishes baseline parity for paired tests.

`benchmarks/compare_runs.py` compares best checkpoints with a paired bootstrap confidence interval and a paired sign-flip permutation test. It applies Holm correction across the 13 retrieval metrics. The intervals measure uncertainty across validation queries. Because all runs use one training seed, they do not measure variation across training seeds.

## Best-checkpoint results

Higher is better for every retrieval metric. `judged@5` is a coverage diagnostic rather than a direct quality score.

| Metric | Baseline (epoch 0) | Run A | Run B | Run C | Run D | Best |
|---|---:|---:|---:|---:|---:|---|
| nDCG@1 | 0.5895 | 0.7183 | 0.7122 | **0.7216** | 0.6746 | C |
| nDCG@2 | 0.5445 | 0.7113 | 0.7022 | **0.7229** | 0.6599 | C |
| nDCG@3 | 0.5596 | 0.7141 | 0.7103 | **0.7174** | 0.6715 | C |
| nDCG@4 | 0.5727 | 0.7211 | 0.7211 | **0.7283** | 0.6757 | C |
| nDCG@5 | 0.5861 | 0.7214 | **0.7307** | 0.7302 | 0.6872 | B |
| Hit@1 | 0.6920 | 0.7681 | 0.7754 | **0.7826** | 0.7391 | C |
| Hit@2 | 0.7645 | 0.8587 | 0.8659 | **0.8877** | 0.8551 | C |
| Hit@3 | 0.8623 | 0.8949 | **0.9130** | **0.9130** | 0.9022 | B and C |
| Hit@4 | 0.9058 | 0.9239 | **0.9565** | **0.9565** | 0.9312 | B and C |
| Hit@5 | 0.9275 | 0.9493 | **0.9710** | 0.9601 | 0.9493 | B |
| Recall@5 | 0.5640 | 0.5845 | **0.6461** | 0.6123 | 0.5761 | B |
| MRR | 0.7853 | 0.8434 | 0.8538 | **0.8596** | 0.8298 | C |
| judged@5 | 1.0000 | 0.7210 | **0.8688** | 0.8551 | 0.7775 | B |
Run B minus the Luna-era epoch-0 baseline, in table order: nDCG@1–5
+0.1227 / +0.1578 / +0.1507 / +0.1484 / +0.1446; Hit@1–5
+0.0833 / +0.1014 / +0.0507 / +0.0507 / +0.0435; Recall@5 +0.0821;
MRR +0.0685; judged@5 −0.1312. `judged@5` starts at 1.0 by construction because
the candidate pool was mined with the base encoder, so its decline is expected and
is not a regression.

## Metric definitions

### nDCG@K

Normalized Discounted Cumulative Gain measures graded relevance and ordering within the first $K$ results. This evaluation uses the teacher's relevance score as gain and discounts lower ranks logarithmically:

$$
\operatorname{DCG@K}=\sum_{i=1}^{K}\frac{\operatorname{relevance}_i}{\log_2(i+1)}
$$

Dividing DCG by the ideal ordering produces nDCG between 0 and 1. A score of 1 means the retriever returned the ideal graded ordering through position $K$. nDCG@1 evaluates only the first result. nDCG@5 evaluates the order and graded relevance of all five results.

Run C leads nDCG@1 through nDCG@4. Run B leads nDCG@5 by 0.0005. The B and C nDCG differences are statistically indistinguishable after Holm correction.

### Hit@K

Hit@K is the proportion of queries with at least one binary relevant document in the first $K$ results. A query scores 1 when the top $K$ contains a relevant document and 0 otherwise.

Run B's Hit@5 of 0.9710 means 97.10% of validation queries retrieve at least one relevant document in the first five results. Run C is stronger at Hit@1 and Hit@2. Run B is stronger at Hit@5.

### Recall@5

Recall@5 measures how much of a query's binary relevant set appears in the first five results:

$$
\operatorname{Recall@5}=\frac{\text{relevant documents retrieved in the top 5}}{\text{number of relevant documents}}
$$

Unlike Hit@5, Recall@5 rewards retrieving several relevant documents. Run B's 0.6461 means its top five contain an average of 64.61% of each query's relevant set.

Run B's Recall@5 advantage is statistically significant against every other arm:

| Comparison | Run B difference | 95% CI | Holm-adjusted p |
|---|---:|---:|---:|
| B minus A | +0.0616 | [+0.0411, +0.0833] | 0.0013 |
| B minus C | +0.0338 | [+0.0145, +0.0531] | 0.0052 |
| B minus D | +0.0701 | [+0.0471, +0.0930] | 0.0013 |

### MRR

Mean Reciprocal Rank measures how early the first relevant document appears. A first relevant result at rank 1 scores 1, rank 2 scores 0.5, and rank 5 scores 0.2. MRR averages that reciprocal rank across queries.

Run C has the highest MRR at 0.8596. Run B follows at 0.8538. Their difference of 0.0058 is not significant after Holm correction.

### judged@5

`judged@5` is the fraction of the returned top five that the teacher scored during dataset construction. Run B's 0.8688 means 86.88% of its top-five results come from the teacher-judged pool.

This value diagnoses evaluation coverage. It does not directly measure retrieval quality. The evaluator assigns no gain to unjudged documents, even when they may be relevant. nDCG and Recall therefore become harder to interpret as judged@5 falls.

### Training and validation loss

Training and validation loss measure how closely the student reproduces the teacher's target distribution on triplets. Lower loss does not guarantee better full-corpus retrieval.

Validation loss is comparable within A versus B and within C versus D. It is not comparable between an unfiltered and filtered arm because those arms score different validation triplet sets. The retrieval metrics remain paired-comparable because all four runs evaluate the same 276 scored query groups against the same corpus.

## Pairwise interpretation

### Run B versus Run C

Run C has higher raw nDCG@1 through nDCG@4, Hit@1, Hit@2, and MRR. None of those gains remains significant after Holm correction. Their nDCG@5 values are effectively tied at 0.7307 and 0.7302.

Run B has significantly higher Recall@5. It therefore retrieves more of the relevant set without a measurable loss in graded ranking quality. This supports selecting Run B.

### Run B versus Run A

Anchoring improves Recall@5 by 0.0616 and judged@5 by 0.1478. Both differences have Holm-adjusted $p=0.0013$. Other retrieval differences do not remain significant after correction.

### Run B versus Run D

Run D is significantly worse on nDCG@2 through nDCG@5, Recall@5, and judged@5. Its nDCG@5 is lower by 0.0435 and its Recall@5 is lower by 0.0701, both with Holm-adjusted $p=0.0013$. The query-only adapter with a frozen document tower does not beat joint adaptation here.

## Persona-swap control

The persona-swap control rotates every query to the wrong persona while retaining the same corpus embedding. A persona-sensitive retriever should degrade under this control.

Swapped minus matched nDCG@5 is:

| Run | nDCG@5 difference | Holm-adjusted p |
|---|---:|---:|
| A | -0.0017 | 1.0000 |
| B | -0.0043 | 1.0000 |
| C | -0.0053 | 1.0000 |
| D | +0.0008 | 1.0000 |

No run has a significant persona-swap effect on any metric. Run B is the best generic retriever, but these Stage 1 results do not demonstrate that the encoder uses the persona prefix.

## Artifacts

The pairwise reports, persona-swap reports, and plots are generated under the ignored local directory `models/ropg/comparisons/`:

- `best_metrics.svg`
- `training_curves.svg`
- `runA_vs_runB.md` through `runC_vs_runD.md`
- `runA_persona_swap.md` through `runD_persona_swap.md`

`models/` is excluded by `.gitignore`. This document contains the committed summary; model weights, checkpoints, raw logs, and local plots remain uncommitted.
