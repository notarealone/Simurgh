# Optimization Methods for Personalizing Large Language Models through Retrieval Augmentation

- **Authors:** Alireza Salemi, Surya Kallumadi, Hamed Zamani
- **Year:** 2024
- **Venue:** SIGIR 2024 (arXiv:2404.05970)
- **Link:** https://arxiv.org/abs/2404.05970

---

## Summary

The first study to train a *retriever* specifically for LLM personalization, instead of treating retrieval as a fixed, off-the-shelf component. Building on the retrieval-augmented personalization pipeline from the LaMP benchmark, the paper introduces two families of methods: **ROPG** (Retriever Optimization for Personalization) and **RSPG** (Retriever Selection for Personalization). The central claim is that retrieval feedback for personalization should come from the *downstream personalized-generation objective*, not from generic relevance, because the two signals diverge. The authors report statistically significant gains on six of seven LaMP tasks, with an average 5.5% SOTA improvement across datasets and 1.0%–33.8% improvement over a non-personalized FlanT5-XXL (average 15.3%).

## Setup: personalized RAG on LaMP

The pipeline they optimize decomposes into four pieces:

1. **Input.** A test input $x$ (e.g., an abstract, a movie description, an article body).
2. **User profile** $P_u$, defined as the user's own history of personal documents. The dataset is therefore a set of triples $(u, x, y)$: user, input, ground-truth personalized output.
3. **Retriever** $\mathcal{R}_\theta$ that selects the top-$k$ personal documents from $P_u$ most relevant to $x$. The base retriever is **Contriever** (Izacard et al. 2022), a pre-trained bi-encoder dense retriever; the ROPG paper fine-tunes it from Contriever's initial weights and serves retrieval with exact kNN over per-user indexes. The comparison retrievers in the LaMP sweep are **BM25**, **Recency** (return the most recent $k$ profile entries), and **no-retrieval** (no personalization). **Reciprocal Rank Fusion (RRF)** over all retrievers is added as a rank-fusion baseline.
4. **Frozen generator** $G$ (called $M$ in the paper). In every experiment the LLM is **FlanT5-XXL** (11B parameters, Chung et al. 2022), instruction-tuned, queried in zero-shot with beam size 4. The generator is **never updated** — all gains in the paper come from the retriever side. Max input/output length 512 tokens. Effective batch size 64 (8 accumulation steps × 8). Adam, learning rate $10^{-5}$, 5% linear warmup, gradient clipping at 1.0.

The query sent to the retriever is the non-template portion of the LLM input $x$. Retrieved entries are formatted into a prompt by the *Per-Profile-Entry Prompt* (PPEP) and *Aggregated Input Prompt* (AIP) templates (one per LaMP task; reproduced in Table 2 of the paper). In all experiments **four items** are retrieved from the user profile, and each profile document is prefixed with its date (`date: [date]`).

The **LaMP benchmark** (Salemi et al. 2023) provides seven personalization tasks across two families:

- **Personalized text classification** — LaMP-1 (Citation Identification, binary, Accuracy), LaMP-2 (Movie Tagging, 15 classes, Accuracy/F1), LaMP-3 (Product Rating, 1–5 ordinal, MAE/RMSE).
- **Personalized text generation** — LaMP-4 (News Headline Gen), LaMP-5 (Scholarly Title Gen), LaMP-6 (Email Subject Gen), LaMP-7 (Tweet Paraphrasing); metric ROUGE-1/ROUGE-L.

LaMP provides two split regimes (user-based and time-based). The ROPG paper uses **only the time-based separation**: each user's data is split by timestamp so the model must produce a personalized output for a test input using only the user's *earlier* documents. This is the regime that lets recency-based retrieval be a meaningful comparison. Statistical significance uses two-tailed paired $t$-test for ROUGE-1/ROUGE-L/MAE/RMSE and McNemar for Accuracy/F1, both at $p < 0.05$. All experiments run on a single A100 80GB + 128GB RAM.

## Method

The paper's contributions live above the LaMP pipeline: how to use the pipeline's own signals to update or select the retriever. Both ROPG variants are initialized from Contriever and trained with the **frozen LLM as the critic** — the only supervision is the LLM's downstream behavior on the task.

### ROPG-RL (reinforcement learning)

A vanilla **policy-gradient (REINFORCE)** method (Williams, 1992) in which the retriever is the policy.

- **Policy.** A single-action trajectory: for each query, sample *one* document $d$ from the user profile. The paper explored multi-document trajectories without replacement and reports no or little improvement over the single-action formulation at much higher cost. The probability of selecting document $d \in P_u$ is a softmax over the retriever's scores, restricted for efficiency to the top-$l=16$ documents according to the *initial* retriever weights:

  $$\pi_\theta(d \mid x) \;=\; \frac{\exp\!\bigl(\mathcal{R}_\theta(\phi_q(x), d)\bigr)}{\sum_{d' \in P_u^{\,l}} \exp\!\bigl(\mathcal{R}_\theta(\phi_q(x), d')\bigr)}.$$

- **Reward.** A per-document advantage with respect to the *un-fine-tuned* retriever's top-1 document $d_b$:

  $$\mathrm{Reward}(d; x, y) \;=\; \mathrm{Eval}\!\bigl(y, M(\phi_p(x, [d]))\bigr) \;-\; \mathrm{Eval}\!\bigl(y, M(\phi_p(x, [d_b]))\bigr).$$

  $\mathrm{Eval}$ is the LaMP-suggested task metric (Accuracy for LaMP-1/2, ROUGE-1 for LaMP-4–7). For **LaMP-3**, where the official metric is MAE (lower is better), the reward is remapped to "higher is better" by:

  $$\mathrm{Eval}_{\text{LaMP-3}}(y, \hat{y}) \;=\; \frac{\max(|1-y|,|5-y|) - \mathrm{MAE}(y, \hat{y})}{\max(|1-y|,|5-y|)},$$

  which yields 1 for a correct prediction and 0 for the worst-case prediction.

- **Objective.** REINFORCE with the per-document advantage above as the score function:

  $$\arg\max_\theta \;\frac{1}{|B|}\sum_{(u,x,y) \in B} \mathbb{E}_{d \sim \pi_\theta}\!\bigl[\mathrm{Reward}(d; x, y) \, \log \pi_\theta(d \mid x)\bigr].$$

  Because the reward is task-defined and may be non-differentiable, gradients do **not** propagate through the LLM — the only gradient signal is the policy-gradient on the retriever's distribution. The subtraction of the initial-retriever baseline $d_b$ reduces variance.

### ROPG-KD (knowledge distillation)

A **supervised distillation** alternative that re-frames the retriever's training signal as a distribution over the LLM's per-document utility, rather than as a single sampled action.

- **Teacher distribution.** For each candidate document $d \in P_u^l$ (again the top-$l=16$ by the initial retriever), the LLM is run on the prompt containing *only* that document, and the resulting task metric becomes the unnormalized logit. The target distribution over documents is a softmax of those per-document scores:

  $$p_t(d \mid x) \;=\; \frac{\exp\!\bigl(\mathrm{Eval}(y, M(\phi_p(x, [d])))\bigr)}{\sum_{d' \in P_u^{\,l}} \exp\!\bigl(\mathrm{Eval}(y, M(\phi_p(x, [d']))) \bigr)}.$$

  This is the LLM's view of "how useful each profile document is, relative to the others, for producing the personalized output" — the very signal a personalized-RAG retriever should encode.

- **Student loss.** The retriever's softmax $\pi_\theta(d \mid x)$ is trained to match $p_t$ via **KL divergence** (Eq. 4 of the paper), following the distillation pattern of Yang and Seo (2020):

  $$\arg\min_\theta \;\frac{1}{|B|}\sum_{(u,x,y) \in B}\sum_{d \in P_u^{\,l}} p_t(d \mid x)\,\log \frac{\pi_\theta(d \mid x)}{p_t(d \mid x)}.$$

The authors note two practical contrasts with ROPG-RL: (i) ROPG-KD considers the *relative* usefulness of profile items — a document that is "the least bad" still gets a non-zero target mass, whereas ROPG-RL's advantage can be negative and effectively punishes the policy; (ii) RL optimization is described as "less stable and more susceptible to overfitting" than the supervised distillation target, at the cost of one extra teacher LLM pass per training example.

### RSPG (retriever selection)

Motivated by a winning-rate analysis (Figure 3 in the paper) showing that **no retrieval method is best across all inputs on any task** — no-personalization wins 8.5%–18.4% of inputs, recency 15.2%–18.1%, and even ROPG-RL/ROPG-KD only 14.9%–20.2% — the authors train a **selection model** that picks, per input, among a fixed pool of retrievers $\mathbf{R} = \{\text{NoPersonalization}, \text{Recency}, \text{BM25}, \text{Contriever}, \text{ROPG-RL}, \text{ROPG-KD}\}$.

- **Selection model architecture.** An **encoder-only** model — **Longformer-base-4096** in the experiments — whose final-layer representation is fed into a linear projection that produces a scalar selection score for each retriever. Max input length is bumped to 1024 tokens to accommodate a prompt plus an LLM output. The two variants differ only in the input to the encoder:

  - **RSPG-Pre** (pre-generation): encoder input is the per-retriever personalized prompt $\phi_p(x, \mathbf{R}_i(\phi_q(x); P_u))$. The selected retriever's prompt is then fed to the LLM.
  - **RSPG-Post** (post-generation): the same prompt is **concatenated with the LLM's actual output** under that prompt before being scored. The selected retriever's already-generated output is returned as the final answer (no further LLM call).

- **Training objective.** Both variants are trained with **KL distillation from the LLM's downstream performance** (Eq. 5–7 of the paper), exactly the same recipe as ROPG-KD but with the softmax taken over *retrievers* rather than documents:

  $$P_{TS}(\mathbf{R}_i \mid x; u, y) \;=\; \frac{\exp\!\bigl(\mathrm{Eval}(y, M(\phi_p(x, \mathbf{R}_i(\phi_q(x); P_u))))\bigr)}{\sum_{j=1}^{|\mathbf{R}|} \exp\!\bigl(\mathrm{Eval}(y, M(\phi_p(x, \mathbf{R}_j(\phi_q(x); P_u))))\bigr)},$$

  $$P_{S_\omega}(\mathbf{R}_i \mid x) \;=\; \frac{\exp\!\bigl(\mathcal{S}_\omega(\mathbf{R}_i, x)\bigr)}{\sum_{j=1}^{|\mathbf{R}|} \exp\!\bigl(\mathcal{S}_\omega(\mathbf{R}_j, x)\bigr)},$$

  $$\mathcal{L} \;=\; \frac{1}{|B|}\sum_{(u,x,y) \in B}\sum_{i=1}^{|\mathbf{R}|} P_{TS}(\mathbf{R}_i \mid x; u, y)\,\log \frac{P_{S_\omega}(\mathbf{R}_i \mid x)}{P_{TS}(\mathbf{R}_i \mid x; u, y)}.$$

  The selection model is trained for 20 epochs (ROPG retrievers, 10 epochs). Baselines for the selection task are unsupervised Query Performance Prediction methods: **WIG**, **NQC**, $\sigma_{\max}$, $\sigma_{50\%}$, plus the random / reciprocal-rank / zero-score defaults required when the underlying retriever does not produce a score (Recency, No-Personalization).

The post-generation variant is more expensive (one LLM call per candidate retriever, totaling $|\mathbf{R}|$ LLM calls per input) but is the strongest method in the paper.

## Experiments & Results

All numbers below are from Table 3 of the ROPG paper (time-based LaMP split, FlanT5-XXL, $k=4$). Superscripts on the paper's numbers indicate significant improvement over No-Personalization, BM25, Recency, Contriever, and RRF respectively (all $p<0.05$).

| Task | Metric | No-Pers. | BM25 | Recency | Contriever | RRF | ROPG-RL | ROPG-KD | RSPG-Pre | RSPG-Post |
|---|---|---|---|---|---|---|---|---|---|---|
| LaMP-1 Citation ID | Acc ↑ | 0.502 | 0.626 | 0.622 | 0.636 | 0.570 | **0.655** | **0.668** | **0.663** | **0.672** |
| LaMP-2 Movie Tag | Acc ↑ | 0.359 | 0.387 | 0.377 | 0.396 | 0.375 | 0.391 | 0.396 | **0.405** | **0.430** |
| LaMP-2 Movie Tag | F1 ↑ | 0.276 | 0.306 | 0.295 | 0.304 | 0.299 | 0.300 | 0.306 | **0.314** | **0.339** |
| LaMP-3 Product Rate | MAE ↓ | 0.308 | 0.298 | 0.296 | 0.299 | 0.314 | **0.286** | **0.290** | **0.282** | **0.264** |
| LaMP-3 Product Rate | RMSE ↓ | 0.611 | 0.611 | 0.605 | 0.616 | 0.614 | **0.591** | **0.604** | **0.585** | **0.568** |
| LaMP-4 News Headline | R-1 ↑ | 0.176 | 0.186 | 0.189 | 0.183 | 0.190 | 0.191 | 0.187 | 0.190 | **0.203** |
| LaMP-4 News Headline | R-L ↑ | 0.160 | 0.171 | 0.173 | 0.169 | 0.176 | 0.177 | 0.172 | 0.176 | **0.186** |
| LaMP-5 Scholarly Title | R-1 ↑ | 0.478 | 0.477 | 0.475 | **0.483** | 0.478 | 0.475 | 0.477 | 0.483 | 0.480 |
| LaMP-5 Scholarly Title | R-L ↑ | 0.428 | 0.427 | 0.426 | **0.433** | 0.428 | 0.427 | 0.428 | 0.431 | 0.429 |
| LaMP-6 Email Subject | R-1 ↑ | 0.335 | 0.412 | 0.403 | 0.401 | 0.394 | 0.394 | **0.415** | **0.426** | **0.433** |
| LaMP-6 Email Subject | R-L ↑ | 0.319 | 0.398 | 0.389 | 0.386 | 0.381 | 0.381 | **0.400** | **0.411** | **0.418** |
| LaMP-7 Tweet Para. | R-1 ↑ | 0.449 | 0.446 | 0.444 | 0.440 | 0.446 | 0.448 | 0.441 | **0.450** | **0.461** |
| LaMP-7 Tweet Para. | R-L ↑ | 0.396 | 0.394 | 0.393 | 0.390 | 0.395 | 0.397 | 0.391 | **0.400** | **0.409** |

Headline takeaways (from the paper, not derived here):

- **Best method overall:** **RSPG-Post** is best on 6 of 7 LaMP datasets (all except LaMP-5, where Contriever wins). The gains are statistically significant over every baseline on those six datasets.
- **Average SOTA gain:** RSPG-Post improves over the previous state of the art by an average of **5.5%** across all seven LaMP datasets.
- **Gain over non-personalized LLM:** **1.0%–33.8%** per task, **average 15.3%**.
- **No-personalization ceiling:** No personalized baseline (BM25, Recency, Contriever, RRF, or even ROPG-RL/KD alone) can beat FlanT5-XXL on LaMP-7. Only RSPG-Pre and RSPG-Post do.
- **ROPG-RL vs. Contriever (the fine-tuning start point):** ROPG-RL improves over Contriever on LaMP-1, LaMP-3, LaMP-4, and LaMP-7. ROPG-KD additionally improves Contriever on LaMP-6. ROPG-KD beats ROPG-RL specifically on the binary-feedback tasks LaMP-1 and LaMP-2; otherwise there is no clear winner.
- **Selection-model success rate** (Table 4): RSPG-Pre and RSPG-Post select the best retriever >80% of the time on all classification tasks (LaMP-1/2/3) and on LaMP-7; 40%–65% on the harder generation tasks LaMP-4/5/6. RSPG-Post beats RSPG-Pre on 6/7 tasks, supporting the value of conditioning on the LLM's actual output.
- **Ablation — ROPG inside RSPG** (Table 5): removing the ROPG-trained retrievers from the pool and keeping only No-Pers./Recency/BM25/Contriever degrades both RSPG-Pre and RSPG-Post on essentially every dataset, showing the gains are not purely from the selection model. The gap to the Oracle upper-bound also remains non-trivial: best method reaches 68%–98% of the oracle depending on task (68% LaMP-3, 75% LaMP-4, 84%–98% on the rest).

## Key Contributions

- **First retriever optimization for LLM personalization** — moves beyond frozen retrievers in personalized RAG.
- **ROPG — two optimization signals for the retriever:**
  - **ROPG-RL:** vanilla policy gradient (REINFORCE) with a task-metric reward and a baseline = the un-fine-tuned retriever's top-1; supports arbitrary non-differentiable metrics via a per-task remap.
  - **ROPG-KD:** distills the LLM's per-document utility (softmax over per-document Eval scores) into the retriever's softmax via KL divergence, following Yang and Seo (2020).
- **RSPG — retriever selection:**
  - **RSPG-Pre:** per-input, pre-generation selection from a pool of retrievers via a Longformer encoder over the per-retriever prompt.
  - **RSPG-Post:** per-input, post-generation selection that scores each (prompt, LLM output) pair — strongest variant, best on 6/7 tasks.
- **Empirical validation on LaMP** (seven personalization tasks, time-based split), with statistically significant improvements on six of seven tasks and an Oracle gap analysis showing headroom remains on the harder generation tasks.
- **Establishes that retrieval feedback for personalization should come from the personalized objective**, since generic relevance and personal usefulness diverge. This is positioned as the *method-level* lesson: any personalized-RAG retriever should be trained against the downstream personalized signal rather than a generic relevance proxy.

## What we take from it (general lessons)

- **Train the retriever on the downstream personalized objective, not on generic relevance.** Relevance and personal-usefulness are not the same signal, and optimizing one does not deliver the other. When a personalized judge is available, it should be the source of retrieval feedback. Standard learning-to-rank supervision is not available here because user feedback in personalized text generation is the *output text* $y$ per input, not per-document relevance.
- **Distillation can be a cheaper, more stable alternative to online RL for retriever training.** A frozen LLM's per-document task scores give a supervised KL target that is easier to optimize than REINFORCE on a sparse, possibly non-differentiable downstream metric. ROPG-KD specifically avoids the failure mode where the policy is punished for "the least bad" document, and the authors flag RL training as less stable and more overfit-prone. The cost is one teacher LLM pass per profile document per training example.
- **Per-input (per-user) selection over a pool of retrieval strategies can beat committing to one retriever globally.** No single retriever (or single $k$) is best across inputs on any LaMP task; a learned selection model, especially with post-generation re-ranking, is a clean way to combine BM25 / dense / recency / personalized retrievers, including the option to return "no personalization" per input.
- **A frozen generator is enough to make retriever optimization meaningful.** The ROPG gains come from updating only the retriever; the personalization signal lives in the *selection of which documents the LLM sees*, not in the LLM's weights. This is also why the cost stays bounded: training touches a Contriever-sized encoder and (in RSPG) a Longformer, not a 11B generator.
- **Reward-hacking risk applies to the retriever too.** The Eval that trains the retriever is also the metric the system is judged on. A retriever optimized against a downstream metric can learn to satisfy the judge's surface preferences rather than fetching the documents the user actually needs. Guarding with retrieval-side metrics (Recall@K, MRR) and per-input selection-skill diagnostics (the paper's success-rate table is one such guard) is what keeps the optimization honest.
- **Approximating the policy over the top-$l$ profile documents keeps both ROPG-RL and ROPG-KD tractable** as profile size grows. Hierarchical softmax is a drop-in alternative.

## Notes

- LaMP personalizes from each user's own historical documents; the *user profile* is the user's prior (input, output) pairs, not a declared attribute vector. Method transfers, but the LaMP setup itself is English, short-output, and benchmark-specific.
- Both ROPG-RL and ROPG-KD are fine-tunings of an existing dense retriever (Contriever); they are not architectures in their own right and can be dropped onto any bi-encoder.
- The retriever pool considered in the RSPG experiments is exactly: {No-Personalization, Recency, BM25, Contriever, ROPG-RL, ROPG-KD}. RRF is the only fusion baseline; the paper does not consider other fusion rules.
- The evaluation function $\mathrm{Eval}$ is *the same metric the system is judged on*. For LaMP-3 the MAE is explicitly remapped to a "higher is better" reward; the remap uses the LaMP-defined worst-case $\max(|1-y|, |5-y|)$.
- RSPG-Post is more expensive than RSPG-Pre by a factor of $|\mathbf{R}|$ LLM calls per input; the paper does not report latency or cost-vs-quality trade-offs explicitly.
- The same authors' follow-ups (Stochastic RAG, LaMP-QA) extend this line and are worth tracking for the retriever-optimization section of the literature review.
