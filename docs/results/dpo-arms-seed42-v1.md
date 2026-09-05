# DPO rewriter arms, seed 42 v1

> **Superseded by [dpo-arms-seed42-v2](dpo-arms-seed42-v2.md)**, which reruns these three
> arms on v4 pair data with the three changes below applied. The diagnosis in this document
> held: likelihood displacement was the binding failure, and removing it moved `wpo` above
> the untrained policy.

## Decision

No arm is promoted. Stage 2 DPO as configured for this screening run does not improve the
query rewriter: `dpo` is statistically indistinguishable from the untrained `base_qwen`
policy, and `wpo` and `robust_dpo` are worse than it. The teacher that produced the
training pairs, `grok-4-1-fast`, beats every trained arm by a wide margin.

The tournament reported `selected_variant: wpo`. That field answers a narrower question —
which of the two *variant* arms should join `dpo` at the replication seeds — and it must
not be read as an endorsement. Read against the full field, it names the second-worst
candidate.

The training logs explain the ranking completely, and the explanation is not about the
three objectives. It is about which epoch each arm promoted. Three configuration changes
follow from that and are recorded in [Changes adopted](#changes-adopted).

## Experiment inputs

| Input | Value |
|---|---|
| Config | `configs/train_dpo.yaml` as of commit-time values below; the file has since changed |
| Policy and reference | `Qwen/Qwen3-4B`, 4-bit, LoRA r=16 on all seven projections |
| Preference data | `data/dpo/train.jsonl` 1,739 rows, `data/dpo/val.jsonl` 351 rows |
| Candidate source | `grok-4-1-fast` rewrites; labels from `gpt-5.6-luna` |
| Training | 3 epochs, 654 optimizer steps, global batch 8, lr $1\times10^{-5}$, $\beta=0.1$ |
| Checkpoint selection | `eval_loss`, `greater_is_better: false` |
| Judge | `gemini-3.5-flash-lite` over the native GenAI protocol, `thinking_level: minimal` |
| Tournament | 272 validation prompts, 5 candidates, 10 pairs, 2,720 jobs, seed 42 |

The judge completed 2,720 of 2,720 jobs with `missing: 0`, so no result below is
contaminated by judge failures. Step-10 `logps/chosen` is −94.143, −94.129 and −94.137
across the three arms, which establishes that all three saw identical data in identical
order under identical seeding.

## Tournament results

Score is $(\text{wins} + \tfrac{1}{2}\text{ties})/n$ over $n = 272$ prompts. 0.5 is parity.
Round-robin is the mean over the four pairings each candidate appears in.

| Candidate | Round-robin | vs. `base_qwen` | Holm $p$ |
|---|---:|---:|---:|
| `grok` | **0.7054** | 0.688 | 4.0e-9 |
| `dpo` | 0.4922 | 0.518 | 0.502 |
| `base_qwen` | 0.4848 | — | — |
| `wpo` | 0.4343 | 0.449 | 0.141 |
| `robust_dpo` | 0.3833 | 0.406 | 0.006 |

Every pairing, left-relative:

| Pairing | Score | Ties | Holm $p$ |
|---|---:|---:|---:|
| `dpo` vs `robust_dpo` | 0.572 | 55 | 0.039 |
| `dpo` vs `wpo` | 0.566 | 100 | 0.037 |
| `wpo` vs `robust_dpo` | 0.553 | 73 | 0.141 |
| `dpo` vs `base_qwen` | 0.518 | 92 | 0.502 |
| `wpo` vs `base_qwen` | 0.449 | 72 | 0.141 |
| `robust_dpo` vs `base_qwen` | 0.406 | 39 | 0.006 |
| `base_qwen` vs `grok` | 0.312 | 2 | 4.0e-9 |
| `dpo` vs `grok` | 0.312 | 2 | 4.0e-9 |
| `wpo` vs `grok` | 0.301 | 2 | 3.7e-10 |
| `robust_dpo` vs `grok` | 0.252 | 1 | 8.8e-16 |

Two rows deserve comment. `dpo` vs `base_qwen` at $p = 0.502$ with a 95% interval of
[0.469, 0.566] is a clean null; the interval's half-width of 0.049 also fixes the
resolution of this design, which cannot see effects smaller than roughly five points at
$n = 272$. And `dpo` vs `grok` is numerically identical to `base_qwen` vs `grok`
(84/186/2). With a 34% tie rate between `dpo` and `base_qwen` that is plausible chance,
and the 180 decisive verdicts in that pairing prove the adapter was loaded and did change
the generated text.

Per-persona scores, averaged over each candidate's four pairings:

| Candidate | crammer | scholar | steady |
|---|---:|---:|---:|
| `grok` | 0.8214 | 0.7799 | 0.5098 |
| `base_qwen` | 0.4368 | 0.4674 | 0.5520 |
| `dpo` | 0.4396 | 0.4891 | 0.5492 |
| `wpo` | 0.4093 | 0.3832 | 0.5126 |
| `robust_dpo` | 0.3929 | 0.3804 | 0.3764 |

The teacher's advantage is concentrated, not uniform: it dominates crammer and scholar but
draws level with an untrained Qwen on steady. Whatever `grok` does better is specific to
the crammer and scholar rewrite styles. Every Qwen-family arm peaks on steady, and
`robust_dpo` is the only candidate that is flat across personas, which is uniform
degradation rather than a trade-off.

## Diagnosis from the training logs

Each arm promoted a different epoch, and that single fact orders the tournament.
$\Delta\log p$ is recovered from the logged reward as $\text{reward}/\beta$ with
$\beta = 0.1$, so it is the total shift in the sequence log-probability against the frozen
reference, in nats.

| Arm | Promoted | Epoch | $\Delta\log p$(chosen) | $\Delta\log p$(rejected) | eval acc | Round-robin |
|---|---|---:|---:|---:|---:|---:|
| `base_qwen` | — | 0 | 0 | 0 | — | 0.4848 |
| `dpo` | `checkpoint-218` | 1 | −0.8 | −5.5 | 0.6903 | 0.4922 |
| `wpo` | `checkpoint-654` | 3 | −8.8 | −14.2 | 0.6392 | 0.4343 |
| `robust_dpo` | `checkpoint-436` | 2 | −14.4 | −24.4 | 0.6534 | 0.3833 |

Pearson $r$ between chosen-side displacement and tournament score is 0.991 over these four
points, with a slope of 0.0073 tournament points lost per nat. Training subtracted quality
in proportion to how far it dragged the chosen response down, and the arm that dragged it
least won.

### The margin grew for the wrong reason

The eval margin never came from making the preferred rewrite more likely. Decomposing each
promoted checkpoint's margin into its chosen and rejected contributions:

| Arm | Margin | From chosen | From rejected |
|---|---:|---:|---:|
| `dpo` | 0.468 | −18% | +118% |
| `wpo` | 0.545 | −161% | +261% |
| `robust_dpo` | 0.993 | −145% | +245% |

In all three arms the chosen side moves the wrong way and more than the whole separation
comes from suppressing the rejected response. This is likelihood displacement, and it is
the expected outcome of a pure margin objective on pairs that are entirely off-policy: both
responses are `grok` text that the `Qwen3-4B` policy never generated, so nothing in the
objective requires the policy to move toward the chosen response. The model learned to
avoid one style without learning to produce the other.

### `eval_loss` is not a valid selector for two of the three arms

Line references are to TRL 0.24.0, the pinned version, in `trl/trainer/dpo_trainer.py`.

**`robust` is unbounded below** (lines 1116–1120):

$$\mathcal{L}_{\text{robust}} = \frac{-(1-\varepsilon)\log\sigma(\beta\Delta) + \varepsilon\log\sigma(-\beta\Delta)}{1-2\varepsilon}$$

As $\Delta \to \infty$ the second term diverges to $-\infty$. This is a bias correction, not
a proper loss, and minimizing it rewards unbounded margin growth. The log confirms it: the
`robust_dpo` training loss turns negative from step 460 and reaches −0.093, with margins at
2.99.

**`wpo`'s loss carries a policy-dependent scale factor** (lines 1653–1662, 1775–1776). The
WPO weight is $\exp(\bar{\ell}_{\text{chosen}} + \bar{\ell}_{\text{rejected}})$ over
length-normalized adjusted log-probabilities, clamped at 1, and it multiplies the
per-sample loss. Two independent estimates put it near 0.11 on this data: the step-10 loss
ratio $0.0763/0.6937 = 0.110$ and the step-10 gradient-norm ratio
$1.175/9.767 = 0.120$. The weight is computed under `no_grad`, so training does not game it
— but checkpoint selection reads the weighted number, and the weight shrinks as the
policy's own log-probabilities fall. The metric therefore rewards a less confident policy:

```
wpo  eval_loss  0.0553 -> 0.0531 -> 0.0486    monotone down, selects epoch 3
wpo  eval acc   0.6648 -> 0.6449 -> 0.6392    monotone down
```

Selection ran opposite to preference accuracy at every epoch and promoted the arm's
most-degraded checkpoint. `dpo` escaped only because for the sigmoid loss the two metrics
happened to agree.

The three arms' best `eval_loss` values are 0.5957, 0.5350 and 0.0486. They differ by more
than an order of magnitude purely from these rescalings, which is why cross-arm ranking was
routed through the judge tournament rather than the loss. That design choice is now
supported by measurement rather than assumption.

### Epoch 1 is the accuracy peak in every arm

| Arm | Epoch 1 | Epoch 2 | Epoch 3 | Selected by `eval_loss` | Best by accuracy |
|---|---:|---:|---:|---:|---:|
| `dpo` | **0.6903** | 0.6648 | 0.6648 | 1 | 1 |
| `robust_dpo` | **0.6733** | 0.6534 | 0.6449 | 2 | 1 |
| `wpo` | **0.6648** | 0.6449 | 0.6392 | 3 | 1 |

Validation preference accuracy declines monotonically in all three arms while late-training
batch accuracy reaches 0.91 to 0.99. Two of the three epochs are memorization, and they
cost two thirds of the GPU budget per arm (roughly 77 minutes per epoch on 2×T4).

The peak itself is modest. 0.69 is the best any arm reaches at telling Luna's chosen
rewrite from its rejected one on held-out data, achieved after a single epoch.

## Changes adopted

Three changes follow directly from the diagnosis and are applied in `configs/train_dpo.yaml`,
`src/rl/dpo_train.py` and the generated notebook.

1. **Select on `eval_rewards/accuracies`, maximizing.** This removes both pathologies above:
   the metric is bounded, comparable across epochs within an arm, and monotone in the
   quantity the objective is supposed to improve. Applied to this run's logs it would have
   promoted epoch 1 in all three arms instead of epochs 1, 2 and 3. Combined with change 2
   there is one checkpoint per arm, so the new metric decides nothing on the next run; it is
   in place so that raising the epoch count later cannot silently reintroduce either
   pathology.
2. **One epoch.** Epoch 1 is the accuracy peak everywhere, and later epochs both degrade
   validation accuracy and deepen the chosen-side displacement that predicts tournament
   loss.
3. **Add the RPO supervised anchor, `rpo_alpha: 1.0`.** TRL adds
   $\alpha \cdot \text{NLL}(\text{chosen})$ to the preference loss
   (`dpo_trainer.py:1772-1773`), computed over completion tokens only because prompt and
   padding positions are zeroed and skipped via `ignore_index=0` (lines 1635, 1670–1672).
   This is the direct counter to likelihood displacement: it makes the objective "imitate
   the chosen rewrite *and* separate it from the rejected one" rather than separation alone,
   which is the right shape for a much stronger off-policy teacher.

Two consequences of change 3 are worth recording because they are not symmetric across arms.
TRL applies the anchor before the WPO weight (`losses + rpo_alpha * nll_loss`, then
`losses * policy_weights`, lines 1772–1776), so at equal `rpo_alpha` the anchor is roughly
nine times weaker in the `wpo` arm than in the other two on this data. No numeric
compensation is applied; the weight varies per batch and over training, so scaling
`rpo_alpha` to cancel it would be unstable. The asymmetry is declared per arm in the config
instead. Separately, TRL's `ignore_index=0` drops any genuine token with id 0 from the NLL,
which for Qwen3 is `!`; this is negligible for query rewrites but is a real quirk of the
implementation.

The $\beta$ setting and an explicit chosen-log-probability guardrail were considered and
deliberately not adopted in this round, to keep the next run's change set small enough to
attribute.

## What this does not settle

The expected gain from the three changes is parity, not a win. At epoch 1 all three arms
sit at −0.8, −0.3 and −0.1 nats of chosen displacement, which is near-`base_qwen` by
construction. Nothing in this sweep reaches above the untrained policy, so the reachable
operating points are "close to base" and "worse than base."

Two structural questions remain open and matter more than any hyperparameter:

- **Is a margin objective the right tool at all?** The evidence says the teacher gap is not
  closable this way, and that knowledge distillation or supervised fine-tuning on the chosen
  `grok` rewrites is the appropriate method — the same approach Stage 1 already uses for the
  retriever.
- ~~**Do Luna's labels predict Gemini's judgments?**~~ **Answered** in
  [judge-agreement-luna-gemini-v1](judge-agreement-luna-gemini-v1.md): two Gemini judges
  agree with the Luna labels on 65–70% of decisive verdicts, and agree with *each other* on
  73.0% of the same items. The labels are not the problem — the rubric leaves ~30% of pairs
  undetermined for any judge, which sets a noise floor the tournament's ±5-point resolution
  cannot see past. This caps the achievable score; it does not cause the ranking above.

## Reproduction

| Artifact | Location |
|---|---|
| Tournament summary and per-job judgments | `models/dpo/oldData-dpo-seed-42/comparisons/seed-42/` |
| Per-arm metrics and manifests | `models/dpo/oldData-dpo-seed-42/runs/{arm}/seed-42/` |
| Notebook that produced the run | `notebooks/train_dpo.ipynb`, built by `notebooks/build_train_dpo.py` |

`models/` is gitignored, so the run tree above — weights, checkpoints, judgment dumps — is
local only. This document is the committed summary; the v2 comparison additionally commits
regenerable figures and a `stats.json`.
