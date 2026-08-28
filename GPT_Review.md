# ROPG Training and Validation Audit

## Scope and executive verdict

This is a **static, read-only audit** of the repository. No local training,
evaluation, data generation, notebook, test, build, or other command was run as
part of this review.

The reported two-T4 runs do **not** reveal a fatal gradient, AMP, or DDP defect.
The implementation uses torchrun environment ranks, one process per GPU,
`DistributedDataParallel`, distributed samplers, `set_epoch`, all-reduced
validation loss, non-reentrant gradient checkpointing, PEFT input gradients,
AMP scaler skipped-step handling, and rank-0 corpus evaluation/checkpointing
(`src/rl/ropg_kd.py:28-43`, `267-333`, `600-673`, and the DDP/training
orchestration around `1194-1750`). The principal validity risks are upstream
labels and evaluation design, not a demonstrated distributed-training failure.

## Prioritized findings

| Priority | Finding | Status | Consequence |
|---|---|---|---|
| P0 | Questions are incompletely constructed: retrieval and judging receive only a stem. | **Implemented in this change** | Many items are ambiguous or unanswerable; existing judged ROPG artifacts are invalid. |
| P0 | Malformed judge output and API exceptions are converted into zero scores. | **Implemented in this change** | Failed calls silently manufacture negatives and can select fake positives. |
| P1 | Splits are question-level rather than demonstrably source/group-level. | Defer | Validation/test independence and shared-passage isolation are not guaranteed. |
| P1 | The teacher signal is a single sampled subjective usefulness judgment. | Defer | Labels are noisy and not the deterministic answer-grounded reader reward described by original ROPG. |
| P1 | The objective allocates signal to noisy middle ranks and hard-negative positives can be arbitrary. | Defer | Filtering or objective changes could alter conclusions without repairing label validity. |
| P1 | Evaluation judges only retrieved candidate pools while ranking the full corpus. | Defer | Unjudged relevant chunks receive zero gain; `judged@K` is diagnostic, not corrective. |
| P1 | Run B changes multiple factors, and the live filtering thresholds contradict the experiment document. | Defer | Anchoring and filtering effects cannot be isolated or compared cleanly. |
| P1 | Checkpoint selection and headline metric governance are not frozen before observation. | Defer | `checkpoint-best` can still be worse than epoch 0 on the declared criterion. |
| P1 | The design promises paired inference and three seeds, but Run B has one seed and no significance result. | Defer | Current claims are provisional. |
| P2 | Personalization and serving/reporting claims have unresolved generalization and index-contract risks. | Defer | Generic retrieval or index mismatch can masquerade as personalization or invalidate serving results. |

## Approved implementation scope (two validity fixes plus subsequent execution/index-contract safeguard; pending Kaggle verification)

The two validity fixes—complete-question rendering and safe judge retries/group skips—are
implemented in the source, configuration, and notebook paths. A subsequent approval authorized
the earlier execution/index-contract safeguard of `max_seq_length: 2048` parity across
generation, training, and index/serving settings; it is implemented in the source, configuration,
and notebook paths. The later GPT-5.6 Luna judge migration changes teacher scores, makes scored
format version 4 current, and requires fresh scored artifacts; version-3 scored and derived data
are incompatible. No local execution occurred, so Kaggle execution and acceptance verification
remain pending.

A later static consistency pass fixed the generator's `OpenAICompatClient`
constructor keyword, made triplet derivation reject non-version-4 scored rows
before replacing derived files, and added `notebooks/build_gen_ropg_data.py` to
generate the standalone notebook from the production module. These are execution
and drift safeguards; they do not change the deferred experimental scope below.

### 1. Render complete questions before retrieval and judging

During the audit, the pre-fix implementation at the then-current
`src/data/gen_ropg_data.py:131-140` returned only `stem`, and that stem alone
entered embedding/retrieval and judging at the then-current `:228-257`. The
current replacement uses `QuestionContext` at `src/data/gen_ropg_data.py:31-37`,
strict parser/judge context at `:55-106`, retries at `:109-200`, and the complete
renderer/loader at `:218-315`; retry configuration is validated at `:332-360`,
and resume/group-skip/output handling is at `:402-520`. The schema
requires shared passages and type-specific options at
`docs/question-extraction.md:32-78,144-147`. A query must therefore contain,
in the appropriate rendered form, the shared passage plus stem plus options;
equivalent pairs/items must also be included when present. It must never
contain the answer or explanation.

Gold answer and explanation may be supplied separately to the judge as rubric
context, but they must not be persisted in or embedded as the retrieval query.
This is a validity fix, not merely a formatting improvement: without the
complete question, many questions cannot be answered from the query and the
existing judged ROPG rows cannot be treated as valid training evidence.

Complete rendering and safe failures are mirrored in
`notebooks/gen_ropg_data.ipynb`, which can now be regenerated from the production
module; retry policy values are synchronized in both datagen YAMLs. No other
experimental scope is approved by this audit.

### 2. Retry judge failures and skip failed groups safely

During the audit, the pre-fix implementation converted malformed judge output
to `0` at the then-current `src/data/gen_ropg_data.py:42-53`, and any API
exception to `0` at the then-current `:78-92`. This silently manufactured
negatives and could cause fake positives to be selected. The current replacement
uses strict parsing/judging at `:55-106`, retries at `:109-200`, retry
configuration validation at `:332-360`, and resume/group-skip/output handling at
`:402-520`, so exhausted failures emit no partial row and do not substitute
`teacher_score=0.0`.

Approved behavior is:

1. Validate every response strictly; malformed output is a failed attempt,
   not a score of zero.
2. Retry both API failures and malformed-output failures with bounded
   exponential backoff.
3. After retry exhaustion, log the failure and skip the entire
   `(question, persona)` group: emit no partial row and never substitute
   `teacher_score=0.0`.
4. Keep successful rows append/resume-safe. A failed group must remain
   retryable on a later rerun rather than being represented as a completed
   zero-scored group.

These two fixes are implemented consistently in the source and notebook paths;
the retry policy values are synchronized in both datagen configurations. They do
not authorize relabeling,
filtering, objective redesign, split rebuilding, or evaluation changes.

## Detailed evidence and deferred recommendations

### P1 — Source-level split leakage

`data/splits/train_qids.txt`, `val_qids.txt`, and `test_qids.txt` contain
different questions from the same exam/source files, including both
`ai_generated_questions` and real exams. This contradicts the source-level
split promise in `docs/question-extraction.md:17,47` and
`docs/experiment-design.md:68-73`. As a result, validation/test independence
and shared-passage isolation are not guaranteed.

**Deferred recommendation:** rebuild and freeze source/group-aware splits.
Do not present the current question-level split as evidence of source-level
generalization.

### P1 — Noisy and misaligned teacher signal

The teacher directly samples subjective usefulness scores once at
`temperature: 1.0` (`configs/datagen_ropg.yaml:9-13` and
`_build_judge_messages`/scoring in `src/data/gen_ropg_data.py:68-200`). That is
not the deterministic,
answer-grounded reader reward described by original ROPG. The repository's
measurements and discussion report single-label reliability of approximately
`0.17`, rank-1/rank-2 ties, weak-best groups, and rank-extreme versus middle
gaps (`docs/methodology.md:177-239`).

Existing data must be regenerated after the complete-question fix. Changes to
teacher design, repeated labeling, and calibration are deferred; the audit
does not claim that the current subjective score is a valid reader reward.

### P1 — Objective mismatch and noisy positives

The listwise `reader_kd` path spends signal on below-noise middle ranks while
`hard_neg` uses rank 1 versus the tail (`src/rl/ropg_kd.py:418-471`,
`configs/train_ropg.yaml:8-15`, and `docs/methodology.md:201-230`). Even where
hard-negative positives are selected, rank-1 can be arbitrary in near-tie or
weak-pool groups.

Keep filters and objective redesign deferred. Filtering must not be described
as solving label validity: it changes the observed training population but
cannot repair incomplete queries or false-zero judgments.

### P1 — Validation-pool and circularity bias

Evaluation ranks the full corpus, but supplies gain/relevance only for each
group's approximately 20 candidates judged after retrieval by the base system
(`src/rl/ropg_kd.py:849-921,968-999`; see also
`docs/methodology.md:127-174`). Unjudged-but-relevant chunks therefore act as
zero-gain items. `judged@K` is a diagnostic of coverage, not a correction for
this bias. Independent gold relevance labels and independent end-to-end
accuracy remain necessary.

### P1 — Experimental confounds and threshold contradiction

Run B changed learning rate, epochs, negatives, and anchoring together
(`docs/results/stage1-ropg-runB.md:1-15`), so it cannot isolate anchoring.
Filtering changes both label composition and the number of training rows, and
validation loss is not cross-arm comparable under those changes.

There is also a live threshold contradiction: `docs/experiment-design.md:169-171`
states `0.08/0.4/0.3`, while `configs/datagen_ropg_filtered.yaml:67-75`
contains `0.05/0.4/0.2`.

**Deferred recommendation:** use factorial and equal-size controls, and
reconcile configs with documentation before making causal or filtering claims.

### P1 — Checkpoint and metric governance

The code selects checkpoints by Recall@K (`is_better` near
`src/rl/ropg_kd.py:1061`, with checkpoint logic near `1474-1668`), while the
headline metric was revised after observing Run B
(`docs/experiment-design.md:118-153` and
`docs/results/stage1-ropg-runB.md:50-64`). A trained `checkpoint-best` can
therefore still be inferior to epoch 0 on a subsequently declared criterion.

**Deferred recommendation:** freeze predeclared selection criteria and require
explicit baseline gating before shipping any checkpoint.

### P1 — Statistics and independent seeds

The experiment design promises paired inference and at least three seeds
(`docs/experiment-design.md:56-62`), but Run B is seed 42 and significance
remains uncomputed (`docs/results/stage1-ropg-runB.md:80-93`).
`benchmarks/compare_runs.py` supports paired bootstrap, sign-flip, and Holm
correction, but those methods do not replace multiple independent seeds.
Claims based on the current run remain provisional.

**Deferred recommendation:** complete the promised independent-seed study and
paired statistical analysis before treating small metric differences as
evidence.

### P2 — Personalization and generalization

Because the same shared corpus is used, persona learning can amount only to
reordering; generic retrieval gains can masquerade as personalization. Persona
swap was added after Run B and is absent from that run
(`docs/results/stage1-ropg-runB.md:66-78`). Validation uses train personas, so
held-out-persona and test-persona claims are not established. The evaluation
path's group/persona handling is relevant at `src/rl/ropg_kd.py:905-917`, and
the methodological limitation is discussed at `docs/methodology.md:297-305`.

**Deferred recommendation:** establish held-out-persona and persona-swap
evidence before claiming personalization rather than generic retrieval.

### P2 — Serving and reporting contract

`doc_frozen` changes the index/query serving contract
(`configs/train_ropg.yaml:126-140`; `src/rl/ropg_kd.py:813-843,863-871`).
Index provenance and query/index parity must therefore be enforced.

Reports should name exact config, data, index, and model versions. The audit
also flags stale or contradictory status documentation where appropriate,
including the source-split promise versus the actual split, the threshold
mismatch, and the historical metric revision.

**Deferred scope:** serving/report cleanup and provenance enforcement remain
deferred, as do split rebuilding, relabeling strategy, noisy-positive and
filter/objective work, independent relevance labels, experimental redesign,
checkpoint policy, and statistics.

## Regeneration warning and acceptance criteria

The old scored `{train,val}.jsonl` files and all derived triplets/pairs are
invalid after the query changes. `--derive-only` cannot repair them. Generate
into a fresh output directory on Kaggle; retain no mixed old/new rows. Before
using the regenerated data, confirm all of the following:

- persisted retrieval queries contain the full context (shared passage, stem,
  options, and equivalent pair/item content when applicable) but no gold
  answer or explanation;
- malformed responses and API failures are retried with bounded exponential
  backoff;
- exhausted failures are logged and emit no row, no partial score, and no
  `teacher_score=0.0` substitute for the affected `(question, persona)` group;
- successful rows remain append/resume-safe and failed groups remain retryable;
- downstream triplets/pairs are derived only from the newly scored output.

## Final disposition

Implementation is complete in code/config/notebook for complete-question
rendering and safe judge retries/group skip. All runtime acceptance checks and
data regeneration are still to be done on Kaggle. Defer all other
recommendations listed above until the data validity, split, evaluation,
experimental, metric, statistical, personalization, and serving questions are
resolved. No commands were executed as part of this audit.
