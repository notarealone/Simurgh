# Data Design: Training Data for ROPG-KD and DPO Rewriter

## 1. Overview

The Simurgh pipeline trains two components beyond the prompted baseline:

1. **ROPG-KD** — fine-tunes the Qwen3-Embedding retriever to rank documents by
   *pedagogical utility* for a given learner profile, using Knowledge Distillation
   from an LLM judge as the teacher.
2. **DPO Rewriter** — fine-tunes the query rewriter to produce rewrites that surface
   the right depth and style of passage for each persona, using Direct Preference
   Optimization.

Each requires a different dataset format, derived from the same source corpus and
question bank. This document specifies what each dataset looks like, why it looks
that way, and how it is generated.

---

## 2. Source Corpus

Three complementary Markdown files in `data/raw/` cover the 9th-grade Persian
curriculum from different angles:

| File | Persian title | Role | Best for |
|---|---|---|---|
| `farsi-9th-grade-textbook.md` | فارسی — پایه نهم | Original poems + literary analysis sections | scholar |
| `farsi-9th-grade-study-guide.md` | راهنمای گام به گام | Verse-by-verse plain-Persian explanations + vocabulary lists | crammer |
| `farsi-9th-grade-gifted-textbook.md` | فارسی و نگارش — استعدادهای درخشان | Socratic discussion questions + comparative literary analysis | scholar / newcomer |

The key insight driving the entire design: **for the same query, the ideal retrieved
document changes by persona, not by topic.** A crammer asking "معنی مصراع X چیست؟"
needs the plain-Persian paraphrase from the study guide. A scholar asking the identical
question needs the literary analysis. The corpus deliberately contains multiple views
of the same content to make persona-conditioned retrieval non-trivial and measurable.

### What the questions are (and are not)

The exam JSON files in `data/questions/` are **queries**, not corpus documents. They
are 9th-grade Persian language exam questions (MCQ and short-answer) covering meaning,
literary devices, and paraphrase of textbook poems. They must not be indexed — a
question about a verse is not an answer to a question about a verse.

In addition to the six real exam files, `ai_generated_questions.json` contains questions
created by GLM 5.2 via the GLM web agent as **exam-seeded imitation**: the generator
received real exam questions and answers and was asked to produce similar items. It was
not grounded against `data/chunks/corpus.jsonl`. See [question-extraction](question-extraction.md),
"AI-generated questions"; all 487 items' `answer` and `explanation` fields are
model-authored and not human-verified.

---

## 3. Chunking Strategy

The existing `chunking.py` uses character-window splits. That is correct for arbitrary
text but wrong here: a fixed window cuts mid-verse or mid-explanation and destroys the
semantic unit. `corpus_chunker.py` splits instead along the Markdown heading structure.

### Section type taxonomy

Each chunk is assigned a `section_type` that captures what kind of content it contains:

| `section_type` | Where found | Content |
|---|---|---|
| `poem` | textbook `### درس Y` | The original poem or prose text |
| `literary-notes` | textbook `#### دانش ادبی` | Literary device explanations (تشبیه, جان‌بخشی, …) |
| `vocabulary` | study-guide `### معنی لغت` | Vocabulary glossary per lesson |
| `verse-explanation` | study-guide `### معنی ابیات` sub-entries | One bold بیت + its plain-Persian meaning |
| `deep-analysis` | gifted textbook per-lesson blocks | Socratic questions + comparative excerpts |
| **`skip`** | textbook `#### خودارزیابی` | Exam-style questions — **excluded from index** |

The `خودارزیابی` sections contain exam-style questions and would confuse the retriever
into treating corpus questions as answers to student questions. They are dropped during
chunking and never indexed.

### Chunk schema (`data/chunks/corpus.jsonl`)

One JSON object per line:

```jsonl
{
  "chunk_id": "study-guide:lesson-01:verse-explanation:003",
  "source": "study-guide",
  "lesson_id": "lesson-01",
  "section_type": "verse-explanation",
  "text": "**بامدادی که تفاوت نکند لیل و نهار...**\nدر سپیده‌دم که روز و شب از هم بازشناخته نمی‌شوند، دشت و تماشای بهار دل‌پذیر است."
}
```

The `section_type` field is a first-class citizen in all downstream datasets: it is
what allows an analysis of which chunk types each persona benefits from, which is the
measurable claim the thesis needs to support.

---

## 4. Why LLM Judge over Recall@K

Standard IR metrics (Recall@K, MRR) measure whether the retriever surfaces the
*same passage the exam happened to quote* — a topical match signal. For this project,
topical match is insufficient because **the same topically-relevant passage can be the
right answer for one persona and the wrong answer for another**.

Consider the query "معنی مصراع «توانا بود هر که دانا بود» چیست؟":

| Chunk | section\_type | crammer utility | scholar utility |
|---|---|---|---|
| study-guide verse-explanation for this بیت | verse-explanation | **0.91** — plain paraphrase is exactly what is needed | 0.18 — too simplified |
| textbook poem block for درس ۱ | poem | 0.40 — has the text but no explanation | 0.55 — primary source |
| textbook literary notes for درس ۱ | literary-notes | 0.10 — analysis is too advanced | **0.80** — exactly the right depth |
| gifted Socratic discussion block for درس ۱ | deep-analysis | 0.05 — actively unhelpful for this learner | **0.85** — comparative depth, ideal for scholar |

Recall@K would call all four documents equally relevant (they all reference the same
verse). The LLM judge captures the *gradient of pedagogical utility* that makes the
crammer's ideal retrieval set and the scholar's ideal retrieval set nearly
non-overlapping — which is the thesis claim made concrete.

Soft teacher scores (floats in [0, 1]) are also required by the KD loss itself. The
ROPG-KD loss is `KL(teacher_distribution || student_log_probs)`, which requires a
probability distribution over the candidate doc list. Binary labels collapse this to
a one-hot vector and discard the information that, for example, a vocabulary chunk is
*somewhat* useful for a crammer (≈ 0.4) while a deep-analysis chunk is nearly useless
(≈ 0.05). The teacher's soft distribution is strictly more informative than binary
relevance for this loss.

The same reasoning applies to DPO rewrite ranking: a rewrite that retrieves the
exam-quoted passage (high Recall@K) but also retrieves the literary-analysis chunk
ahead of the plain paraphrase is a *worse* outcome for a crammer, even if its
topical precision is identical. The judge evaluates whether the rewrite's framing
would attract the right-depth passages for this learner's profile.

---

## 5. ROPG-KD Dataset

### Schema (`data/ropg_kd/{train,val}.jsonl`)

Each line is one `(question_stem, persona_id)` pair with the top-K retrieved chunks
scored by the LLM teacher:

```jsonl
{
  "query": "معنی مصراع «توانا بود هر که دانا بود» چیست؟",
  "persona_id": "crammer",
  "docs": [
    {
      "chunk_id": "study-guide:lesson-01:verse-explanation:007",
      "text": "...",
      "teacher_score": 0.91
    },
    {
      "chunk_id": "textbook:lesson-01:literary-notes",
      "text": "...",
      "teacher_score": 0.18
    }
  ]
}
```

`docs` contains the top-20 chunks retrieved by the baseline index, ordered by
retrieval rank (not by teacher score — the model must learn the reordering).

`query` is the complete rendered question, not the bare stem. Both this dataset and
`data/dpo/` build it with `data.questions.load_question`, the single definition of that
form: a `Passage:` section for grouped reading-comprehension items, then `Question:`,
then `Options:` / `Pairs:` / `Items:` where the question has them. Gold `answer` and
`explanation` are carried beside the query in `QuestionContext` and passed to the judge
as reference context only — never folded into the query, which would leak the answer into
the retriever's input.

### Generation procedure (`src/data/gen_ropg_data.py`)

For each `(question, persona_id)` in the split:

1. Retrieve the top-20 chunks using the phase-1 dense index (Qwen3-Embedding,
   no instruction prefix — the baseline, not the fine-tuned model).
2. For each of the 20 chunks call the LLM judge once with the prompt below.
3. Collect the 20 scores; write one JSONL record.

Config key: `retriever.top_k` (default 20).

### Teacher judge prompt

```
You are an expert Persian language tutor evaluating study materials.

A student with the following profile is trying to answer an exam question:
Profile: {persona_rendered}

Exam question: {query}

Candidate study passage:
{chunk_text}

Rate 0.0–1.0 how useful this passage is for helping this specific student answer
the question. Consider:
  - Does the depth match the student's comprehension level?
  - Does it provide what this student needs (simple paraphrase vs. deep analysis)?
  - Is the style appropriate (hand-holding vs. terse treatment)?

Respond with a single decimal number only, e.g. 0.73
```

---

## 6. DPO Rewriter Dataset

The pair files have been generated two ways. **Format 2 (rubric era)** is current;
**format 1 (scalar era)** produced the frozen files behind the seed-42 screening result
and is kept below because every published Stage-2 number so far was measured on it.

### Schema (`data/dpo/{train,val}.jsonl`) — format 2

Each line is one preference pair conditioned on a persona, carrying the judge verdict
that produced it:

```jsonl
{
  "format_version": 2,
  "question_ref": "ai_generated_questions:q97",
  "persona_id": "crammer",
  "query": "Question:\nمفهوم عبارت \"فکر انعام تو هرگز نکند شکرگزار\" را به زبان ساده توضیح دهید.",
  "chosen": "توضیح ساده و گام به گام مفهوم عبارت «فکر انعام تو هرگز نکند شکرگزار» با مثال برای دانش‌آموز پایه نهم جهت امتحان",
  "rejected": "مفهوم عبارت «فکر انعام تو هرگز نکند شکرگزار» با تمرکز بر ریشه، ساختار و پیوندهای مفهومی آن",
  "pair_type": "cross_persona",
  "chosen_score": 0.9495,
  "rejected_score": 0.4025,
  "margin": 0.547,
  "chosen_sub_scores": {"meaning_preservation": 0.98, "persona_fit": 0.95, "specificity": 0.9},
  "rejected_sub_scores": {"meaning_preservation": 0.55, "persona_fit": 0.2, "specificity": 0.45},
  "chosen_temperature": 0.8,
  "rejected_temperature": 0.2,
  "rejected_persona_id": "scholar",
  "judge_samples": 1
}
```

`chosen` is the rewrite the judge rates better at surfacing pedagogically useful material
for this learner; `rejected` is worse. `pair_type` is `within_persona` or `cross_persona`.
`rejected_persona_id` names the persona the rejected rewrite was *written for*, which
differs from `persona_id` exactly on cross-persona rows.

`chosen_score` and `rejected_score` are weighted rubric aggregates in `[0, 1]`, and
`margin` is their difference. **On a cross-persona row `rejected_score` is the foreign
rewrite scored under the target persona, not the score it earned under its own persona.**
The row records the comparison that was actually made; the two are not interchangeable.

`query` is the **complete rendered question** produced by `data.questions.load_question`
— the same function and therefore the same string form used to build `data/ropg_kd/`
(`Passage:` / `Question:` / `Options:` / `Pairs:` / `Items:` sections). This matters
because the rewriter's output is consumed by the retriever distilled on that form; a bare
stem would train the rewriter on an input the pipeline never serves. `question_ref` is the
`{exam_stem}:{qid}` line from the split file, so any row can be traced back to its source
question.

`rl.dpo_train.load_pairs` refuses any `format_version` other than 2, and additionally
requires `pair_type` and the three numeric score fields. A format-1 file therefore fails
loudly at load rather than training silently on rows whose provenance is absent.

Row counts for the current build are recorded in the run manifest, not here, because the
filters below reject a variable fraction of candidate pairs.

The candidates remain off-policy for Qwen3-4B: Grok generates them, Luna ranks them, and
Qwen is the policy/reference family.

### Candidate dumps (`data/dpo/candidates/`, `data/dpo/candidates_filtered/`)

Every candidate is persisted, one row per candidate — not per judge sample:

```jsonl
{
  "format_version": 1,
  "question_ref": "ai_generated_questions:q97",
  "persona_id": "scholar",
  "query": "Question:\n…",
  "rewrite": "…",
  "temperature": 0.8,
  "sub_scores": {"meaning_preservation": 0.55, "persona_fit": 0.82, "specificity": 0.88},
  "score": 0.727,
  "judge_samples": 1,
  "filter_reason": null,
  "scored_as": {"crammer": {"sub_scores": {…}, "score": 0.4025, "judge_samples": 1}}
}
```

`candidates/` holds every candidate including rejects, each stamped with the
`filter_reason` that discarded it (`min_judge_samples`, `duplicate_rewrite`).
`candidates_filtered/` holds the survivors. `scored_as` is populated only for each
persona's winner and records that rewrite judged under the *other* personas' rubric
context — the numbers cross-persona pairing compares.

The dumps exist so pair filtering can be retuned offline. Changing a threshold means
re-running selection over these files, not paying for generation again. This is the
direct fix for the format-1 limitation that forced a full regeneration to answer any
question about score distribution.

### Generation procedure (`src/data/gen_dpo_data.py`) — format 2

For each question in the split:

1. Generate 4 candidate rewrites per persona with `PromptedRewriter` backed by
   `grok-4-1-fast` at temperatures 0.2, 0.5, 0.8, 1.1 — 12 rewrites per question over the
   three training personas. The four *draws* matter; the four *values* do not. Measured
   over 5,184 candidates, mean judge score by temperature is 0.796 / 0.798 / 0.797 /
   0.795 and wins per group are 312 / 314 / 339 / 331, so the ladder behaves like random
   assignment. What produces the quality spread the judge ranks is the candidate count as
   an order statistic (mean best-worst margin 0.129 at four candidates, 0.106 at three,
   0.062 at two). Calls within a question are parallelised (`ThreadPoolExecutor`).
2. Score each candidate with `gpt-5.6-luna` at `temperature: 0`, `reasoning_effort: none`,
   and a 200-token budget, under a **strict `json_schema` response format**. The judge
   returns one sub-score per rubric criterion rather than a single gestalt decimal, and
   the configured weights collapse them into the aggregate that pair selection orders by.
   With `judge.samples_per_candidate > 1` the sub-scores are averaged across samples
   before weighting. The judge also receives the question's gold answer and explanation as
   reference context; those fields are judge-only and never enter the rewriter's prompt or
   the stored `query`, since a rewrite conditioned on the answer would leak it into the
   retrieval query.
3. Apply the candidate filters: drop candidates judged fewer than `min_judge_samples`
   times, and collapse candidates that normalize to the same string, keeping the
   highest-scored copy. Write both the full and the surviving sets to the dumps.
4. **Within-persona pair** — take the top-scored survivor as `chosen`, then scan upward
   from the worst survivor for a `rejected` that clears `min_margin` and is not a
   near-duplicate of `chosen` (`max_pair_similarity`, difflib over NFKC-normalized text).
   Scanning rather than taking the extreme keeps the widest genuine gap instead of
   discarding the persona when the extreme pair happens to be a cosmetic twin. A persona
   with no qualifying negative yields no row.
5. **Cross-persona negatives** — re-score each persona's winner under every *other*
   persona's rubric context, then emit a row for target persona X against a rewrite
   written for Y only when X's own winner beats that foreign rewrite **as judged for X**
   by `cross_persona_min_margin`. This costs 6 extra scorings per question and is the
   reason cross-persona rows are labelled rather than assumed.

Rewrite and judge calls are retried with capped exponential backoff. An exhausted rewrite
drops the candidate; an exhausted judge sample is skipped, and a candidate with no usable
sample is dropped. No pair is written with `chosen == rejected`.

Filters live in `configs/datagen_dpo.yaml` under `filters:` — `min_judge_samples`,
`dedup_candidates`, `min_chosen_score`, `min_margin`, `max_pair_similarity`,
`cross_persona_min_margin`. The generator refuses to start when `min_judge_samples`
exceeds `judge.samples_per_candidate`, which would otherwise discard every candidate after
the entire run had been paid for.

Candidate dump directories default to `data.candidates_dir` and
`data.filtered_candidates_dir` and are overridable with `--candidates-out` and
`--filtered-candidates-out`.

**Verified endpoint capabilities.** `gpt-5.6-luna` accepts `temperature: 0` and strict
`json_schema` response formats; both were confirmed against the live route before the
recipe was adopted. `OpenAICompatClient` forwards `response_format` only when set, so
other call sites are unaffected.

**Calibrated threshold.** `min_margin` was a starting guess at 0.10 and is now **0.04**,
set by rescoring 300 frozen candidates three times in `benchmarks/judge_noise.py` —
method and numbers in [judge-noise-luna-v1](results/judge-noise-luna-v1.md). The judge is
not deterministic at `temperature: 0` (18.0% byte-identical replies, single-draw
SD 0.0265), which is why `samples_per_candidate` is 3 rather than 1. Raising the
threshold from 0.04 to 0.10 discards 36.6% of within-persona rows while leaving modeled
label error at essentially zero either way, so the low value is the better trade. Two
thresholds remain uncalibrated: `cross_persona_min_margin`, because the noise experiment
formed within-persona pairs only, and `min_chosen_score`, which sits in the score band
where the judge is least reliable.

**As-built measurements.** The shipped v4 files were audited after generation:
[dpo-data-v4-audit](results/dpo-data-v4-audit.md). Headlines — 2,658 train / 534 val rows
over the same 432 / 92 questions, 96.9% group yield, modeled label error 0.08%, zero
same-persona contradictions and zero split leakage, and a measured persona-discrimination
effect (a rewrite scores 0.180 lower under a foreign persona, in 97.1% of cross-persona
rows). The audit also records the open risk: `scholar` appears as the negative 1.6× more
often than as the positive, because the 0.40-weighted `meaning_preservation` criterion
penalises advanced rewrites of questions whose own wording prescribes a simple register.
Row counts and distributions in this section describe the schema; per-generation numbers
live in the audit, which is versioned alongside the data.

### DPO judge prompt — format 2

```
You are an expert Persian language tutor evaluating query rewrites for a RAG
retrieval system.

A student with the following profile is searching for study material:
Profile: {persona_rendered}

Original exam question: {original_query}

Reference answer and rubric (judge context only; the rewriter never saw this):
Gold answer/reference:
{answer}

Gold explanation/rubric:
{explanation}

Candidate rewrite: {rewrite}

Score how well this rewrite would help retrieve the right study material for this
specific student. Rate each criterion independently from 0.0 to 1.0, where 0.0 is a
complete failure on that criterion and 1.0 could not be improved:
  - meaning_preservation (weight 0.4): Does the rewrite preserve the original
    question's meaning, scope, and intent without inventing facts, narrowing the
    topic, or answering the question itself?
  - persona_fit (weight 0.35): Does the vocabulary, depth, and framing match this
    specific learner's background, goal, and tolerance for detail, rather than being
    generically well written?
  - specificity (weight 0.25): Is the query concrete enough to surface the right
    passages — naming the actual concepts and prerequisite terms — without being so
    narrow it misses them?

Judge each criterion on its own merits — a rewrite may score high on one and low on
another. Use the full range; reserve scores above 0.9 for rewrites you cannot improve.
```

The rubric block and the response schema are both generated from `judge.rubric` in the
config, so adding or reweighting a criterion is a config edit. Weights must sum to 1.0 or
the generator refuses to start.

**The weights encode a policy choice.** At 0.40 vs 0.35, fidelity to the question outranks
fitting the learner. On questions whose wording prescribes a register — e.g. one asking for
an explanation «به زبان ساده» — a scholar-appropriate rewrite that pivots to etymology and
structure is penalised on `meaning_preservation` and can lose to a simpler rewrite even
when judged *for the scholar*. That is a defensible ranking, not a bug, but it
systematically disadvantages the scholar persona on such items and should be checked
against the `pair_type` × `persona_id` cross-tab after each build.

The reference block is omitted defensively only when a question carries neither gold
field. Every question in the current bank has both: **618/618** have a gold `answer` and
**618/618** have an `explanation`, so the block is currently never omitted.

---

### History: format 1 (scalar era, retired)

Format 1 produced the frozen files behind
[dpo-arms-seed42-v1](results/dpo-arms-seed42-v1.md). Its rows carried only
`format_version`, `question_ref`, `persona_id`, `query`, `chosen`, `rejected`:

```jsonl
{
  "format_version": 1,
  "question_ref": "khordad1403-keshvari:q5",
  "persona_id": "crammer",
  "query": "Question:\nمعنی بیت «پاک و بی‌عیب خدایی که به تقدیر عزیز» چیست؟",
  "chosen": "معنی ساده بیت «پاک و بی‌عیب خدایی که به تقدیر عزیز» به زبان روزمره با مثال برای دانش‌آموز پایه نهم",
  "rejected": "تحلیل صور خیال و وزن عروضی در بیت «پاک و بی‌عیب خدایی که به تقدیر عزیز»"
}
```

Procedure: 3 rewrites per `(question, persona)` from `grok-4-1-fast` at temperatures
0.2/0.5/0.9 (`max_workers=4`); one judge call per candidate to `gpt-5.6-luna` at
`temperature: 1.0`, `reasoning_effort: none`, `max_completion_tokens: 16`, replying with a
single bare decimal parsed by a strict regex; `chosen` = highest, `rejected` = lowest, with
**no margin filter**; cross-persona negatives appended in both orientations whenever the
two personas' best scores differed by more than `cross_persona_threshold: 0.25`, with no
additional LLM calls.

Frozen contents: 1,739 training rows over 432 questions and 351 validation rows over 92
questions, 272 unique validation `(question_ref, persona_id)` keys. No empty completion,
identical chosen/rejected pair, exact duplicate row, or train/validation question overlap.

Why it was replaced — four defects, each addressed above:

1. **No provenance.** Rows stored no scores, margins, temperatures, source persona, or
   pair type, so no question about the data could be answered without regenerating it.
   Row order is not provenance.
2. **Best-of-3 vs worst-of-3 from one model.** With no margin filter, any persona with two
   surviving candidates produced a pair, so the contrast was frequently style jitter rather
   than a quality difference. Measured on the frozen files: 0 identical pairs, but 5.06% of
   train rows are near-duplicates at difflib ratio ≥ 0.90, and mean lengths differ by under
   3 characters (chosen 110.1, rejected 107.1).
3. **Noisy labels.** A single bare scalar at `temperature: 1.0` with a 16-token budget.
   Judge self-agreement was 73.0% and Luna–Gemini agreement 65–70%; held-out preference
   accuracy peaked at 0.69, roughly the labels' own self-consistency ceiling. See
   [judge-agreement-luna-gemini-v1](results/judge-agreement-luna-gemini-v1.md).
4. **Unlabelled cross-persona rows.** The 0.25 gate compared two scores produced under two
   *different* persona prompts, so the gap was not a like-for-like quantity and the pair
   was never actually judged.

*Approach choice, unchanged:* direct rewrite scoring was chosen over end-to-end scoring
(retrieve → generate → judge the final answer) on cost grounds — end-to-end would require
N generation calls of ~800 tokens per (question, persona). Note the cost argument applies
to *answer* generation; a retrieval-grounded reward was considered separately and rejected
on the design ground that the rewriter is evaluated independently of retrieval. See
[methodology](methodology.md).

*Rewriter model:* an initial version used a local Gemma-4-E4B via Unsloth (4-bit
quantised). Rewrite quality was insufficient, so the rewriter was replaced with
`grok-4-1-fast`. The judge remains a separate model to preserve the judge-independence
rule.

*Judge prompt, format 1:* identical framing to the current one down to the reference
block, then:

```
Rate 0.0–1.0 how well this rewrite would help retrieve the right study material
for this specific student. A good rewrite should:
  - Preserve the original question's meaning
  - Use vocabulary and framing that matches the student's profile
  - Be specific enough to surface relevant passages at the right depth

Respond with a single decimal number only, e.g. 0.61
```

---

## 7. Split Strategy

The current files use a deterministic **question-level** random split with seed 42 and
70/15/15 train/validation/test ratios. No `question_ref` crosses train and validation,
which prevents leakage from the multiple persona and pair rows attached to one question.

This split is not source-exam-disjoint. Questions from all seven source files were pooled
before shuffling, and source files occur in more than one split. Results must therefore be
described as question-disjoint but source-overlapping. The splitter also does not enforce
shared-passage (`group_id`) isolation, so a shared passage may cross splits.

Existing DPO and ROPG assets stay frozen for this experiment; this pass does not re-split them.

An optional `--fixed-exam-splits` mode pins the six real exam files to their historical
splits and randomizes only remaining files. It can reproduce the original per-exam
assignment, but it does not describe the frozen files used by the current runs.

| Split | Fixed exam files (`--fixed-exam-splits` only) | Personas |
|---|---|---|
| train | `khordad-1402-arzeshyabi-ostani.json`, `midterm-unknown_1.json`, `keshvari-unknown_1.json` | crammer, scholar, steady |
| val | `khordad1403-keshvari.json` | crammer, scholar, steady |
| test | `khordad1404-keshvari.json`, `khordad1404-khorasan.json` | crammer, scholar, steady, **newcomer** |

The `newcomer` persona (bright student, no prior knowledge) appears only at test time.
It shares features with both `crammer` (low prior knowledge) and `scholar` (high
comprehension), making it a genuine out-of-distribution probe for generalization across
learner profiles.

Question IDs for each split are written to `data/splits/{train,val,test}_qids.txt`,
one `{exam_file}:{question_id}` per line. Regenerate at any time with:
`uv run python -m data.corpus_chunker --splits-only --questions-dir data/questions`

---

## 8. File Layout

```
data/
├── raw/                                         # source corpora — read-only
│   ├── farsi-9th-grade-textbook.md
│   ├── farsi-9th-grade-study-guide.md
│   └── farsi-9th-grade-gifted-textbook.md
├── chunks/
│   └── corpus.jsonl                             # generated by corpus_chunker.py
├── questions/                                   # exam JSONs — read-only
│   ├── ai_generated_questions.json              # LLM-generated augmentation questions
│   ├── khordad-1402-arzeshyabi-ostani.json
│   ├── khordad1403-keshvari.json
│   ├── khordad1404-keshvari.json
│   ├── khordad1404-khorasan.json
│   ├── keshvari-unknown_1.json
│   └── midterm-unknown_1.json
├── splits/
│   ├── train_qids.txt                           # generated by corpus_chunker.py
│   ├── val_qids.txt
│   └── test_qids.txt
├── ropg_kd/
│   ├── train.jsonl                              # generated by gen_ropg_data.py
│   └── val.jsonl
└── dpo/
    ├── train.jsonl                              # generated by gen_dpo_data.py
    └── val.jsonl
```
