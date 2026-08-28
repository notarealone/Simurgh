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

In addition to the six real exam files, `ai_generated_questions.json` contains
LLM-generated questions grounded in the same 9th-grade Persian curriculum, added to
augment data quantity and improve lesson and question-type coverage across the corpus.

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

### Schema (`data/dpo/{train,val}.jsonl`)

Each line is one preference pair conditioned on a persona:

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

`chosen` is the rewrite the LLM judge rates as better at surfacing pedagogically
useful material for this learner. `rejected` is worse.

`query` is the **complete rendered question** produced by `data.questions.load_question`
— the same function and therefore the same string form used to build `data/ropg_kd/`
(`Passage:` / `Question:` / `Options:` / `Pairs:` / `Items:` sections). This matters
because the rewriter's output is consumed by the retriever distilled on that form; a bare
stem would train the rewriter on an input the pipeline never serves. `question_ref` is the
`{exam_stem}:{qid}` line from the split file, so any row can be traced back to its source
question.

`format_version` is 1 for pairs built this way. `rl.dpo_train.load_pairs` refuses any
other version, which is what stops the earlier stem-only pairs from being trained on
silently — they load and train without error otherwise.

### Generation procedure (`src/data/gen_dpo_data.py`)

For each `(question, persona_id)` in the split:

1. Generate N=3 candidate rewrites using `PromptedRewriter` backed by a remote Grok
   model (`grok-4-1-fast`) at temperatures 0.2, 0.5, 0.9 to ensure diversity.
   Rewriter and judge calls within a question are parallelised
   (`ThreadPoolExecutor`, `max_workers=4`) to reduce wall-clock time.
2. For each candidate call the LLM judge (`gpt-5.6-luna` with `reasoning_effort: none`
   and a 16-token completion budget — with reasoning enabled the budget is consumed by
   reasoning tokens and the reply comes back empty) once with the prompt below. The judge
   also receives the question's gold answer and explanation as reference context, so it
   scores a rewrite against the material that actually resolves the question rather than
   guessing at it. Those gold fields are judge-only: they never enter the rewriter's
   prompt or the stored `query`, since a rewrite conditioned on the answer would leak it
   into the retrieval query. Collect one score per candidate.
3. Pair the highest-scored and lowest-scored candidates as `(chosen, rejected)`.
4. **Cross-persona negatives**: within each question, the `chosen` rewrite for
   `scholar` becomes a `rejected` for `crammer` (and vice versa) without additional
   LLM calls. These cross-persona pairs are appended to the same output file.

Rewrite and judge calls are retried with capped exponential backoff; a candidate whose
retry budget is exhausted is dropped rather than scored 0.0, and a persona left with
fewer than two surviving candidates is skipped — so no pair is ever written with
`chosen == rejected`.

Config keys: `rewriter.model`, `rewriter.temperatures`, `rewriter.max_workers`,
`rewriter.max_attempts`, `judge.model`, `judge.reasoning_effort`, `judge.max_attempts`.

*Approach choice:* Direct rewrite scoring was chosen over end-to-end scoring
(retrieve → generate → judge the final answer) on cost grounds — end-to-end would
require N generation calls (800 tokens each) per (question, persona), which is
prohibitive on a 4-week thesis budget. See [methodology](methodology.md) for the
full trade-off analysis.

*Rewriter model:* An initial version used a local Gemma-4-E4B via Unsloth (4-bit
quantised) to generate rewrites. Rewrite quality was insufficient, so the rewriter
was replaced with a remote Grok model (`grok-4-1-fast`). The judge remains a
separate model to preserve the judge-independence rule.

### DPO judge prompt

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

Rate 0.0–1.0 how well this rewrite would help retrieve the right study material
for this specific student. A good rewrite should:
  - Preserve the original question's meaning
  - Use vocabulary and framing that matches the student's profile
  - Be specific enough to surface relevant passages at the right depth

Respond with a single decimal number only, e.g. 0.61
```

The reference block is omitted entirely when a question carries neither field (7 of 524
have no gold answer, 509 have no explanation).

---

## 7. Split Strategy

Splits are assigned at the **question level** with a deterministic random shuffle
(default seed 42, ratios 70/15/15 train/val/test). All questions from all seven source
files — six real exam files and the LLM-generated set — are pooled and shuffled
together, then partitioned. This ensures the synthetic questions are distributed across
all splits and that no single exam dominates any one split.

An optional `--fixed-exam-splits` mode pins the six real exam files to their
historically designated splits (table below) and only randomizes the remaining files
(currently `ai_generated_questions.json`). This is useful for reproducing the original
per-exam assignment if needed, but the default question-level shuffle is the canonical
split for all training runs.

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
