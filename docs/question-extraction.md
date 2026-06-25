# Question Extraction

> How exam questions are turned into the structured QA dataset that drives evaluation and preference-pair generation. This is the provenance record for the question data — distinct from [[data-extraction]], which covers building the RAG *corpus* from textbook scans. See [[methodology]] and [[experiment-design]] for what the questions feed into downstream (splits, judging, DPO pairs).

---

## Why this matters

The corpus ([[data-extraction]]) is what the RAG system *reads*. The questions documented here are what it is *tested on* and trained against. They start as exam PDFs — often scanned, sometimes with a separate answer key in a different file — and must become a clean, machine-parseable dataset that scripts split into train/validation/test without leakage. Documenting the path lets a reviewer judge the data and lets me rebuild it the same way later.

## Pipeline

1. **Collect exam sources.** PDFs / images of exam papers (چهارگزینه‌ای, جای خالی, تشریحی, …). Answer keys are sometimes in the same file, sometimes a separate file, sometimes absent.
2. **Extract with a VLM.** Feed the exam page images to a vision model with the [prompt below](#vlm-extraction-prompt). It returns **one JSON object per exam**: a `passages` array plus a `questions` array.
3. **Store raw JSON.** One file per source, e.g. `data/exams/raw/<source-slug>.json`. The filename carries the source identity. Never commit large dataset files (see [[CLAUDE]] repo rules).
4. **Join answers (when separate).** Where the answer key is a different file, a downstream script matches it to the question rows on `number` (unique within the file) and fills the `answer` field wherever it is `null`.
5. **Flatten + split.** Scripts read the JSON, flatten to CSV, and produce train/val/test splits **by source** (never by row) so no passage/question crosses splits — the no-leakage rule in [[CLAUDE]] / [[things-to-consider]].

The VLM step is not byte-reproducible (a second run of the same page may differ slightly). Treat each JSON as a reviewed artifact, not a deterministic build output — spot-check a sample against the source PDF.

## Design decisions (and their rationale)

- **JSON, not markdown.** One JSON object per exam parses with `json.loads` and needs no fragile delimiter scanning. The `type` field discriminates question shape so each kind keeps its natural structure.
- **No batch metadata in the JSON.** Source, grade, subject, and language are constant within one exam file (or recorded by the filename), so they are not repeated. The flatten step can attach them from the filename if a CSV needs them as columns.
- **Only what is printed is captured.** Inferred enrichment (topic, difficulty, cognitive type, etc.) is deliberately omitted — it is either a noisy model guess or non-essential, and it is a one-line addition later if an analysis ever needs it. Keeping the schema to page facts keeps it reliable.
- **The VLM never invents answers.** `answer` is filled **only** if the answer is printed on the page; otherwise `answer: null`. A null is the signal that the answer must be joined in later from a separate key (or does not exist yet).
- **`number` is the join key.** The printed question number is transcribed verbatim and is unique within the file, so a separate answer-key file is joined on `number`. `id` is a normalized ASCII handle (`q<number>`) for a stable per-row reference after flattening.
- **Persian content is preserved verbatim.** No ZWNJ / ye-ke / digit normalization at extraction — that is applied consistently downstream in `src/data/persian.py` to both corpus and queries. The VLM transcribes exactly what is printed; it does not translate or silently "correct" the script.
- **Splits are not assigned here.** Extraction produces questions only; train/val/test assignment is a separate, source-aware step.

## Question types

The VLM picks the `type` that best fits each question and follows that type's field rules.

| `type` | Use for | Answer encoding |
| --- | --- | --- |
| `mcq` | چهارگزینه‌ای / any multiple-choice | 1-based index into `options` (array of indices if multi-correct) |
| `fill_blank` | جای خالی / «کامل کنید» / complete-the-phrase | array of fill strings, in blank order; blanks shown as `____` in `stem` |
| `true_false` | درست / نادرست | `true` or `false` |
| `short_answer` | کوتاه‌پاسخ / معنی واژه | string (array if several acceptable answers) |
| `essay` | تشریحی / open-ended | usually `null`; a printed sample answer or rubric goes in `explanation` |
| `matching` | تطبیق دو ستون | array of `[left_index, right_index]` pairs |
| `ordering` | مرتب‌کردن | array of `item` indices in correct order |
| `other` | anything else | best-effort; describe shape in `notes` |

**Passage-grouped questions** (a shared متن / reading passage with sub-questions) are kept together: the shared text goes once in the top-level `passages` array, and each sub-question carries the matching `group_id`. Downstream, a group is never split across train/val/test.

## JSON schema

```jsonc
{
  "exam_title": "string|null",     // as printed, if any
  "passages": [                     // omit or [] if no passage-based questions
    { "group_id": "string", "text": "string" }   // shared متن, Persian verbatim
  ],
  "questions": [
    {
      "id": "string",              // normalized handle, "q<number>"
      "number": "string",          // printed question number, VERBATIM (join key)
      "type": "mcq|fill_blank|true_false|short_answer|essay|matching|ordering|other",
      "stem": "string",            // question text, Persian verbatim; blanks as ____
      "group_id": "string|null",   // links to passages[].group_id, else null

      // --- type-specific (include only those that apply) ---
      "options": ["string", ...],            // mcq: in printed order
      "pairs": { "left": [...], "right": [...] },   // matching
      "items": ["string", ...],              // ordering

      // --- answer (only if printed on the page; null otherwise) ---
      "answer": "varies|null",     // see type table; null when not on the page
      "explanation": "string|null",          // worked solution / rubric if printed

      "points": "number|null",     // marks if printed
      "notes": "string|null"       // unreadable chars, ambiguities, type=other shape
    }
  ]
}
```

## VLM extraction prompt

Paste this verbatim alongside the exam page image(s). It is self-contained (schema + per-type rules + an example) because the VLM sees only the prompt.

```text
You extract exam questions from the provided page image(s) into a single JSON object.
The pages are from a Persian (Farsi) exam. Read every question on every page.

OUTPUT
- Output ONLY one JSON object. No commentary, no explanation, no markdown code fences.
- The object must follow the SCHEMA below exactly. Use null for any field you cannot fill.

FAITHFULNESS
- Transcribe Persian text EXACTLY as printed. Do NOT translate it. Do NOT normalize it:
  keep the original ZWNJ (نیم‌فاصله), the printed ی/ک vs ي/ك, and the original digit forms.
- Do NOT silently fix suspected typos. If a character is unreadable, write ? in its place
  and describe the issue in "notes".
- NEVER invent an answer. Fill "answer" ONLY if the correct answer is printed on the page
  (an answer key, a marked choice, a bolded solution). Otherwise set "answer": null.

PER QUESTION
- Copy the printed question number EXACTLY into "number" (this is a join key). Set "id"
  to "q" followed by that number in ASCII digits (e.g. number "۲" -> id "q2").
- Choose the single best "type" and follow its rule:
  - mcq         : "options" = the choices in printed order; "answer" = 1-based index into
                  options (or an array of indices if multiple are correct). Treat الف/ب/ج/د
                  or ۱/۲/۳/۴ labels as positions 1..n.
  - fill_blank  : write each blank as ____ in "stem"; "answer" = array of the fill strings
                  in blank order.
  - true_false  : "answer" = true or false.
  - short_answer: "answer" = the expected text (array if several answers are acceptable).
  - essay       : "answer" = null; put any printed sample answer or rubric in "explanation".
  - matching    : "pairs" = {"left":[...], "right":[...]} in printed order; "answer" =
                  array of [left_index, right_index] pairs.
  - ordering    : "items" = the items as presented; "answer" = array of item indices in
                  the correct order.
  - other       : describe the shape in "notes".
- Keep all options/items/pairs in their printed order.
- "points": printed marks for the question, else null.

PASSAGE-BASED QUESTIONS
- If several questions share a reading passage (متن), put the passage ONCE in the top-level
  "passages" array with a "group_id", and give each of its questions that same "group_id".

SCHEMA
{
  "exam_title": "string|null",
  "passages": [ { "group_id": "string", "text": "string" } ],
  "questions": [
    {
      "id": "string",
      "number": "string",
      "type": "mcq|fill_blank|true_false|short_answer|essay|matching|ordering|other",
      "stem": "string",
      "group_id": "string|null",
      "options": ["string"],
      "pairs": { "left": ["string"], "right": ["string"] },
      "items": ["string"],
      "answer": null,
      "explanation": "string|null",
      "points": null,
      "notes": "string|null"
    }
  ]
}

EXAMPLE (shape only)
{
  "exam_title": "آزمون نوبت اول فارسی نهم",
  "passages": [],
  "questions": [
    {
      "id": "q1",
      "number": "۱",
      "type": "mcq",
      "stem": "در کدام گزینه آرایهٔ «تشبیه» به کار رفته است؟",
      "group_id": null,
      "options": ["گزینهٔ اول ...", "گزینهٔ دوم ...", "گزینهٔ سوم ...", "گزینهٔ چهارم ..."],
      "answer": 3,
      "explanation": null,
      "points": 1,
      "notes": null
    },
    {
      "id": "q2",
      "number": "۲",
      "type": "fill_blank",
      "stem": "جمع واژهٔ «کتاب»، ____ است.",
      "group_id": null,
      "answer": null,
      "explanation": null,
      "points": 1,
      "notes": "answer key not on this page"
    }
  ]
}
```

## Limitations

- **Not reproducible byte-for-byte.** The VLM runs behind a model API; a rerun of the same page will differ. Treat each JSON as a reviewed artifact and spot-check against the source.
- **Transcription errors can survive review.** A misread character that still forms a plausible Persian word passes both the model and a quick read.
- **`answer: null` is not a claim about correctness.** It only means the answer was not on the page — the join step still has to supply (and a human should spot-check) the gold answer.
