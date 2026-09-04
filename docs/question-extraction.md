# Question Extraction

> How exam questions are turned into the structured QA dataset that drives evaluation and preference-pair generation. This is the provenance record for the question data — distinct from [data-extraction](data-extraction.md), which covers building the RAG *corpus* from textbook scans. See [methodology](methodology.md) and [experiment-design](experiment-design.md) for what the questions feed into downstream (splits, judging, DPO pairs).

---

## Why this matters

The corpus ([data-extraction](data-extraction.md)) is what the RAG system *reads*. The questions documented here are what it is *tested on* and trained against. They start as exam PDFs — often scanned, sometimes with a separate answer key in a different file — and must become a clean, machine-parseable dataset. The current split prevents row-level leakage (`question_ref` does not cross splits), but source files and shared passages may overlap.

## Pipeline

1. **Collect exam sources.** PDFs / images of exam papers (چهارگزینه‌ای, جای خالی, تشریحی, …). Answer keys are sometimes in the same file, sometimes a separate file, sometimes absent.
2. **Extract with a VLM.** Feed the exam page images to a vision model with the [prompt below](#vlm-extraction-prompt). It returns **one JSON object per exam**: a `passages` array plus a `questions` array.
3. **Store reviewed JSON.** The committed question files live in `data/questions/*.json`, one file per source. The filename carries the source identity. Validate every file with `json.loads` before trusting it — a model occasionally emits a raw newline inside a string (`Invalid control character`) or a stray `...`; re-prompt or repair those. Never commit large dataset files (see [CLAUDE](../CLAUDE.md) repo rules).
4. **Join answers (when separate).** Where the answer key is a different file, a downstream script matches it to the question rows on `number` (unique within the file) and fills the `answer` field wherever it is `null`.
5. **Flatten + split.** Scripts read the JSON, flatten to CSV, and produce question-level train/val/test assignments in `src/data/corpus_chunker.py`, not row-level assignments. The frozen files are `question_ref`-disjoint but source-overlapping, and shared-passage (`group_id`) isolation is not enforced; see [data-design](data-design.md), "Split Strategy".

The VLM step is not byte-reproducible (a second run of the same page may differ slightly). Treat each JSON as a reviewed artifact, not a deterministic build output — spot-check a sample against the source PDF.

## Design decisions (and their rationale)

- **JSON, not markdown.** One JSON object per exam parses with `json.loads` and needs no fragile delimiter scanning. The `type` field discriminates question shape so each kind keeps its natural structure.
- **No batch metadata in the JSON.** Source, grade, subject, and language are constant within one exam file (or recorded by the filename), so they are not repeated. The flatten step can attach them from the filename if a CSV needs them as columns.
- **Only what is printed is captured.** Inferred enrichment (topic, difficulty, cognitive type, etc.) is deliberately omitted — it is either a noisy model guess or non-essential, and it is a one-line addition later if an analysis ever needs it. Keeping the schema to page facts keeps it reliable.
- **The VLM never invents answers.** `answer` is filled **only** if the answer is printed on the page; otherwise `answer: null`. A null is the signal that the answer must be joined in later from a separate key (or does not exist yet).
- **`number` is the join key.** The printed question number is transcribed verbatim and is unique within the file, so a separate answer-key file is joined on `number`. `id` is a normalized ASCII handle (`q<number>`) for a stable per-row reference after flattening.
- **Persian content is preserved verbatim.** No ZWNJ / ye-ke / digit normalization at extraction — that is applied consistently downstream in `src/data/persian.py` to both corpus and queries. The VLM transcribes exactly what is printed; it does not translate or silently "correct" the script.
- **Source emphasis is preserved.** Underlined / bold / highlighted spans — the "بخش مشخص شده" that many questions depend on — are wrapped in Markdown bold (`**…**`) inside the text, so the marked word survives into the dataset. It is the only markup added to otherwise-verbatim text, stays distinct from the `____` blank marker, and downstream can render or strip it.
- **Splits are not assigned here.** Extraction produces questions only; train/val/test assignment is a separate, question-level step in `src/data/corpus_chunker.py`. The optional `--fixed-exam-splits` mode is source-aware but did not build the frozen files.

## Question types

The VLM picks the `type` that best fits each question and follows that type's field rules.

| `type` | Use for | Answer encoding |
| --- | --- | --- |
| `mcq` | چهارگزینه‌ای / any multiple-choice | 1-based index into `options` (array of indices if multi-correct) |
| `fill_blank` | جای خالی / «کامل کنید» / complete-the-phrase | array of fill strings, in blank order; blanks shown as `____` in `stem` |
| `true_false` | درست / نادرست | `true` or `false` |
| `short_answer` | کوتاه‌پاسخ / معنی واژه | string (array if several acceptable answers) |
| `essay` | تشریحی / open-ended | usually `null`; a printed sample answer or rubric goes in `explanation` |
| `matching` | تطبیق دو ستون | array of `[left_pos, right_pos]` pairs, 1-based |
| `ordering` | مرتب‌کردن | array of item positions in correct order, 1-based |
| `other` | anything else | best-effort; describe shape in `notes` |

**Passage-grouped questions** (a shared متن / reading passage with sub-questions) are kept together in the extraction schema: the shared text goes once in the top-level `passages` array, and each sub-question carries the matching `group_id`. The current splitter does not enforce `group_id` isolation, so a shared passage can appear across splits; do not describe the frozen assets as passage-disjoint.

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
      "stem": "string",            // question text, Persian verbatim; blanks as ____, marked spans as **…**
      "group_id": "string|null",   // links to passages[].group_id, else null

      // --- type-specific (include only those that apply) ---
      "options": ["string", ...],            // mcq: in printed order
      "pairs": { "left": [...], "right": [...] },   // matching
      "items": ["string", ...],              // ordering

      // --- answer (only if printed on the page; null otherwise) ---
      "answer": "varies|null",     // see type table; null when not on the page
      "explanation": "string|null",          // worked solution / rubric if printed

      "points": "string|null",     // printed marks, verbatim, e.g. "۰/۲۵" (Persian decimal uses /)
      "notes": "string|null"       // unreadable chars, ambiguities, type=other shape
    }
  ]
}
```

## VLM extraction prompt

Paste this verbatim alongside the exam page image(s). It is self-contained (schema + per-type rules + an example) because the VLM sees only the prompt.

````text
You extract exam questions from the provided page image(s) into a single JSON object.
The pages are from a Persian (Farsi) exam. Read every question on every page.

OUTPUT
- Output ONLY one JSON object, wrapped in a single Markdown fenced code block (a line of three
  backticks then "json" to open, a line of three backticks to close) and nothing else — no
  commentary or explanation outside the fence. The code block preserves exact spacing —
  ordinary spaces and ZWNJ / نیم‌فاصله — so the Persian text is not collapsed when rendered or copied.
- The object must follow the SCHEMA below exactly. Use null for any field you cannot fill.
- It MUST be valid JSON that a standard parser accepts. In particular:
  - Escape every line break inside a string as \n; NEVER put a raw line break inside a string.
  - Escape any " inside a string as \". Prefer the Persian quotes « » in content.
  - JSON numbers (e.g. "answer" indices) use ASCII digits only; Persian/Arabic digits appear
    ONLY inside quoted strings.
  - No comments, no trailing commas, and never use ... as a value or a placeholder.

FAITHFULNESS
- Transcribe Persian text EXACTLY as printed. Do NOT translate it. Do NOT normalize it:
  keep the original ZWNJ (نیم‌فاصله), the printed ی/ک vs ي/ك, and the original digit forms.
- Do NOT silently fix suspected typos. If a character is unreadable, write ? in its place
  and describe the issue in "notes".
- NEVER invent an answer. Fill "answer" ONLY if the correct answer is printed on the page
  (an answer key, a marked choice, a bolded solution). Otherwise set "answer": null.

MARKED TEXT (underline / bold / highlight)
- Many questions refer to a "بخش مشخص شده": a word or phrase the source underlines, bolds, or
  highlights. Preserve that marking inline so the question stays meaningful — wrap the marked
  span in **double asterisks** exactly where it appears, inside "stem", "options", or a passage.
  Printed (with بصیرت underlined): «معنی واژهٔ بصیرت را بنویسید»
  -> "stem": "معنی واژهٔ **بصیرت** را بنویسید".
- Mark ONLY spans that are actually emphasized in the source; never add emphasis of your own.
- ** is the only formatting markup allowed; everything else stays a verbatim transcription. Do
  NOT use underscores for emphasis — underscores are reserved for the blank marker ____.
- When you mark the span inline, you do not also need to describe it in "notes".

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
  - matching    : "pairs" = an object with "left" and "right" arrays, in printed order;
                  "answer" = array of [left_position, right_position] pairs, 1-based.
  - ordering    : "items" = the items as presented; "answer" = array of item positions,
                  1-based, in the correct order.
- ALL positions/indices are 1-based (the first option/item/row is 1, not 0).
  - other       : describe the shape in "notes".
- Keep all options/items/pairs in their printed order, and capture EVERY option (do not
  stop after the first two).
- "points": the printed marks as a STRING, verbatim (Persian uses / as the decimal
  separator, e.g. "۰/۲۵"); else null.

PASSAGE-BASED QUESTIONS
- If several questions share a reading passage (متن), put the passage ONCE in the top-level
  "passages" array with a "group_id", and give each of its questions that same "group_id".

SCHEMA
```json
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
```

EXAMPLE (shape only)
```json
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
      "options": ["متن گزینهٔ اول", "متن گزینهٔ دوم", "متن گزینهٔ سوم", "متن گزینهٔ چهارم"],
      "answer": 3,
      "explanation": null,
      "points": "۱",
      "notes": null
    },
    {
      "id": "q2",
      "number": "۲",
      "type": "fill_blank",
      "stem": "جمع واژهٔ **کتاب**، ____ است.",
      "group_id": null,
      "answer": null,
      "explanation": null,
      "points": "۱",
      "notes": "answer key not on this page"
    }
  ]
}
```
````

## AI-generated questions

`data/questions/ai_generated_questions.json` was produced outside the VLM extraction
pipeline above by **GLM 5.2 via the GLM web agent**. The agent was seeded with real exam
questions together with their answers and asked to produce similar questions. The result
is **exam-seeded**, curriculum-adjacent by imitation, and was not checked against
`data/chunks/corpus.jsonl`; it is not corpus-grounded.

Generation happened in a chat web agent, not a repository script. No prompt file,
temperature, or seed exists, so the JSON is a reviewed artifact and is not reproducible.
It uses the same schema as the VLM output, so downstream handling is identical. The file
contains **487 of 618 questions (78.8%)**: 337/432 train, 74/92 validation, and 76/94
test. Its type mix is `short_answer` 199, `fill_blank` 106, `true_false` 82,
`matching` 74, `mcq` 26, and no essays. All 487 `answer` and `explanation` fields are
model-authored and not human-verified.

## Limitations

- **Not reproducible byte-for-byte.** The VLM runs behind a model API; a rerun of the same page will differ. Treat each JSON as a reviewed artifact and spot-check against the source.
- **Transcription errors can survive review.** A misread character that still forms a plausible Persian word passes both the model and a quick read.
- **`answer: null` is not a claim about correctness.** It only means the answer was not on the page — the join step still has to supply (and a human should spot-check) the gold answer.
- **Model-authored gold dominates the pool.** All 487 AI-generated questions use
  model-authored, non-human-verified answers and explanations. Any EM/F1 or
  reader-in-the-loop signal therefore rests on model-authored gold for roughly four
  fifths of the pool; the held-out test set is 76/94 (**80.9%**) AI-generated.
