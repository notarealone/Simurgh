# Data Extraction

> How the raw corpus in `data/raw/` is built from source textbooks. This is the provenance record for every file the RAG index ingests — read it before trusting or rebuilding the corpus. See [methodology](methodology.md) for what happens to this text downstream (chunking, indexing, retrieval).

---

## Why this matters

Every `data/raw/*.md` file starts as a scanned PDF of a printed book. The path from scan to clean markdown passes through OCR and an LLM, so it introduces errors a reader cannot see in the final file. Documenting the path lets a reviewer judge the data and lets me rebuild it the same way later.

## Pipeline

Four steps turn a printed book into one markdown file:

1. **Download the source PDF.** These scans are images, not searchable text — selecting or copying from them returns nothing.
2. **OCR to plain text** with [fastOCR.org](https://fastocr.org). The output is solid overall and a good starting point, but it carries two kinds of error: page numbers and other print artifacts land in the middle of the text, and the OCR misreads some characters or layout — most often where Persian script, diacritics, or tables confuse it.
3. **Clean and structure with an LLM.** Feed the OCR text to Kimi K2.6 or GLM 5.2 in agent mode through the provider's web UI. The model reads the book chapter by chapter and writes a cleaned markdown version of each.
4. **Review and revise.** The model asks clarifying questions and proposes fixes; I answer and correct until the markdown reads cleanly to both a human and an LLM. The result lands in `data/raw/`.

Neither AI step is byte-reproducible. fastOCR.org runs an AI model, so a second OCR of the same scan may differ slightly, though it stays far more stable than the cleaning step. The cleaning step is interactive and depends on my judgment during the back-and-forth, so two runs will differ more. Treat the markdown as a reviewed artifact, not a deterministic build output.

## Cleaning prompt

This prompt opens step 3. It tells the agent to clean, structure, keep only the educational content, and ask before assuming:

The prompt is reproduced verbatim, typos and all — it is the exact text fed to the agent, so it stays unedited:

```text
analyze this file and do the following : 
1. clean the text, remove the page numbers / irrelevant number (page numbers are caused by OCR, its an ocr of a book)
2. put it in markdown format
3. do any other cleaning you can think of 
4. I onlt wanna keep the educational content, no metadata etc
Do not make any assumption, ask before going further
I want the output file in a .md file (no matter how big) 
read every chapter, write a cleaned version of it, move to the next. use raw reading and writing, no script!
```

Two instructions carry most of the weight. "Ask before going further" forces the model to surface ambiguous OCR rather than guess at it. "No script" keeps the model reading and rewriting the text directly instead of running regex passes that strip real content along with the page numbers.

## Persian-specific care

The OCR misreads Persian script in predictable ways: it confuses the Arabic ي/ك with the Persian ی/ک, drops or inserts ZWNJ, and mixes Persian and Arabic digit forms. The cleaning step should preserve the correct Persian forms, not normalize them away — normalization belongs downstream in `src/data/persian.py`, where it is applied consistently to both the corpus and the query. A spot check of a few cleaned passages against the source PDF catches the worst OCR substitutions. These quirks are first-class concerns for this corpus, not afterthoughts.

## Source inventory

One row per file in `data/raw/`. All sources are the 1403–1404 edition, all cleaned with GLM 5.2.

| File | Content | Edition | Cleaning model | Added |
| --- | --- | --- | --- | --- |
| `farsi-9th-grade-textbook.md` | Official 9th-grade Persian textbook | 1403–1404 | GLM 5.2 | 2026-06-19 |
| `farsi-9th-grade-gifted-textbook.md` | Persian & composition, gifted-schools (استعدادهای درخشان) edition | 1403–1404 | GLM 5.2 | 2026-06-19 |
| `farsi-9th-grade-study-guide.md` | Step-by-step study guide for 9th-grade Persian | 1403–1404 | GLM 5.2 | 2026-06-19 |

## Limitations

- **Neither step is reproducible byte-for-byte.** Both OCR and cleaning run AI models behind a web UI. The OCR is the more stable of the two, but only the cleaning model is recorded per file (the table above); a rerun of either will differ.
- **OCR errors can survive review.** A misread character that still forms a plausible Persian word will pass both the OCR and a quick read. The spot check reduces this risk but does not remove it.
