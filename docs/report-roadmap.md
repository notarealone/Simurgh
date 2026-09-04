# Final Report Roadmap

> How to turn this repo's docs into the Persian BSc report, in the register and structure of
> `BScThesis_template_new.docx`. The roadmap is English (repo rule); the report itself is
> Persian. Every item names its source doc, what must be *written*, and what must be
> *expanded* because the doc does not yet carry it.

---

## 0. What the template actually demands (measured, not guessed)

Extracted from `BScThesis_template_new.docx`:

| Property | Value |
|---|---|
| Chapters | 5 (intro, background+related work, framework/method, experiments, conclusion) |
| Heading levels | 11 × H1, 29 × H2, 42 × H3 |
| Body length | ≈9,800 words: ch1 2,206 / ch2 2,674 / ch3 1,371 / ch4 2,686 / ch5 838 |
| Figures / tables | 12 / 11 |
| Footnotes | 84 — almost all first-use English glosses of Latin terms |
| References | 8, Chicago author-date |
| Section numbering | `chapter-section-subsection` (۲-۳-۱) |
| Figure/table numbering | `index-chapter` (شکل ۳-۲ = third figure of chapter 2) — reversed vs sections |
| Front matter | title page, تعهدنامه اصالت اثر, تقديم, تشكر و قدرداني, چکيده + کلمات کلیدی, 4 lists (contents, figures, tables, abbreviations) |
| Back matter | مراجع, English Abstract + Keywords |

Two defects in the template not to copy: its chapter-1 roadmap sentence advertises a section
`۱-۴ پرسش‌های پژوهش` that does not exist (real headings are `۱-۳ اهمیت مسئله`, `۱-۴ ساختار
گزارش`), and `۴-۴-۲` is used twice (for both روش and نتیجه).

Register rules for the Persian prose live in the `persian-technical-writing` skill. The
load-bearing ones: negation-then-assertion carries every judgment; numbers argue and prose
does not; each result states what it does *not* mean; hypotheses are labelled as hypotheses;
tables get a weakness column and are then read aloud in prose.

## 1. Source inventory → chapter map

| Report location | Primary sources in `docs/` | Readiness |
|---|---|---|
| چکیده / Abstract | `thesis-proposal.md`, `results/*` headline numbers | write last |
| ۱ مقدمه | `thesis-proposal.md` (A/B/C + revisions v1.1–v1.4) | high (needs Persian stakes rewrite) |
| ۲-۱ RAG/RL foundations | `literature-review.md` | **thin** — 2 papers only |
| ۲-۲ related work | `references/*.md` (5 notes) | **thin** — 5 logged papers vs ~20 needed |
| ۲-۳ Persian NLP | `data-extraction.md` (ي/ك, ZWNJ, digits), `methodology.md` Phase 0 | **thin** — zero citations |
| ۲-۴ datasets & metrics | `experiment-design.md`, `data-design.md`, `question-extraction.md` | high |
| ۲-۵ research gap | `literature-review.md` "Positioning" + `methodology.md` "structural difference from LaMP" | high |
| ۳-۱ architecture | `methodology.md` (System Architecture + Inference) | high |
| ۳-۲ personas/profile | `personas.md` | high (prose/Persian rendering unfrozen) |
| ۳-۳ data construction | `data-design.md`, `data-extraction.md`, `question-extraction.md` | high |
| ۳-۴ Stage 1 ROPG-KD | `methodology.md` Stage 1 | very high (over-length; must be cut) |
| ۳-۵ Stage 2 DPO | `methodology.md` Stage 2 | very high |
| ۳-۶ metrics & protocol | `experiment-design.md` Metrics/Protocol | high |
| ۴-۱ setup | scattered; `results/*` inputs blocks | **incomplete** — no environment record |
| ۴-۲ judge reliability | `results/judge-noise-luna-v1.md`, `results/judge-agreement-luna-gemini-v1.md` | very high |
| ۴-۳ data audit | `results/dpo-data-v4-audit.md` | very high |
| ۴-۴ Stage 1 results | `results/ropg-runs-comparison-v1.md` (+ `stage1-ropg-runB.md` as history) | very high |
| ۴-۵ Stage 2 results | `results/dpo-arms-seed42-v1.md` | high but **null result only** |
| ۴-۶ end-to-end ladder | — | **missing entirely** |
| ۵ جمع‌بندی/محدودیت‌ها | `things-to-consider.md` (Open items), every `gaps` above | very high |

Load asymmetry to plan around: `docs/` holds ~44k words, but it is concentrated in method and
diagnostics. Chapters 3 and 4 are over-supplied and must be *cut*; chapters 2 and the
end-to-end part of 4 must be *grown*.

---

## Phase A — decisions before any Persian sentence

- [ ] **A1. Freeze the report's claim, in one negation-then-assertion sentence.**
  The evidence supports: *personalization of a Persian educational RAG pipeline is not
  obtained by training a retriever on judge-scored document utility; the retriever's gain
  there is generic retrieval quality, and the persona signal has to be carried elsewhere.*
  Source: `ropg-runs-comparison-v1.md` (persona-swap −0.0017/−0.0043/−0.0053/+0.0008, all
  p = 1.0000) plus `methodology.md` (persona-blind ceiling 0.9573 vs matched 1.0000 →
  headroom 0.0427).
  Acceptance: the sentence appears in چکیده, ۱-۲, ۴-۶ and ۵-۵ with identical terminology.

- [ ] **A2. Write 4 research questions and pre-assign each an answering experiment.**
  Candidate set, all answerable from committed evidence: (1) does judge-distilled retriever
  training improve retrieval on Persian textbook text? (`ropg-runs-comparison-v1`); (2) does
  it personalize? (persona-swap control); (3) can persona-conditioned rewriting be learned
  offline from proxy-judge preference pairs? (`dpo-arms-seed42-v1`); (4) how reliable is the
  measuring instrument itself? (`judge-noise-luna-v1`, `judge-agreement-luna-gemini-v1`).
  Acceptance: every question maps to a `۴-x` section, and `۵-۲` answers all four in the same
  wording.

- [ ] **A3. Decide the scope sentence for what is *not* claimed.**
  End-to-end rungs 1–4 were never run (`experiment-design.md` Baselines: only Rung 0 is
  `[x]`); the primary metric (persona-alignment judge on a held-out test set) and the 50–100
  human validation are both unrun. Either run them (Phase D) or state in ۱-۳ and ۵-۳ that
  the report evaluates the two trained components in isolation, not the assembled system.
  Acceptance: no sentence in the report implies an end-to-end gain.

- [ ] **A4. Fix numbering conventions and the two template defects.**
  Sections `فصل-بخش-زیربخش`; figures/tables `شماره-فصل`; no duplicate ids; roadmap sentences
  list only sections that exist.

- [ ] **A5. Build the label-era guard into the writing plan.**
  `methodology.md` requires every Stage-1 number to name nano vs Luna era. In the Persian
  text this becomes a fixed formula (e.g. «برچسب‌های دورهٔ Luna») attached to every table
  caption in ۴-۴, and the nano-era Run B numbers appear only in a clearly-marked historical
  subsection.
  Acceptance: no table mixes eras; no Luna number is compared against 0.548.

## Phase B — grow what is too thin (do this in parallel with Phase C)

- [ ] **B1. Expand `literature-review.md` from 2 papers to a chapter-2 backbone.**
  Current file covers DPO and ROPG/LaMP only, deliberately ("not a broad RAG/RL survey").
  A background chapter needs the pipeline foundations too: RAG itself, dense vs lexical
  retrieval, query rewriting for RAG, PEFT, and the RLHF→DPO lineage.
  Missing-citation list (no logged paper today, one `references/` note each):
  RAG (Lewis et al. 2020) · BM25 (Robertson & Zaragoza 2009) · DPR (Karpukhin et al. 2020) ·
  Contriever (Izacard et al. 2022) · query rewriting for RAG (Ma et al. 2023 "Rewrite-Retrieve-Read";
  HyDE, Gao et al. 2023) · LaMP benchmark (Salemi et al. 2023, distinct from the 2024 ROPG paper) ·
  RLHF/PPO (Christiano et al. 2017; Ouyang et al. 2022) · LoRA (Hu et al. 2021) · QLoRA (Dettmers et al. 2023) ·
  WPO (Zhou et al. 2024) · robust/conservative DPO label smoothing (Chowdhury et al. 2024) ·
  RPO supervised anchor (Liu et al. 2024) · MNRL / multiple-negatives ranking (Henderson et al. 2017) ·
  nDCG (Järvelin & Kekäläinen 2002) · LLM-as-judge reliability and position bias (Zheng et al. 2023) ·
  Qwen3 and Qwen3-Embedding technical reports · Persian NLP: hazm, ParsBERT (Farahani et al. 2021),
  Persian normalization/ZWNJ treatment · bootstrap (Efron & Tibshirani 1993) · Holm (1979) · McNemar (1947).
  Acceptance: each cited work has a `docs/references/<slug>.md` from `TEMPLATE.md`, is
  registered in `INDEX.md`, and is actually cited in the Persian text. Target ~20 references
  (template has 8; a personalized-RAG+RL thesis cannot be argued on 8).

- [ ] **B2. Add a `references/` note for every method arm the report names.**
  `wpo`, `robust_dpo`, and `rpo_alpha` are used as experimental arms in
  `experiment-design.md` with no logged source. Un-sourced arms read as arbitrary.
  Acceptance: ۳-۵'s arm table has a citation per row.

- [ ] **B3. Create a Persian-NLP subsection with real sources.**
  `data-extraction.md` names the hazm normalizer, ZWNJ preservation, ي/ك folding and
  Persian/Arabic digit folding; `src/data/persian.py` implements it. Nothing is cited and no
  measurement exists.
  Expand: cite hazm and one Persian-NLP paper, and add one measured sentence — e.g. count how
  many corpus chunks contain Arabic ye/kaf before normalization, and what BM25 Recall@K does
  with normalization off. That is a cheap, honest, chapter-2-worthy number.
  Acceptance: ۲-۳ contains at least one measured Persian-specific number, not just a list of
  risks.

- [ ] **B4. Mark the base paper explicitly in `references/INDEX.md`.**
  `INDEX.md` has no base-paper flag; the report's "we extend X" claim needs it. ROPG-KD
  (Salemi et al. 2024) is the extended method; DPO is the companion method.
  Acceptance: INDEX marks base vs companion vs context, and ۲-۵ states the border with the
  base paper in one paragraph (shared 171-chunk corpus vs LaMP's 55–205 docs *per user* —
  from `methodology.md`, "A structural difference from LaMP").

- [ ] **B5. Record the missing reproducibility metadata now, while it is still recoverable.**
  Every `results/*` doc is missing the same fields, and chapter ۴-۱ cannot be written without
  them: GPU model/count (2×T4 appears only incidentally), CUDA/torch/TRL/PEFT/transformers
  versions (only TRL 0.24.0 is recorded), commit hash per run, datagen seeds (none recorded
  for question generation, rewrite candidates, or judge calls), corpus statistics (171 chunks
  is known; chunk window/overlap, index build version, embedding dim are not), OCR
  engine/version (fastOCR.org, no version), VLM model/version/temperature for question
  extraction.
  Acceptance: one `docs/results/environment.md` table that ۴-۱ can be written from verbatim.

- [ ] **B6. Decide and document the corpus/index numbers the report will quote.**
  `data-design.md` lacks a current chunk count and any chunking config; the only count in the
  repo prose is the 171-chunk ROPG corpus.
  Acceptance: chunk count, window, overlap, embedder `max_seq_length: 2048`, top-K, and index
  version all appear in one place and match the configs.

## Phase C — write the chapters, in dependency order (not reading order)

- [ ] **C1. ۴-۱ بسترهای آزمایش.**
  From B5. Include the deliberate-dualism sentence: local/Kaggle 2×T4 for training vs API
  models for judging, and why each is needed («این دوگانگی عمدی است»).
  Acceptance: a reader can reproduce each run from this section alone; one table with a
  "what this platform makes possible" column.

- [ ] **C2. ۴-۲ آزمایش یکم — the instrument before the results.**
  Write judge reliability first, because every later number is measured with it. From
  `judge-noise-luna-v1.md`: Luna at `temperature: 0` is not deterministic — 18.0% byte-identical
  replies (CI [0.141, 0.227]), single-draw SD 0.0265 (p90 0.041, max 0.137), 900 calls, 0
  failures; noise is heteroscedastic (SD 0.041–0.043 in the 0.40–0.70 band vs 0.0101 above
  0.85); flip rate 0.244 overall, 0.088 at margin 0.04; K=3 gives σ 0.0153. From
  `judge-agreement-luna-gemini-v1.md`: Luna vs Gemini-3.7 0.688 (n=100), vs Gemini-3.5-lite
  0.641 (n=250), Gemini-vs-Gemini 0.730, McNemar p=0.541.
  Register: this is the section that carries «معیارهای موجود ... اما», the "what the number
  does not mean" bind (0.244 flip rate is not 24% wrong pairs — it is 24% of *pairs* whose
  ordering is unstable at zero margin), and an explicit honesty note that no kappa was computed
  and that only replicate noise, not prompt sensitivity, was measured.
  Acceptance: `min_margin: 0.04` is justified *from* the measurement, not asserted.

- [ ] **C3. ۴-۳ آزمایش دوم — as-built data audit.**
  From `dpo-data-v4-audit.md`: 2,658 train / 534 val pairs over 432/92 questions, 96.9%/97.1%
  group yield, 18,864 scoring calls, modeled label error 0.08% (vs ~27% implied for format 1),
  cross-persona discrimination +0.180 correct in 97.1% of 1,564 rows, `persona_fit` Δ +0.468
  positive in 100%, zero contradictions/duplicates/split crossings.
  Must include the two honest notes: scholar sits on the rejected side 1.6× more often
  (ratio 0.61; 33 of 40 empty groups) and 58.8% of pairs are cross-persona, i.e. the easier
  discrimination — so a high preference accuracy must be checked against `pair_type` before
  being read as "learned quality ranking".
  Acceptance: the temperature ladder is reported as a negative finding (means
  0.796/0.798/0.797/0.795; wins 312/314/339/331 — indistinguishable), and the order-statistic
  explanation (best−worst margin 0.1288 at 4 candidates, 0.1063 at 3, 0.0619 at 2) replaces it.

- [ ] **C4. ۴-۴ آزمایش سوم — Stage 1 ROPG runs A–D.**
  From `ropg-runs-comparison-v1.md`: selected Run B — nDCG@5 0.7307, Recall@5 0.6461,
  Hit@5 0.9710, MRR 0.8538, `judged@5` 0.8688 against the Luna-era epoch-0 baseline
  0.5861 / 0.5640 / 0.7853 / 1.0000; B's Recall@5 deltas vs A/C/D +0.0616 / +0.0338 / +0.0701
  with Holm p 0.0013 / 0.0052 / 0.0013.
  Then the section's real point, in negation-then-assertion form: the gain is generic
  retrieval, not personalization — all four persona-swap tests are null (p = 1.0000)
  although ranking with a mismatched (rotated) persona costs 0.1363 nDCG@5 under the
  same labels, so the control has power and still returns nothing.
  Required guards to state: `judged@5` must accompany every nDCG (a model surfacing
  unjudged-but-relevant chunks is punished for improving; epoch-0's 1.0 is structural);
  the metric shares the training labels' noise (circularity); CIs cover validation-query
  uncertainty only, not seed variance — everything is seed 42.
  Acceptance: the historical nano-era Run B (nDCG@5 0.548 → 0.522, criterion failed) appears
  as a labelled history subsection, and the post-hoc metric change (nDCG@5 → nDCG@1 + Recall@5)
  is justified in the text rather than hidden, exactly as `experiment-design.md` does.

- [ ] **C5. ۴-۵ آزمایش چهارم — Stage 2 DPO screening, a null result reported as a result.**
  From `dpo-arms-seed42-v1.md`: 272 prompts × 10 pairings = 2,720 judge calls, 0 missing;
  round-robin grok 0.7054 > dpo 0.4922 > base_qwen 0.4848 > wpo 0.4343 > robust 0.3833;
  DPO-vs-base 0.518 with Holm p 0.502 and 95% CI [0.469, 0.566] — a clean null; robust worse
  (0.406, p 0.006); grok 0.688 (p 4.0e-9). Diagnosis: margins grew by pushing both
  log-probabilities down (Pearson r 0.991), so selection moved to `eval_rewards/accuracies`,
  one epoch, and an RPO anchor at `rpo_alpha: 1.0`.
  Register: this is where «گزارشِ یافتهٔ منفی» is mandatory. State the pre-registered
  failed-transfer rule from `experiment-design.md` and that it fired, then that format-2 data
  was the response — and that the retrained arms and seeds 43/44 are **not yet run**, so the
  section ends in an explicitly open state rather than an implied win.
  Acceptance: no sentence implies the rewriter works; the format-1/format-2 boundary is
  stated (results are format-1, current data is format-2, `load_pairs` refuses to mix).

- [ ] **C6. ۴-۶ جمع‌بندی فصل — the ladder.**
  Numbered rungs, one measured number each: instrument noise (SD 0.0265) → data quality
  (0.08% modeled error) → retriever gain (Recall@5 +0.0821 over baseline) → personalization
  null (persona-swap p = 1.0000) → rewriter null (DPO-vs-base 0.518, p = 0.502).
  Acceptance: every rung carries its number and its era/seed label.

- [ ] **C7. ۳-۱ مدل مسئله و معماری.**
  From `methodology.md` System Architecture + Inference: the two trained components, the frozen
  generator, the "profile is the contract" switch, and the ASCII data flow redrawn as
  شکل ۱-۳ with a full descriptive caption.
  Acceptance: the figure shows where the profile enters and where the no-profile switch is.

- [ ] **C8. ۳-۲ قرارداد پروفایل یادگیرنده.**
  From `personas.md`: 4 axes (comprehension, prior knowledge, learning goal, explanation
  style), ordinal L1–L4 stored, prose rendered; `crammer` (L1,L1,L1,L4), `scholar`
  (L4,L3,L4,L1), `steady` (L3,L2,L2,L2) train; `newcomer` (L3,L1,L4,L4) test-holdout as an
  unseen recombination → the claim is compositional generalization.
  Expand before writing: the persona prose and its Persian rendering are **not frozen**, and
  `newcomer` has never been evaluated. Either freeze + evaluate, or declare in ۵-۳ that the
  compositional-generalization claim is designed but untested.
  Acceptance: the two senses of "persona" (simulator/judge vs system input) are separated in
  the first paragraph, as `personas.md` does, and the judge-independence rule is stated.

- [ ] **C9. ۳-۳ ساخت دادگان.**
  From `data-extraction.md` (scanned PDFs → OCR → LLM cleaning → review; ي/ك, ZWNJ, digit
  risks), `question-extraction.md` (VLM → per-exam JSON schema, join on `number`, type
  taxonomy), `data-design.md` (three corpus files and their persona affinity, section-type
  chunking, ROPG and DPO row schemas, question-level split seed 42 / 70-15-15).
  Mandatory honesty content: 487/618 questions (78.8%) are GLM-5.2-generated with
  model-authored, unverified answers and explanations, and the test split is ~80.9% generated
  gold; the split is question-disjoint but **source-overlapping** with no `group_id`
  isolation, and the frozen DPO/ROPG assets are pinned to it.
  Acceptance: ۳-۳ states the leakage boundary in the same words `CLAUDE.md` uses, and the
  synthetic-gold share appears as a number, not an adjective.

- [ ] **C10. ۳-۴ مرحلهٔ یکم — ROPG-KD, cut to ~700 Persian words.**
  `methodology.md` Stage 1 is ~180 lines; the report needs: teacher signal (0–1 utility per
  `(query, persona, document)` triple, 20 calls per group, 31,440 calls), objective
  (`hard_neg` MNRL primary, `reader_kd` retained ablation) and the *reason* — teacher
  single-label reliability ≈0.17 in the nano era means KD spends its gradient on adjacent-rank
  gaps of 0.02–0.03, below the noise floor, while MNRL reads the 0.711 rank-1-vs-tail contrast;
  anchoring (λ_doc 0.5 / λ_query 0.05, personalization is query-side only); filters
  (0.05 / 0.4 / 0.25 → 929/1296 train, 206/276 val); encoder parity gate (cosine ≥ 0.999).
  Everything else — padding waste, gradient checkpointing, NCCL flags, LR-scheduler warning —
  goes to a footnote or is dropped. They are engineering, not argument.
  Acceptance: the metric-definition paragraph fits one table (nDCG@1..5, Recall@5, MRR,
  Hit@K, `judged@5`, persona-swap) with a "what it ignores" column.

- [ ] **C11. ۳-۵ مرحلهٔ دوم — DPO rewriter, cut to ~700 Persian words.**
  Keep: policy/reference (Qwen3-4B 4-bit + LoRA, reference = adapter disabled), proxy-judge
  pair construction and why the end-to-end alternative was rejected (≈3× API cost —
  `thesis-proposal.md` v1.3), the three arms and their purpose, sequence caps 576/224/768 with
  the refuse-to-truncate preflight, checkpoint selection on `eval_rewards/accuracies` and why
  `eval_loss` ranks nothing, the independent-judge tournament design, and the pre-registered
  failure rule.
  Acceptance: the off-policy chain (Grok generates → Luna ranks → Qwen3-4B is the policy) is
  stated explicitly, because it is the leading explanation for C5's null.

- [ ] **C12. ۳-۶ معیارها و پروتکل ارزیابی.**
  From `experiment-design.md`: primary = persona-alignment/pedagogical judge on held-out test
  + human sample; secondary = EM/F1 and Recall@K/MRR as *diagnostics*; paired bootstrap CIs +
  Holm-corrected sign-flip over ≥3 seeds; judge independence.
  Honest note required: the primary metric has not been measured yet and only seed 42 exists,
  so ۳-۶ describes the protocol while ۴ reports a subset of it. Say so here, not only in ۵-۳.
  Acceptance: ۳-۶ contains a "what has been executed / what has not" line.

- [ ] **C13. ۳-۷ جمع‌بندی فصل + tool-cost table.**
  Mirror the template's جدول ۱-۳ ("tools introduced in chapter 3 and the cost of each"):
  per component, what it answers and what it costs in API calls (ROPG teacher 31,440 calls at
  ≤16 completion tokens; DPO datagen 18,864 scoring calls; tournament 2,720 calls; the planned
  3-seed protocol 5,440 total).
  Acceptance: cost is quoted in calls, not money, and matches the results docs.

- [ ] **C14. ۲ فصل دوم — background and related work.**
  Order: ۲-۱ RAG and retrieval foundations → ۲-۲ preference optimization (RLHF/PPO → DPO and
  its variants) → ۲-۳ personalization and learner modelling (`two-tales`, `simulating-students`,
  TASA) → ۲-۴ Persian NLP → ۲-۵ datasets/metrics in prior work → ۲-۶ gap.
  Open the chapter with its own conclusion («نتیجهٔ این مرور دو ادعای مشخص است ...»), give each
  related work the fixed skeleton (هدف / سازوکار / محدودیت), and close with the gap in the same
  words ۳ will use.
  Quantitative anchors already available: LaMP's RSPG-Post best on 6/7 tasks, +5.5% over prior
  SOTA and +15.3% over a non-personalized LLM, ROPG-KD mid-table; DPO ≈61% vs PPO ≈57% on
  TL;DR, tested only up to 6B; ≈half of surveyed student-simulation studies report no formal
  validation.
  Acceptance: every ۲-x subsection ends in a limit sentence, and ۲-۶ names two concrete gaps —
  no Persian personalized-RAG benchmark, and no persona-dependent teacher signal in a
  shared-corpus setting.

- [ ] **C15. ۲-۵ معیارها و بنچمارک‌ها — with a weakness column.**
  Table rows: EM/F1, ROUGE, Recall@K/MRR, nDCG, LLM-judge rubric, human evaluation; columns
  "what it measures / strength / what it ignores". Then read the table in prose: the reason
  this thesis cannot inherit a metric is that every existing one scores *text*, while the
  claim is about *fit to a learner*.
  Acceptance: the first finding of this section is the negative one — there is no Persian
  personalized-RAG benchmark to compare against.

- [ ] **C16. ۱ فصل اول — intro, written after the numbers exist.**
  Structure: ۱-۱ زمینه و انگیزه (RAG is standard; conventional RAG treats every learner
  identically — `thesis-proposal.md` A) → ۱-۲ بیان مسئله in four steps: آنچه انتظار داریم /
  آنچه در عمل رخ می‌دهد / چرا توضیح رایج کافی نیست («persona in the prompt is enough») /
  ریشهٔ واقعی (a shared corpus means personalization must come from *reordering the same
  documents*, and nothing in an MNRL objective asks why this chunk beats the one another
  persona wanted) → ۱-۳ اهمیت مسئله (asymmetric cost: an explanation pitched above a
  ninth-grader is not merely suboptimal, it is unusable; at most one direct address to the
  reader) → ۱-۴ پرسش‌های پژوهش (A2) → ۱-۵ ساختار گزارش.
  Acceptance: ۱-۲-۳'s rejection sentence is short and bold, its evidence follows immediately
  (persona-swap null; 58.8% of pairs are the easy cross-persona contrast), and the roadmap
  sentence lists only sections that exist.

- [ ] **C17. ۵ فصل پنجم — ~800 words, five subsections, no new numbers.**
  ۵-۱ جمع‌بندی; ۵-۲ answers to the four questions in A2's wording; ۵-۳ محدودیت‌ها; ۵-۴ کارهای
  آینده; ۵-۵ سخن پایانی in negation-then-assertion form closed by a parallel triple.
  ۵-۳ must list, from `things-to-consider.md` and the audit gaps: single seed everywhere;
  synthetic and unverified gold (78.8%); source-overlapping split with no shared-passage
  isolation; teacher-metric circularity (nDCG graded by the same labels that trained the
  encoder); judge non-determinism and ~30% cross-judge disagreement; `cross_persona_min_margin`
  and `min_chosen_score` uncalibrated; scholar skew; unrun end-to-end ladder; `newcomer`
  holdout never evaluated; no human validation.
  ۵-۴ must be the deferred fixes, each labelled as hypothesis: reader-in-the-loop
  persona-conditioned teacher `Eval(y, M(φp(x,[d],persona)))` (computable on 100% of the bank —
  618/618 have answer and explanation, 350 closed-form), persona-contrastive negatives (64.7%
  of the 2,592 candidate additions are genuinely new; 29.5% collide with the row's own
  positive and must be dropped), a ~100-group human-labelled eval set, and on-policy Qwen
  candidates.
  Acceptance: every limitation traces to a number or a named open item; every future item says
  what it would measure.

## Phase D — the work the report needs but the repo has not run

Ordered by report damage if skipped.

- [ ] **D1. Run at least the retrieval rungs of the ladder (Rung 0 BM25 vs Rung 1 frozen
  Qwen3-Embedding vs Rung 3 Run B), Recall@K/MRR per persona on the same val set.**
  Without it, chapter 4 has no baseline-ladder table at all and `experiment-design.md`'s
  central "a gain counts only relative to the rung below" rule is unhonoured. Epoch-0 already
  *is* Rung 1 through the identical code path, so the missing piece is BM25 measured on the
  same rows.

- [ ] **D2. Seeds 43 and 44 for Stage 2 (and, if affordable, a second Stage-1 seed).**
  Every number in the repo is seed 42; `experiment-design.md` itself says seed-42 screening
  alone is not a result, and CIs currently cover query uncertainty only.

- [ ] **D3. Retrain the three arms on format-2 data with the RPO anchor and re-run the
  tournament.** The current ۴-۵ can only report a null on retired format-1 data while the
  shipped data is format-2 — a gap a defence committee will find.

- [ ] **D4. Human validation on 50–100 samples.** It is the only clean exit from the
  metric-circularity problem (`things-to-consider.md`), and the template's register expects a
  human-validated sample beside every judge claim.

- [ ] **D5. Evaluate the `newcomer` holdout.** Otherwise the compositional-generalization
  claim in ۳-۲ is a design, not a result, and must be demoted to ۵-۴.

- [ ] **D6. Cheap wins that buy a figure or a number each:** Persian normalization
  on/off retrieval delta (B3); `pair_type` split of preference accuracy (C3's warning);
  prompt-perturbation judge noise, since only replicate noise was measured (C2).

## Phase E — assembly and delivery

- [ ] **E1. Front matter** — title page, تعهدنامه, تقديم, تشكر (name each person's specific
  contribution, in template order: supervisor, collaborators, lab, family).
- [ ] **E2. چکیده then Abstract** — expectation → contradicting fact → one concrete stake →
  what the work does → negation-then-assertion verdict; keywords Persian + English in
  parentheses, «؛»-separated; the English abstract is a faithful translation, not a paraphrase.
- [ ] **E3. Abbreviations list** — every acronym used: RAG, DPO, WPO, RPO, MNRL, KD, ROPG,
  RSPG, LoRA, PEFT, nDCG, MRR, EM, F1, LLM, ZWNJ, DDP, CI. Each must actually appear in the text.
- [ ] **E4. Footnote pass** — first-use English gloss for every Latin term, template style
  (~84 footnotes there); never gloss a term twice.
- [ ] **E5. Terminology freeze** — one Persian term per concept, never rotated for variety
  («بازنویس پرسمان», «بازیاب», «سیاست», «داور»); build a glossary and grep for violations.
- [ ] **E6. Orthography pass** — ZWNJ, ezafe kasra, Persian ی/ک, هٔ, Persian digits and «٫»
  (0.7307 → ۰٫۷۳۰۷ in prose; keep Latin numerals inside code/config identifiers).
- [ ] **E7. Consistency pass** — every figure/table referenced in text and listed; bidirectional
  cross-references resolve; no duplicated ids; auto-updated TOC matches headings; every quoted
  number matches its `results/*` source and carries its era/seed label.
- [ ] **E8. Register audit** — no «من»/«قطعاً»/«جالب است»/self-praise/mid-sentence dash;
  at least one negation-then-assertion per chapter; every result states what it does not mean;
  every table read aloud in prose; delete any paragraph whose first sentence is filler.
