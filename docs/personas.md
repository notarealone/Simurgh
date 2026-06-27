# Learner Personas

> The profile schema that threads through the whole pipeline (rewriting → retrieval →
> generation). Defines the axes, the scale, the four personas, and the train/test split.
> The detailed plan lives here; positioning in [methodology](methodology.md), [experiment-design](experiment-design.md),
> [things-to-consider](things-to-consider.md). Grounded in [simulating-students-with-large-language-models-a-review-of-architecture-mechanisms-and-role-modelling-in-education-with-generative-ai](references/simulating-students-with-large-language-models-a-review-of-architecture-mechanisms-and-role-modelling-in-education-with-generative-ai.md), [teaching-according-to-students-aptitude-personalized-mathematics-tutoring-via-persona-memory-and-forgetting-aware-llms](references/teaching-according-to-students-aptitude-personalized-mathematics-tutoring-via-persona-memory-and-forgetting-aware-llms.md), [two-tales-of-persona-in-llms-a-survey-of-role-playing-and-personalization](references/two-tales-of-persona-in-llms-a-survey-of-role-playing-and-personalization.md).

---

## 1. Two senses of "persona" — the two hats

"Persona" is overloaded ([two-tales-of-persona-in-llms-a-survey-of-role-playing-and-personalization](references/two-tales-of-persona-in-llms-a-survey-of-role-playing-and-personalization.md)), and Simurgh uses **both** senses. Stating which is which prevents a methodological leak:

- **Role-playing (the simulator/judge).** A big model *acts as* a learner with this profile to decide which rewrite/answer is better *for that student*. This produces the DPO preference labels.
- **Personalization (the system).** The rewriter consumes the profile to serve the learner — and so does the generator, on the persona-aware arm.

The same schema below feeds both. The persona that **labels** preference pairs (role-play) must stay distinct from the rubric that **scores** evaluation — the judge-independence rule in [experiment-design](experiment-design.md), now also a persona-level rule: do not let the simulating persona double as the grading persona.

## 2. The four axes

The four axes come in two kinds: two **competence** axes (a real better/worse ordering) and two **orientation** axes (a preference between two poles, with no "better" end). Big Five / affective traits are **deliberately excluded**: they steer multi-turn conversational tone, not a single-turn answer to a textbook question — and the review flags prompt-encoded personality as prone to idealized, homogenized behavior ([simulating-students-with-large-language-models-a-review-of-architecture-mechanisms-and-role-modelling-in-education-with-generative-ai](references/simulating-students-with-large-language-models-a-review-of-architecture-mechanisms-and-role-modelling-in-education-with-generative-ai.md)).

| Axis | Type | Poles / quality | Primarily affects |
|---|---|---|---|
| **Comprehension level** | competence | weak reader → strong abstract reasoner | retrieval + phrasing |
| **Prior knowledge** | competence | no topic exposure → deep background | **retrieval** (prerequisites) |
| **Learning goal** | orientation | exam-pass ↔ deep-understanding | **retrieval** (answer- vs explanation-bearing passages) |
| **Explanation style** | orientation | concise-direct ↔ elaborated-with-examples | phrasing (generation-side) |

Three of four shift *what to retrieve* — the lever the trained rewriter actually pulls — so persona effects show up in retrieval diagnostics (Recall@K/MRR per persona), not only in the judge score. Explanation style is mostly a generation-side knob, relevant on the persona-aware generator arm.

## 3. Scale and representation

- **One 4-level ordinal everywhere.** Every axis is stored canonically as a level **L1–L4**. The uniform ordinal keeps split-balancing, ablation, and per-axis analysis tractable.
- **Labels differ by axis type.** Competence axes read **Bad / Average / Good / Excellent**; orientation axes read as a 4-step scale between named poles (e.g. learning goal L1 = *exam-focused* … L4 = *mastery-focused*). The underlying L1–L4 is identical; only the rendered words change.
- **Store ordinal, prompt prose.** The canonical record is the ordinal tuple; for the model it is **rendered to a natural-language description**. Numbers in a prompt steer behavior unreliably, prose reliably — a representation choice the closest tutoring precedent (TASA: persona = NL description + concept keywords) confirms ([teaching-according-to-students-aptitude-personalized-mathematics-tutoring-via-persona-memory-and-forgetting-aware-llms](references/teaching-according-to-students-aptitude-personalized-mathematics-tutoring-via-persona-memory-and-forgetting-aware-llms.md)).

Per-level meanings:

**Comprehension level** — Bad: struggles with textbook prose; needs short sentences, defined terms, step-by-step. Average: follows straightforward explanations, trips on abstraction/multi-step. Good: handles standard explanations and moderate abstraction. Excellent: parses dense, abstract phrasing; comfortable with terse, high-level treatment.

**Prior knowledge** — Bad: no exposure; prerequisites and definitions must be retrieved alongside the answer. Average: shaky/partial background; key prerequisites must be surfaced. Good: solid background; prerequisites can be assumed. Excellent: deep background; can skip basics and follow advanced connections.

**Learning goal** — L1 exam-focused: wants the answer and exactly what scores marks, minimal "why". L2 mostly-exam: answer first, brief justification. L3 mostly-understanding: explanation prioritized, answer in context. L4 mastery-focused: wants the underlying why/how and connections, length tolerated.

**Explanation style** — L1 terse: shortest correct answer, no padding. L2 compact: answer + one-line justification. L3 explained: worked reasoning. L4 elaborated: examples, analogies, scaffolding, restated simply.

Example of the store→prose rendering (persona P1 below):

```
canonical:  {comprehension: L1, prior_knowledge: L1, learning_goal: L1, explanation_style: L4}
rendered →  "A ninth-grader who finds the textbook hard to follow and has little background
             on this topic. Mainly wants to pass the exam — give the answer and what is needed
             to score — but it must be spelled out simply, step by step, with examples."
```

(This doc writes the rendered text in English per the repo's English-everywhere rule; the runtime emits it in the prompt language — `prompt_variant: en|fa`, see [methodology](methodology.md) — so a Persian-answering model can receive a Persian prompt.)

## 4. The four personas

| ID | Handle | Comprehension | Prior knowledge | Learning goal | Explanation style | Split |
|---|---|---|---|---|---|---|
| P1 | `crammer` | Bad (L1) | Bad (L1) | exam-focused (L1) | elaborated (L4) | train |
| P2 | `scholar` | Excellent (L4) | Good (L3) | mastery (L4) | terse (L1) | train |
| P3 | `steady` | Good (L3) | Average (L2) | mostly-exam (L2) | compact (L2) | train |
| P4 | `newcomer` | Good (L3) | Bad (L1) | mastery (L4) | elaborated (L4) | **test-holdout** |

Prose renderings (the canonical English source; the runtime renders to the prompt language):

- **P1 `crammer` — struggling exam-crammer.** Finds textbook prose hard, little background on the topic, mainly wants to pass — but needs everything spelled out simply with examples. (Weak + exam-driven, yet still needs scaffolding — a realistic, non-contradictory combination.)
- **P2 `scholar` — advanced deep-diver.** Reads dense material easily, solid background, wants the underlying *why* and connections, and prefers it delivered tersely without hand-holding.
- **P3 `steady` — solid pragmatist.** A capable, typical student with average background; wants a correct answer with brief justification, balanced toward exam needs.
- **P4 `newcomer` — curious newcomer (held out).** Bright and reads well, but new to this topic; wants real understanding and needs worked examples to bridge the missing background.

**Separation (homogenization hedge).** The three training personas differ on **all four axes** pairwise — no two are adjacent — so a persona effect, if real, has room to show. This counters the review's warning that LLMs collapse weakly-separated learners into one idealized voice ([simulating-students-with-large-language-models-a-review-of-architecture-mechanisms-and-role-modelling-in-education-with-generative-ai](references/simulating-students-with-large-language-models-a-review-of-architecture-mechanisms-and-role-modelling-in-education-with-generative-ai.md)).

## 5. Train/test split and the generalization claim

Training and validation use `crammer`, `scholar`, and `steady`; `newcomer` **appears only at test**. The split targets a stronger claim than "handles a 4th preset":

- **Every axis value in `newcomer` appears in some training persona** — Comprehension *Good* (in `steady`), Prior knowledge *Bad* (in `crammer`), Learning goal *mastery* (in `scholar`), Explanation style *elaborated* (in `crammer`).
- **The *combination* never appears in training.** `newcomer` is therefore an unseen *recombination*, so a gain on it tests **compositional generalization** — did the rewriter learn the axes, or memorize three templates?
- It also deliberately **decorrelates comprehension from prior knowledge** (high comprehension, low prior knowledge — the bright-but-new student), the case real data rarely isolates and LLMs tend to smooth over.

This is the "generalization to an unseen profile" framing carried from [methodology](methodology.md) and [experiment-design](experiment-design.md), made concrete.

## 6. Where the profile enters the pipeline

- **Always into the rewriter** — the profile is part of the policy input `π(rewrite | profile, query)`.
- **Into the generator only on the persona-aware arm** — the persona-aware-vs-blind comparison ([methodology](methodology.md), [experiment-design](experiment-design.md)).
- **Every personalized path has a "no profile" switch** (the "profile is the contract" rule) — drop the profile and the component reverts to generic behavior, which is exactly the persona-removed ablation.

## 7. How each axis should move a rewrite (illustrative, not a rule table)

- **Prior knowledge ↓** → expand the query with prerequisite/definition terms so the retriever also surfaces foundational passages; **↑** → assume basics, target the specific point.
- **Comprehension ↓** → bias toward passages with plain, worked explanations; **↑** → tolerate dense/abstract source passages.
- **Learning goal toward exam** → favor answer-bearing, summary, "key points" passages; **toward understanding** → favor explanation/derivation/example passages.
- **Explanation style** → mostly shapes the *generator's* phrasing; its retrieval effect is secondary (e.g. pulling an extra worked example for the elaborated pole).

These are hypotheses the DPO signal is meant to *discover*, not hard-coded rules — the rewriter is trained, not scripted.

## 8. Validity and honest caveats

- **Synthetic personas, no real students** — consistent with the constraints; honesty about this stays in the thesis.
- **Behavioral homogenization** → mitigated by well-separated values (§4) and by the decorrelated holdout (§5).
- **Knowledge leakage** in the simulator (expert knowledge bleeding into the "student") → frame the simulator as *"act as a teacher imagining how this student would respond"* ([simulating-students-with-large-language-models-a-review-of-architecture-mechanisms-and-role-modelling-in-education-with-generative-ai](references/simulating-students-with-large-language-models-a-review-of-architecture-mechanisms-and-role-modelling-in-education-with-generative-ai.md)).
- **Validation gap** (≈half of simulated-learner studies validate nothing) → the 50–100-sample human check in [experiment-design](experiment-design.md) is the guard; it must confirm the personas read as *distinct and plausible* 9th-graders.
- **Marginal-uplift caveat** — on a small (3-book) corpus the rewriter's persona effect may be modest, and a capable persona-aware generator may absorb it; a null/modest result is reportable ([things-to-consider](things-to-consider.md)).

## 9. Open items

- [ ] Final prose wording for each persona (and its Persian rendering); freeze before the eval set is built.
- [ ] Confirm L-values once a few pilot rewrites are inspected — tune `steady`/`newcomer` if two personas read too alike to the judge.
- [ ] Simulator/judge prompt template implementing the "imagine a student" leakage guard.
- [ ] Persian rendering must respect ZWNJ / ye-ke / digit normalization so any persona-driven query expansion matches the index ([methodology](methodology.md)).
