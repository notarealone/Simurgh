# Simulating Students with Large Language Models: A Review of Architecture, Mechanisms, and Role Modelling in Education with Generative AI

- **Authors:** Luis Marquez-Carpintero, Alberto Lopez-Sellers, Miguel Cazorla
- **Year:** 2025
- **Venue:** Computer Science Review 62 (2026) 101008 (arXiv:2511.06078)
- **Link:** https://arxiv.org/abs/2511.06078

---

## Summary

A review of how LLMs **simulate students** for tutoring, formative evaluation, and
teacher practice. It taxonomizes (a) the **dimensions** prior work uses to define a learner model,
(b) the **representation methods** for those dimensions, and (c) the **validity pitfalls** of
LLM-based student simulation. This is the methodological grounding for Simurgh's persona schema:
it is the paper that answers "which axes, what scale, and how do these simulations fail."

## Key Contributions

- **Dimension taxonomy** for learner models, in three families: **cognitive** (knowledge/mastery of concepts, prior knowledge, misconceptions, cognitive load), **affective/personality** (Big Five, emotional state, motivation/self-efficacy), and **behavioral** (learning style — flagged as *theoretically contested* — communication style, engagement).
- **Three representation families:** direct prompt-encoding (natural language; most common), knowledge-tracing models (numeric mastery parameters over time), and knowledge graphs / heuristics (interpretable but hand-curated).
- **Catalogue of validity failure modes:** behavioral homogenization (idealized/stereotyped learners), knowledge leakage (expert knowledge bleeding into the "student"), pervasive validation gaps (≈half of older studies did no formal validation), short-term-memory limits, and narrowed ZPD.
- **Best practices:** multi-component (cognitive + affective) profiles, structured prompting with mastered-vs-confused examples, uncertainty expressions for believability, and hybrid (expert + automated) validation.

## Relevance to This Thesis

This review **justifies Simurgh's persona-design choices** point for point. (1) The cognitive/affective/
behavioral taxonomy is the menu from which Simurgh draws its **4 axes** — and it supports leaning
**cognitive + goal + style** while **dropping Big Five** for a single-turn QA task where personality
does not drive the answer (the review notes prompt-encoded personality "may lead to overly idealized
behaviours"). (2) The representation survey endorses Simurgh's **ordinal-band-stored, prose-rendered**
scheme as the mainstream hybrid (categorical bands for proficiency + natural-language descriptions for
steering). (3) The validity failure modes map onto Simurgh's hedges: **separated band values**
(Bad vs Excellent, not adjacent) counter homogenization; the **50–100-sample human check**
([experiment-design](../experiment-design.md)) addresses the validation gap; and the *"act as a teacher imagining how a student
would respond"* framing mitigates knowledge leakage in the simulator/judge.

## Notes

- Distinguishes the **role-playing** simulator (the LLM *acting as* a student to label pairs) from the **personalization** system (serving the learner) — the same "two hats" split as [two-tales-of-persona-in-llms-a-survey-of-role-playing-and-personalization](two-tales-of-persona-in-llms-a-survey-of-role-playing-and-personalization.md); keep the simulator/judge separate from the eval judge (judge independence).
- Knowledge-tracing and knowledge-graph representations require data/curation Simurgh lacks; the **prompt-encoding** path is both the most common and the right fit for synthetic profiles.
- The homogenization warning is the strongest argument for **few, well-separated personas** over many fine-grained ones.
