# Teaching According to Students' Aptitude: Personalized Mathematics Tutoring via Persona-, Memory-, and Forgetting-Aware LLMs

- **Authors:** Yang Wu, Rujing Yao, Tong Zhang, Yufei Shi, Zhuoren Jiang, Zhushan Li, Xiaozhong Liu
- **Year:** 2025
- **Venue:** AAAI 2026 Workshop (arXiv:2511.15163)
- **Link:** https://arxiv.org/abs/2511.15163

---

## Summary

TASA personalizes mathematics tutoring by conditioning an LLM tutor on a structured, per-student
**persona** (a proficiency profile), an **event memory** of recent problem-solving episodes, and
an explicit **forgetting** model that decays mastery over time. What matters for Simurgh: the persona
is *not* a rigid numeric schema — each entry is a **natural-language description** ("excels at basic
arithmetic but struggles with multi-step word problems") paired with **concept keywords** from a
skill taxonomy. Personas and memories are mined from real student interaction logs by LLM-based
agent generators, retrieved by semantic similarity, and *forgetting-aware-rewritten* before being
inserted into the tutoring prompt.

## Key Contributions

- **Persona-, memory-, and forgetting-aware tutoring framework** that adapts the tutor to a learner's evolving aptitude rather than treating all students identically.
- **LLM agent generators** (a Persona Generator and a Memory Generator) that excavate learning patterns from interaction logs zero-shot — no hand-labeled persona schema.
- **Persona representation = natural-language text + concept keywords**, retrieved by similarity and rewritten to reflect temporal decay before generation.
- Empirical gains on personalized math tutoring over persona-agnostic baselines.

## Relevance to This Thesis

TASA is the **closest precedent for a structured learner persona feeding a generation pipeline**,
and it directly validates Simurgh's representation decision: store the profile canonically, but
**render it to a natural-language description for the prompt** — TASA shows prose-plus-keywords
outperforms a rigid numeric schema for steering an LLM. Its concept-keyword tagging is an analogue
of Simurgh's grounded-vs-skill / personalization-headroom question tags ([experiment-design](../experiment-design.md)).
Where Simurgh diverges, by design: TASA *derives* personas from **real interaction logs** and tracks
**multi-turn memory + forgetting**, whereas Simurgh uses **synthetic** profiles, is **single-turn**,
and carries **no history**. So TASA informs the *persona representation*, not the memory machinery.

## Notes

- Memory and forgetting dynamics are deliberately **out of Simurgh's scope** (4-week timeline, single-turn QA); state this explicitly so the omission reads as a scoping choice, not an oversight.
- The transferable ideas are (1) prose persona representation and (2) **LLM-as-persona-extractor** — though Simurgh inverts the direction: it *authors* synthetic personas rather than mining them from logs (cf. the *Scaling Synthetic Data Creation with 1,000,000,000 Personas* line of persona-driven synthetic-data generation).
- TASA personalizes the *generator*; Simurgh's trained component is the *rewriter*. The persona schema is shared, but the injection point differs — relevant to the persona-aware-vs-blind generator axis in [methodology](../methodology.md).
