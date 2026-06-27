# Scaling Synthetic Data Creation with 1,000,000,000 Personas

- **Authors:** Tao Ge, Xin Chan, Xiaoyang Wang, Dian Yu, Haitao Mi, Dong Yu (Tencent AI Lab)
- **Year:** 2024
- **Venue:** arXiv:2406.20094 [cs.CL]
- **Link:** https://arxiv.org/abs/2406.20094

---

## Summary

Introduces **Persona Hub** and a **persona-driven** methodology for synthetic data creation: by
conditioning a generator on a diverse persona, the *same* prompt yields varied, persona-flavored
outputs, so a large collection of personas becomes a lever for scaling and diversifying synthetic
data. Personas are curated automatically via **Text-to-Persona** (infer a persona from a text) and
**Persona-to-Persona** (derive related personas), and data is synthesized with zero-shot, few-shot,
or **persona-enhanced few-shot** prompting.

## Key Contributions

- **Persona-driven synthesis**: a persona acts as a distributional "view" that steers an LLM to produce diverse data from one task prompt.
- **Two scalable persona-curation methods** (Text-to-Persona, Persona-to-Persona) yielding ~1B personas.
- **Three synthesis prompting modes** (zero-shot / few-shot / persona-enhanced few-shot), demonstrated on math problems, instructions, knowledge text, and more.

## Relevance to This Thesis

This is Simurgh's **method citation for LLM-driven, persona-conditioned synthetic data** — which is
exactly what Simurgh's DPO preference pairs are (persona-conditioned synthetic rewrites + answers,
labeled by a simulator). The **persona-enhanced few-shot** mode is the template for generating
persona-conditioned questions/answers from the train split ([methodology](../methodology.md), Training Procedure).
Where Simurgh differs sharply: Persona Hub's value is **scale and diversity** (a billion sampled
personas), whereas Simurgh needs **four hand-placed, well-separated** personas with a holdout — so
the transferable part is the *conditioning technique and prompting modes*, **not** the curation
pipeline. The paper's implicit warning — sampled personas can collapse toward generic outputs — is
the same homogenization risk flagged in [simulating-students-with-large-language-models-a-review-of-architecture-mechanisms-and-role-modelling-in-education-with-generative-ai](simulating-students-with-large-language-models-a-review-of-architecture-mechanisms-and-role-modelling-in-education-with-generative-ai.md).

## Notes

- The Text-to-Persona / Persona-to-Persona machinery is overkill for 4 curated personas; if Simurgh later needs *eval-coverage* breadth, persona-multiplication (cf. data-scarcity mitigations in [experiment-design](../experiment-design.md)) could borrow it.
- Persona-conditioned generation here is in English; Simurgh's synthesis must stay Persian and pass ZWNJ / ye-ke normalization so generated terms match the index.
