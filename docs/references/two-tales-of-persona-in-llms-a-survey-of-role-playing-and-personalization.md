# Two Tales of Persona in LLMs: A Survey of Role-Playing and Personalization

- **Authors:** Yu-Min Tseng, Yu-Chao Huang, Teng-Yun Hsiao, Wei-Lin Chen, Chao-Wei Huang, Yu Meng, Yun-Nung Chen
- **Year:** 2024
- **Venue:** Findings of the Association for Computational Linguistics: EMNLP 2024 (arXiv:2406.01171)
- **Link:** https://aclanthology.org/2024.findings-emnlp.969/

---

## Summary

A survey that disentangles two largely separate research lines that both use the word "persona":
**LLM role-playing** (a persona is *assigned to* the model, which then acts as that character) and
**LLM personalization** (the model *serves* a user described by a persona). It argues the two have
been conflated and maps the methods, benchmarks, and open problems of each.

## Key Contributions

- A clean conceptual split between **persona-as-character (role-play)** and **persona-as-user (personalization)**, with separate taxonomies for each.
- A consolidated map of datasets, methods, and evaluation practices across both lines.
- Identification of open problems, including evaluation rigor and the leakage between the two framings.

## Relevance to This Thesis

This survey gives Simurgh the vocabulary for its **"two hats."** Simurgh occupies *both* lines at once:
the **LLM-as-simulator/judge** does **role-playing** (it acts as a 9th-grade learner to decide which
rewrite/answer is better *for that student*), while the **rewriter + generator** do **personalization**
(they consume the profile to serve the learner). Naming this split is a methodological safeguard, not
semantics: the persona that *labels* preference pairs (role-play) must be kept distinct from the rubric
that *scores* evaluation — the same spirit as the judge-independence rule in [experiment-design](../experiment-design.md) and
[things-to-consider](../things-to-consider.md). The survey also frames why Simurgh's personalization signal (a *declared profile*)
differs from history-based personalization ([optimization-methods-for-personalizing-large-language-models-through-retrieval-augmentation](optimization-methods-for-personalizing-large-language-models-through-retrieval-augmentation.md), and Zhang et al. 2025's Personalize-Before-Retrieve).

## Notes

- Reinforces that "persona" is overloaded; the persona doc should state up front which sense is meant where (character for the simulator, user-profile for the system).
- The personalization-line taxonomy complements broader LLM-personalization surveys; this one is sharper on the role-play vs personalize boundary.
