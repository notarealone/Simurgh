# References Index

> One file per paper. Copy `TEMPLATE.md` to start a new entry.
> PDFs go in `docs/references/pdfs/` using the same title-based kebab-case naming convention.
>
> The literature review is deliberately narrowed to the two papers this thesis builds
> on directly (DPO and ROPG). The remaining entries are persona-schema references cited
> by [`personas.md`](../personas.md), not by the literature review.

| Paper | Year | Topic | Relevance | PDF |
|---|---|---|---|---|
| [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](direct-preference-optimization-your-language-model-is-secretly-a-reward-model.md) | 2023 | Preference optimization / RL for generation | Core method of the literature review: reward-model-free, offline alternative to RLHF/PPO that trains a policy directly from preference pairs | [PDF](pdfs/direct-preference-optimization-your-language-model-is-secretly-a-reward-model.pdf) |
| [Optimization Methods for Personalizing Large Language Models through Retrieval Augmentation](optimization-methods-for-personalizing-large-language-models-through-retrieval-augmentation.md) | 2024 | Personalized retriever optimization | Core method of the literature review: first to train the retriever from the downstream personalized objective (ROPG-RL / ROPG-KD, RSPG selection) | [PDF](pdfs/optimization-methods-for-personalizing-large-language-models-through-retrieval-augmentation.pdf) |
| [Two Tales of Persona in LLMs: A Survey of Role-Playing and Personalization](two-tales-of-persona-in-llms-a-survey-of-role-playing-and-personalization.md) | 2024 | Persona survey | Persona-schema reference for [`personas.md`](../personas.md): the role-playing vs personalization split | — |
| [Simulating Students with Large Language Models: A Review](simulating-students-with-large-language-models-a-review-of-architecture-mechanisms-and-role-modelling-in-education-with-generative-ai.md) | 2025 | Student-simulation review | Persona-schema reference for [`personas.md`](../personas.md): persona-axis taxonomy and validity pitfalls | — |
| [Teaching According to Students' Aptitude: Personalized Mathematics Tutoring via Persona-, Memory-, and Forgetting-Aware LLMs](teaching-according-to-students-aptitude-personalized-mathematics-tutoring-via-persona-memory-and-forgetting-aware-llms.md) | 2025 | Persona-aware tutoring | Persona-schema reference for [`personas.md`](../personas.md): structured-persona precedent, prose over numeric schema | — |
