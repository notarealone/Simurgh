# From Problem-Solving to Teaching Problem-Solving: Aligning LLMs with Pedagogy using Reinforcement Learning

- **Authors:** David Dinucu-Jianu, Jakub Macina, Nico Daheim, Ido Hakimi, Iryna Gurevych, Mrinmaya Sachan
- **Year:** 2025
- **Venue:** EMNLP 2025 (Main, oral) — arXiv:2505.15607
- **Link:** https://arxiv.org/abs/2505.15607

---

## Summary

LLMs are trained to *solve* problems and therefore tend to hand students the answer, which conflicts with good pedagogy — a tutor should withhold the solution and guide the learner to it. This paper presents an **online RL framework that turns a base LLM into a tutor using simulated student–tutor interactions, with no human annotations**. A 7B model trained this way matches much larger proprietary tutors (e.g. LearnLM) on teaching quality while preserving its reasoning ability better than single-turn supervised fine-tuning. The reward uses **controllable weighting between pedagogical support and student-solving accuracy**, exposing the trade-off (Pareto frontier) between "teach" and "solve"; optional thinking tags surface the model's instructional planning.

## Key Contributions

- **Online RL alignment for tutoring**: adapts an LLM into an effective tutor from *simulated* student–tutor dialogues, with no human-annotated tutoring data and no distillation from a larger teacher model.
- **Controllable multi-objective reward**: an explicit, tunable weighting between pedagogical guidance and answer accuracy, tracing the teaching-vs-solving Pareto frontier instead of collapsing to one behavior.
- **Small model, big result**: a 7B tutor reaches LearnLM-level teaching quality while retaining reasoning ability better than single-turn SFT baselines.
- **Interpretability**: optional thinking tags expose the tutor's instructional plan.

## Relevance to This Thesis

This paper validates Simurgh's exact recipe in the adjacent task of *generation*: small model (7B) + RL + **synthetic learner simulation** + a **pedagogical-quality reward** + no real student data — every one of Simurgh's hard constraints, shown to work. Concretely: (1) its **simulated student–tutor loop** is a template for Simurgh's LLM-as-simulator generating persona-conditioned preference pairs / rewards; (2) its **controllable teach-vs-solve reward weighting** is a concrete design for Simurgh's judge rubric, which must score *pedagogical fit*, not just answer correctness — the paper shows how to keep accuracy from being sacrificed; (3) it demonstrates **pedagogical quality as a trainable reward**, the central premise of the whole project. Where Simurgh differs: this work tunes the *generator/tutor* in English, whereas Simurgh personalizes *retrieval and query rewriting* for Persian text — so it is the strongest evidence for the *reward and simulation* design, and a natural extension target if a generation-side rung is added later.

## Notes

- The teach-vs-solve Pareto framing is a direct answer to a reward-hacking risk Simurgh shares: optimize teaching too hard and the tutor stops being correct/faithful. Mirror this with a faithfulness term in the judge rubric (cf. "Reward faithfulness, not fluency" in [[CLAUDE]] / [[methodology]]).
- Uses *online* RL with a live student simulator; Simurgh's default is offline DPO. The transferable parts are the **reward decomposition and the simulator design**, not necessarily the online loop — DPO over judge-scored persona pairs is the cheaper path to the same signal.
- English-only; pedagogical norms and the student simulator would need Persian-language grounding (and a Persian-fluent simulator/judge) to reuse directly.
- Complements the retrieval-side papers: [[optimization-methods-for-personalizing-large-language-models-through-retrieval-augmentation]] / [[s3-you-dont-need-that-much-data-to-train-a-search-agent-via-rl]] cover *what to retrieve for a learner*; this covers *how to teach once retrieved* — together they frame the full personalized pipeline.
