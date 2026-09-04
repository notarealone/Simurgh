# Training Language Models to Follow Instructions with Human Feedback

- **Authors:** Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe
- **Year:** 2022
- **Venue:** NeurIPS 2022 (arXiv:2203.02155)
- **Link:** https://arxiv.org/abs/2203.02155

---

## Summary

InstructGPT establishes the standard three-stage RLHF pipeline: supervised fine-tuning, reward-model training from ranked outputs, and policy optimization with PPO. It demonstrates that human preferences can steer a language model, but the policy stage requires online sampling and several models in memory.

## Key Contributions

- Trains a reward model from human rankings of candidate responses.
- Optimizes the language-model policy with PPO while penalizing divergence from a reference policy.
- Shows that a 1.3B InstructGPT model can be preferred to the 175B GPT-3 model on the evaluated prompt distribution.

## Relevance to This Thesis

This paper provides the RLHF and PPO baseline that DPO replaces. Its multi-stage, online training cost explains why this compute-limited project uses offline preference optimization instead.

## Notes

The paper studies broad instruction following with human labels. It does not test synthetic, persona-conditioned Persian query rewriting.
