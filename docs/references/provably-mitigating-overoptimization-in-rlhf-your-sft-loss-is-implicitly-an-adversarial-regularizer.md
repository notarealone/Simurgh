# Provably Mitigating Overoptimization in RLHF: Your SFT Loss Is Implicitly an Adversarial Regularizer

- **Authors:** Zhihan Liu, Miao Lu, Shenao Zhang, Boyi Liu, Hongyi Guo, Yingxiang Yang, Jose Blanchet, Zhaoran Wang
- **Year:** 2024
- **Venue:** arXiv preprint (arXiv:2405.16436)
- **Link:** https://arxiv.org/abs/2405.16436

---

## Summary

Regularized Preference Optimization (RPO) combines a preference loss with supervised negative log-likelihood on a suitable baseline distribution. For language-model alignment, the practical objective adds an SFT loss on the preferred response to DPO.

## Key Contributions

- Frames preference overoptimization as distribution shift and reward uncertainty.
- Derives an objective that combines DPO and supervised imitation.
- Uses the supervised term to anchor the policy to preferred responses.

## Relevance to This Thesis

The thesis adds the RPO negative-log-likelihood term to each DPO-family arm. This directly counters the observed failure mode in which the preference margin grows while both preferred and rejected rewrites become less likely.

## Notes

The paper's experiments concern general language-model alignment, not Persian query rewriting. The anchor weight remains an experimental choice.
