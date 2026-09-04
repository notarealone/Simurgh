# WPO: Enhancing RLHF with Weighted Preference Optimization

- **Authors:** Wenxuan Zhou, Ravi Agrawal, Shujian Zhang, Sathish Reddy Indurthi, Sanqiang Zhao, Kaiqiang Song, Silei Xu, Chenguang Zhu
- **Year:** 2024
- **Venue:** EMNLP 2024 (arXiv:2406.11827)
- **Link:** https://aclanthology.org/2024.emnlp-main.475/

---

## Summary

WPO targets the distribution gap created when preference pairs come from a model other than the policy being trained. It weights each pair by its probability under the current policy so that offline training better resembles on-policy training.

## Key Contributions

- Defines a policy-dependent weight for off-policy preference pairs.
- Adds no online generation or separate reward model.
- Reports gains over DPO on the evaluated instruction-following benchmarks.

## Relevance to This Thesis

The thesis's Grok-generated pairs are off-policy for Qwen3-4B. WPO is therefore a direct test of whether reweighting reduces that mismatch.

## Notes

A smaller WPO loss does not by itself indicate a better policy because the policy-dependent weight also changes the loss scale.
