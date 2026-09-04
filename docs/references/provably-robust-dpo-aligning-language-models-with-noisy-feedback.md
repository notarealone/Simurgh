# Provably Robust DPO: Aligning Language Models with Noisy Feedback

- **Authors:** Sayak Ray Chowdhury, Anush Kini, Nagarajan Natarajan
- **Year:** 2024
- **Venue:** ICML 2024 (PMLR 235)
- **Link:** https://proceedings.mlr.press/v235/ray-chowdhury24a.html

---

## Summary

Robust DPO treats preference labels as subject to random flips. Its corrected loss uses an assumed noise rate to remove the average bias that ordinary DPO inherits from mislabeled pairs.

## Key Contributions

- Derives a noise-corrected DPO loss under a uniform label-flip model.
- Provides a finite-sample robustness bound for the corrected policy.
- Tests robustness on sentiment generation and helpful-harmless dialogue.

## Relevance to This Thesis

Synthetic judge labels may be wrong or ambiguous. Robust DPO tests whether modeling a fixed label-noise rate improves the query rewriter.

## Notes

The guarantee depends on a uniform, independent flip model and a noise rate below 0.5. Real judge errors may instead depend on the prompt, persona, or candidate pair.
