# LaMP: When Large Language Models Meet Personalization

- **Authors:** Alireza Salemi, Sheshera Mysore, Michael Bendersky, Hamed Zamani
- **Year:** 2024
- **Venue:** ACL 2024
- **Link:** https://aclanthology.org/2024.acl-long.399/

---

## Summary

LaMP is a benchmark for training and evaluating personalized language models. It contains seven user-conditioned tasks, comprising three text-classification tasks and four text-generation tasks, and represents each user through multiple historical input-output records.

## Key Contributions

- Introduces seven personalized NLP tasks with repeated records per user profile.
- Defines user-based and time-based evaluation settings.
- Evaluates retrieval augmentation that selects personal records through term matching, semantic matching, or recency.
- Compares non-personalized and retrieval-augmented language-model baselines in zero-shot and fine-tuned settings.

## Relevance to This Thesis

LaMP establishes retrieval augmentation as a practical route to language-model personalization and supplies the benchmark later used by ROPG. Simurgh adopts the separation between profile, retrieval, and generation, but its corpus structure differs: all synthetic learners share the same Persian educational passages instead of retrieving from a separate history owned by each user.

## Notes

The distinction between per-user profile corpora and Simurgh's shared corpus is load-bearing. In LaMP, changing the user changes the candidate documents; in Simurgh, the retriever must learn whether the ordering of the same candidate passages should change with the learner profile.