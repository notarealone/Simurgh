# Query Rewriting in Retrieval-Augmented Large Language Models

- **Authors:** Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, Nan Duan
- **Year:** 2023
- **Venue:** EMNLP 2023
- **Link:** https://aclanthology.org/2023.emnlp-main.322/

---

## Summary

Ma et al. insert a query-rewriting stage before retrieval and reading, producing the Rewrite-Retrieve-Read pipeline. They first prompt a large language model to rewrite the input and then train a smaller rewriter from feedback supplied by a frozen black-box reader.

## Key Contributions

- Treats the mismatch between user input and retriever-effective queries as a separate optimization problem.
- Introduces the Rewrite-Retrieve-Read decomposition for retrieval-augmented language models.
- Uses a prompted large language model as an initial rewriter and a smaller trainable model for deployment.
- Trains the smaller rewriter with reinforcement-learning feedback from the frozen reader and evaluates it on open-domain and multiple-choice question answering.

## Relevance to This Thesis

The paper motivates placing a trainable rewriter before a fixed retriever-reader pipeline. Simurgh keeps this modular role but conditions rewriting on a learner profile and replaces online reinforcement learning with offline preference optimization.

## Notes

The feedback in the paper rewards downstream task performance, not fit to a learner. Its web-search and English QA setting also differs from retrieval over a fixed Persian educational corpus.
