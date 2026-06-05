# Methodology

> Detailed description of the proposed approach.

---

## System Architecture

<!-- High-level architecture diagram / description of the RAG + RL pipeline. -->

- [ ] Draw architecture diagram showing RAG pipeline + RL optimization loop
- [ ] Describe data flow from user query → personalized response

## RAG Pipeline

<!-- Retriever, generator, and how they compose. -->

- [ ] Specify retriever model and index structure
- [ ] Specify generator model
- [ ] Describe chunking, embedding, and retrieval strategy

## RL Formulation

<!-- State space, action space, reward function, policy, training algorithm. -->

- [ ] Define state space (what the agent observes)
- [ ] Define action space (what the agent can change)
- [ ] Define reward function (how pedagogical quality is measured)
- [ ] Choose training algorithm (PPO, DPO, REINFORCE)

## Personalization Module

<!-- How user context/preferences are modeled and injected. -->

- [ ] Define user profile schema (grade level, learning style, etc.)
- [ ] Describe how profile influences query rewriting and retrieval
- [ ] Decide: profile-conditioned, persona embeddings, or per-user policy

## Training Procedure

<!-- Step-by-step training workflow. -->

- [ ] Write step-by-step training loop (data → preference pairs → DPO → RL)
- [ ] Specify training hyperparameters and hardware requirements
- [ ] Define stopping criteria and evaluation checkpoints

## Inference

<!-- How the trained system serves personalized responses. -->

- [ ] Describe inference pipeline: user query + profile → personalized response
- [ ] Specify latency and deployment considerations
