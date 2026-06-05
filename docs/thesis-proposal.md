### Project Information

A – Problem Definition:
Retrieval-Augmented Generation (RAG) systems are the standard approach for document-based question answering. Yet conventional RAG pipelines treat every user the same — they cannot personalize responses to a user's profile, learning needs, or comprehension level (for example, a ninth-grade student). This limitation matters most in education and Persian textbook comprehension, where one student needs step-by-step explanations while another benefits from a direct answer or a challenging question that builds critical thinking.

The core problem: personalize a RAG pipeline to respond to both textbook content and the user's persona and educational needs. This requires personalizing pipeline components — the Query Rewriter and Document Retriever — so the output matches the educational goals of each user.

B – Project Goal and Significance:
Design and implement a Reinforcement Learning (RL)-based optimization framework to personalize the RAG pipeline for Persian textbooks. The goal: improve educational effectiveness and reading comprehension for users with diverse learning profiles.

Secondary objectives:
* Use reinforcement learning to optimize the pipeline (query rewriting through retrieval) for pedagogical quality.
* Focus on Persian educational resources to build infrastructure for localized educational systems.
* Extend prior DPO-based work by adding RL optimization for the Retriever model.

C – Project Implementation Methodology:

1. Development and Improvement of the Query Rewriter
   * *Iterative DPO Training:* Use "LLM as a Judge" to generate preference pairs. Train the rewriter with iterative Direct Preference Optimization (DPO).

2. Reward Definition
   * Use the LLM-as-Judge output (e.g., a 1–10 score) as the reward signal for training an RL policy model.

3. Optimizing the Retriever Model with Personalization
   * Personalizing Embeddings
   * Using Reward-based Fine-tuning

4. Evaluation and Validation
   * Create a dataset of Persian textbook questions with diverse user profiles (beginner, advanced, challenging).
   * Compare against standard RAG baselines using Exact Match and F1.
