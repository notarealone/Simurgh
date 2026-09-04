# QLoRA: Efficient Finetuning of Quantized LLMs

- **Authors:** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer
- **Year:** 2023
- **Venue:** NeurIPS 2023 (arXiv:2305.14314)
- **Link:** https://arxiv.org/abs/2305.14314

---

## Summary

QLoRA backpropagates through a frozen four-bit quantized base model into LoRA adapters. Its NF4 representation, double quantization, and paged optimizers reduce memory without updating the quantized base weights.

## Key Contributions

- Combines four-bit base-model storage with trainable LoRA adapters.
- Introduces NF4 and double quantization for memory-efficient fine-tuning.
- Demonstrates single-GPU fine-tuning of models up to 65B parameters in the evaluated setup.

## Relevance to This Thesis

The Qwen3-4B rewriter uses a four-bit base model with LoRA adapters. This QLoRA configuration makes DPO-family training feasible on two T4 GPUs.

## Notes

The paper's quality results do not establish that every four-bit adaptation matches full fine-tuning. That comparison remains model- and task-dependent.
