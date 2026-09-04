# Simurgh

Optimizing Retrieval-Augmented Generation (RAG) systems with Reinforcement Learning for personalization. Undergraduate thesis project by Alireza Hosseini.

## Constraints (read before proposing approaches)

A fixed context shapes every decision here — weigh each suggestion against it:

- **Timeline:** ~4 weeks, solo. Favor simple, proven methods over novel-but-untested ones; a working baseline beats an ambitious pipeline that doesn't run.
- **Compute:** minimal (Kaggle + limited university GPUs). Design for small models (≤7B), PEFT/LoRA adapters, and **offline RL (DPO) over online RL (PPO)**. API LLMs are fine as judges, not as the trainable policy.
- **Data:** no real student data — user profiles and preference pairs are **synthetic** (LLM-as-simulator).
- **Subject:** personalized RAG for **Persian** educational text. Persian NLP quirks (ZWNJ, Arabic vs. Persian ye/ke, digit forms) are first-class concerns, not afterthoughts.

## Project Structure

```
Simurgh/
├── docs/                    # All documentation (proposal, lit review, methodology)
│   ├── thesis-proposal.md   # ← start here
│   ├── literature-review.md
│   ├── methodology.md
│   ├── experiment-design.md
│   └── results/             # Benchmark results & analysis
├── src/
│   ├── rag/                 # RAG pipeline (retriever + generator)
│   ├── rl/                  # RL training loop, reward models, policies
│   ├── personalization/     # User modeling, preference tracking
│   └── data/                # Data loading, preprocessing
├── configs/                 # YAML/JSON experiment configs
├── benchmarks/              # Evaluation scripts & metrics
├── notebooks/               # Jupyter notebooks (exploration + demos)
├── samples/                 # Source documents to ingest (e.g. Persian corpus)
├── app/                     # Web demo (framework TBD)
├── pyproject.toml           # Project config (UV, ruff)
└── CLAUDE.md                # ← you are here
```

## Tooling

- **Package manager:** [UV](https://docs.astral.sh/uv/) — `uv add <package>` to add dependencies
- **Linter / Formatter:** [Ruff](https://docs.astral.sh/ruff/) — configured in `pyproject.toml`
- **Python:** 3.11 (pinned in `.python-version`)
- **Lint check:** `uv run ruff check .`
- **Format:** `uv run ruff format .`

## Getting Started

```bash
# Install UV if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install project
uv sync

# Install dev dependencies
uv sync --extra dev

# Set up local secrets — API key and LLM endpoint
cp .env.example .env

# Verify setup
uv run ruff check .
```

## How to work on this project

This is a thesis, not a product. The goal is a *defensible research claim*, so process matters as much as code.

### Research discipline

- **Reproducible by construction.** Every experiment is a YAML config in `configs/`; set and log a random seed; pin model and dataset versions. An experiment must be re-runnable from its config alone — no magic numbers in code, no uncommitted "I changed it in the notebook."
- **The baseline ladder is sacred.** Always report against it: naive RAG → persona prompting → DPO rewriter → full system. A gain only counts *relative to the rung below*.
- **Ablate what you add.** Every component (RL, personalization, query rewriting) needs a run with it removed, so gains trace to a specific cause, not the system overall.
- **Evaluate honestly.** Primary metric = persona alignment / pedagogical quality (LLM-judge + a human-validated sample); EM/F1 and retrieval metrics are secondary. Report variance over ≥3 seeds and sanity-check significance before claiming an improvement. Hold out a true test set and never tune on it.
- **Guard the judge.** The judge that *scores* results must not be the judge/prompt that *generated* preference pairs. Spot-check judge scores against human ratings, and watch for reward hacking (a rewriter that games the judge instead of helping the student).
- **Report the negatives.** Null and negative results stay in the thesis. State limitations; don't cherry-pick the one seed or metric that looks good.
- **Ground in the literature.** Extend one base paper. Log every paper read in `docs/references/` (one file per paper from `TEMPLATE.md`, registered in `INDEX.md`) and record open decisions + rationale in `docs/things-to-consider.md`. Cross-link docs with **relative-path Markdown links** (e.g. `[methodology](methodology.md)`, or `[two-tales](references/two-tales-….md)` across folders) — these render and click through on GitHub and VS Code. Do **not** use `[[wikilinks]]`; they only resolve in Obsidian-class editors, which this repo isn't set up as.

### Domain conventions (RAG / RL / personalization)

- **No leakage.** The rule is that no question or passage may cross splits. The frozen assets currently use a question-level split (seed 42, 70/15/15, pooled across all seven sources): `question_ref` is disjoint, but source files overlap across splits. The splitter does not enforce shared-passage (`group_id`) isolation. Any re-split must be a separate, explicitly versioned change because DPO and ROPG assets are frozen against these files.
- **Evaluate retrieval before generation.** Measure Recall@K / MRR per persona on the retriever in isolation; an end-to-end gain on top of broken retrieval means nothing. Log retrieval hits so failures trace to retrieval vs. generation.
- **Version the index.** Chunking, embedding model, and top-K are config, not constants; rebuild and version the index whenever embeddings change.
- **Profile is the contract.** Define the user-profile schema once (grade level, learning style, goal) and thread it through rewriting → retrieval → generation. Every personalized component must be switchable back to "no profile."
- **Preference pairs are persona-conditioned.** "Better" means better *for that learner*. Keep a frozen reference model for DPO; prefer small LoRA adapters; watch for policy degeneration (length/repetition hacking).
- **Reward faithfulness, not fluency.** The judge rubric scores persona-fit + pedagogical quality + faithfulness to retrieved context; a fluent answer that isn't grounded must score low.

### Code & repo

- Source in `src/` imports as `rag` / `rl` / `personalization` / `data`. Add deps with `uv add`; run ruff (`uv run --extra dev ruff check . && uv run --extra dev ruff format .`) before committing.
- **English everywhere in code, comments, and docs.** Persian belongs only in corpus data (`samples/`) and model output; the example queries in notebooks and the `fa` prompt variant are the deliberate exceptions.
- **Phases are configs, not folders.** Each baseline-ladder rung is a config over the shared `src/` modules (`configs/phase0_naive.yaml`, then `phase1_*`, …) — never `src/phase-N/` directories.
- **Secrets and the endpoint live in the environment.** `OPENAI_API_KEY` and `OPENAI_BASE_URL` come from `.env`, read only through `src/data/settings.py`; don't touch `os.environ` elsewhere or hard-code a URL. Experiment parameters stay in `configs/*.yaml`.
- Never commit: model weights, large datasets, the built index (`data/index/`, `*.db`), `.env`, notebook outputs, `dist/`. Notebooks are for exploration; production code goes in `src/`.
- **Ignore `supervisor_additions/` entirely** unless the user explicitly mentions it. This directory contains scripts and data provided by supervisor Ali Edalat as project scaffolding — treat it as a sealed external resource, never modify or reference it without a direct user instruction.
