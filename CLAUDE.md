# Simurgh

Optimizing Retrieval-Augmented Generation (RAG) systems with Reinforcement Learning for personalization. Undergraduate thesis project by Alireza Hosseini.

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
├── notebooks/               # Jupyter notebooks for exploration
├── app/                     # Web demo (framework TBD)
├── pyproject.toml           # Project config (UV, ruff, pytest)
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

# Verify setup
uv run ruff check .
```

## Conventions

- Source code lives in `src/` (import as `rag`, `rl`, `personalization`, `data`).
- All experiments are configured via YAML files in `configs/`.
- Notebooks are for exploration only — production code goes in `src/`.
- Model weights and large data are gitignored; use Git LFS or DVC if needed.
