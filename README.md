# Simurgh

Optimizing Retrieval-Augmented Generation (RAG) systems with Reinforcement Learning for personalization.

Undergraduate thesis project by **Alireza Hosseini**.

---

## AI Usage Disclosure

This project uses AI tools during development.

### Coding Assistance

I use **Claude Code** (Anthropic's CLI coding agent) as a development aid:

- Generating project structure and boilerplate
- Drafting and revising documentation
- Writing, reviewing, and debugging code
- Suggesting architectural approaches

I review, understand, and approve every AI-generated suggestion before inclusion. I take full responsibility for all code, documentation, and decisions in this repository.

### Models Used

| Period | Model | Provider               | Purpose |
|---|---|------------------------|---|
| June 2026 – present | Claude Opus 4.8 / Sonnet 4.6 | Anthropic (via Claude Code) | Coding, documentation, architecture |
| June 2026 – present | GLM-5.1 | Z.ai (via Claude Code) | Coding, documentation |

Updated as models change.

### Authorship

I am the sole author of this thesis. I design the research methodology, set the direction, and make all decisions — except where I explicitly reference another author, paper, or source. AI tools accelerate implementation. They do not think, judge, or take responsibility.

---

## Project Structure

```
docs/                Thesis documentation (proposal, lit review, methodology)
src/                 Source code (rag, rl, personalization, data)
configs/             Experiment configurations
benchmarks/          Evaluation scripts
notebooks/           Jupyter notebooks for exploration
app/                 Web demo (framework TBD)
```

## Setup

Requires [UV](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync --extra dev
```

## License

MIT
