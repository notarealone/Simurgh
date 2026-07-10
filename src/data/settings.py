"""Centralized environment access — import settings from here, don't read os.environ elsewhere.

Secrets and the deployment endpoint come from the environment (a project-root ``.env``,
loaded once on import). Experiment parameters stay in the YAML configs, not here.
"""

import os

from dotenv import load_dotenv

# Load the project's .env once. find_dotenv walks up from this file, so it works regardless
# of the current working directory (e.g. a notebook running from notebooks/). Real environment
# variables already set take precedence — load_dotenv does not override them.
load_dotenv()

# OpenAI-compatible client credentials. "local" lets keyless local servers work.
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "local")
# Endpoint; None lets the OpenAI SDK fall back to its default (the official API).
OPENAI_BASE_URL: str | None = os.environ.get("OPENAI_BASE_URL")

# Gemini judge credentials.
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_ENDPOINT: str | None = os.environ.get("GEMINI_ENDPOINT")

# Rewriter credentials (separate from judge; e.g. xAI/Grok endpoint).
REWRITER_API_KEY: str = os.environ.get("REWRITER_API_KEY", "")
REWRITER_BASE_URL: str | None = os.environ.get("REWRITER_BASE_URL")
