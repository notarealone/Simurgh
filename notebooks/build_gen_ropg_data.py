"""Regenerate notebooks/gen_ropg_data.ipynb from the production generator.

The data-generation notebook runs without a repository checkout on Kaggle and
Colab. Its generation logic therefore has to be inlined, but maintaining a
second hand-written implementation caused the notebook and
src/data/gen_ropg_data.py to drift. This builder keeps the production module as
the source of truth and only inlines its four local dependencies.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "gen_ropg_data.ipynb"


def _module_body(path: Path) -> str:
    """Return module code without its docstring or future import."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines(keepends=True)
    first_line = 0
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        first_line = tree.body[0].end_lineno or 0
    body = "".join(lines[first_line:]).lstrip("\n")
    return body.replace("from __future__ import annotations\n\n", "", 1)


module = (ROOT / "src" / "data" / "gen_ropg_data.py").read_text(encoding="utf-8")
profiles = _module_body(ROOT / "src" / "personalization" / "profiles.py")
embedder = _module_body(ROOT / "src" / "rag" / "embedder.py")
llm = _module_body(ROOT / "src" / "rag" / "llm.py")
config = (ROOT / "configs" / "datagen_ropg.yaml").read_text(encoding="utf-8")

worker = module.replace(
    "from data.settings import OPENAI_API_KEY, OPENAI_BASE_URL",
    "import os\n\nOPENAI_API_KEY = os.environ.get(\"OPENAI_API_KEY\", \"local\")\n"
    "OPENAI_BASE_URL = os.environ.get(\"OPENAI_BASE_URL\")",
).replace(
    "from personalization.profiles import render_profile, train_personas",
    "# --- inlined from src/personalization/profiles.py ---\n"
    + profiles.rstrip()
    + "\n# --- end inlined profiles ---",
).replace(
    "from rag.embedder import Qwen3Embedder",
    "# --- inlined from src/rag/embedder.py ---\n"
    + embedder.rstrip()
    + "\n# --- end inlined embedder ---",
).replace(
    "from rag.llm import OpenAICompatClient",
    "# --- inlined from src/rag/llm.py ---\n"
    + llm.rstrip()
    + "\n# --- end inlined LLM client ---",
)

for local_import in (
    "data.settings",
    "personalization.profiles",
    "rag.embedder",
    "rag.llm",
):
    assert local_import not in worker, f"Local import not inlined: {local_import}"
assert "max_completion_tokens=judge_cfg" not in worker, (
    "OpenAICompatClient accepts max_tokens; the generator call has drifted"
)
compile(worker, "gen_ropg_data_worker.py", "exec")


def _markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": hashlib.sha1(f"markdown\0{text}".encode()).hexdigest()[:16],
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": hashlib.sha1(f"code\0{text}".encode()).hexdigest()[:16],
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    _markdown(
        """# ROPG-KD data generation

Generates version-4 `data/ropg_kd/{train,val}.jsonl` scored groups and their derived
triplet/pair files. The worker cell is generated from `src/data/gen_ropg_data.py` by
`notebooks/build_gen_ropg_data.py`; do not hand-edit its generation logic.

## Kaggle setup

1. Enable one GPU and internet access.
2. Attach `simurgh-data`, containing `chunks/corpus.jsonl`, `questions/`, and `splits/`.
3. Add the `OPENAI_API_KEY` and `OPENAI_BASE_URL` secrets.
4. Run all cells through generation. A rerun skips successful version-4 groups and retries
   groups that exhausted judge retries.

Old stem-only, version-2, and version-3 scored files and their derived files are incompatible.
`--derive-only` cannot repair them. Use an empty output directory, and never combine old and
version-4 rows.
"""
    ),
    _code("!pip install -q sentence-transformers==5.6.0 openai pyyaml numpy tqdm"),
    _markdown("## Runtime configuration\n"),
    _code(
        f'''import os
from pathlib import Path

import yaml

RUNTIME = "kaggle"  # "kaggle" | "colab" | "local"
GDRIVE_BASE = "/content/drive/MyDrive/simurgh-data"

if RUNTIME == "kaggle":
    from kaggle_secrets import UserSecretsClient

    secrets = UserSecretsClient()
    os.environ["OPENAI_API_KEY"] = secrets.get_secret("OPENAI_API_KEY")
    base_url = secrets.get_secret("OPENAI_BASE_URL")
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    data_root = Path("/kaggle/input/datasets/alirezahsn/simurgh-data")
    output_dir = Path("/kaggle/working/data/ropg_kd")
    work_dir = Path("/kaggle/working")
elif RUNTIME == "colab":
    from google.colab import drive, userdata

    drive.mount("/content/drive")
    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
    base_url = userdata.get("OPENAI_BASE_URL")
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    data_root = Path(GDRIVE_BASE)
    output_dir = data_root / "ropg_kd"
    work_dir = Path("/content")
else:
    data_root = Path("data")
    output_dir = data_root / "ropg_kd"
    work_dir = Path(".")

work_dir.mkdir(parents=True, exist_ok=True)
os.chdir(work_dir)

cfg = yaml.safe_load({config!r})
if RUNTIME in {{"kaggle", "colab"}}:
    cfg["embedder"].update({{"device": "cuda", "batch_size": 8, "fp16": True}})
cfg["data"].update(
    {{
        "chunks": str(data_root / "chunks" / "corpus.jsonl"),
        "questions_dir": str(data_root / "questions"),
        "splits_dir": str(data_root / "splits"),
        "output_dir": str(output_dir),
    }}
)

config_path = (work_dir / "datagen_ropg_runtime.yaml").resolve()
config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
worker_path = (work_dir / "gen_ropg_data_worker.py").resolve()
print(f"Input:  {{data_root}}")
print(f"Output: {{output_dir}}")
print(f"Config: {{config_path}}")
'''
    ),
    _markdown("## Canonical worker\n"),
    _code(f"%%writefile gen_ropg_data_worker.py\n{worker}"),
    _markdown("## Generate and resume\n"),
    _code(
        '''get_ipython().run_line_magic(
    "run", f"{worker_path} --config {config_path}"
)
'''
    ),
    _markdown(
        """## Re-derive only

Run this cell by itself after changing only `cfg["triplets"]` and rewriting
`config_path`. It validates that scored rows use the complete-query format before replacing
triplet or pair files.
"""
    ),
    _code(
        '''config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
get_ipython().run_line_magic(
    "run", f"{worker_path} --config {config_path} --derive-only"
)
'''
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
print(f"Wrote {NOTEBOOK_PATH}")
