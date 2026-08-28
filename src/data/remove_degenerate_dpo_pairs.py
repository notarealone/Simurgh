"""Remove DPO rows whose chosen and rejected rewrites are equivalent."""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATHS = (Path("data/dpo/train.jsonl"), Path("data/dpo/val.jsonl"))
EDGE_PUNCTUATION = " .،؛:!?؟\"'«»"


@dataclass(frozen=True)
class CleanedFile:
    path: Path
    retained_lines: tuple[str, ...]
    removed: tuple[tuple[int, str, str], ...]
    mode: int


def normalize_rewrite(text: str) -> str:
    """Normalize differences that cannot express a useful retrieval preference."""
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split()).strip(EDGE_PUNCTUATION)


def clean_file(path: Path) -> CleanedFile:
    retained_lines: list[str] = []
    removed: list[tuple[int, str, str]] = []

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        chosen = record.get("chosen")
        rejected = record.get("rejected")
        if not isinstance(chosen, str) or not isinstance(rejected, str):
            raise ValueError(f"{path} line {line_number} has non-string chosen/rejected fields")
        if normalize_rewrite(chosen) == normalize_rewrite(rejected):
            removed.append(
                (
                    line_number,
                    str(record.get("question_ref", "<missing question_ref>")),
                    str(record.get("persona_id", "<missing persona_id>")),
                )
            )
        else:
            retained_lines.append(line)

    return CleanedFile(
        path,
        tuple(retained_lines),
        tuple(removed),
        stat.S_IMODE(path.stat().st_mode),
    )


def write_atomically(cleaned: CleanedFile) -> None:
    cleaned.path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=cleaned.path.parent,
        prefix=f".{cleaned.path.name}.",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        for line in cleaned.retained_lines:
            temporary.write(line)
            temporary.write("\n")

    os.chmod(temporary_path, cleaned.mode)
    try:
        os.replace(temporary_path, cleaned.path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove DPO pairs equivalent after Unicode, whitespace, and edge-punctuation normalization"
    )
    parser.add_argument("paths", nargs="*", type=Path, default=list(DEFAULT_PATHS))
    parser.add_argument("--dry-run", action="store_true", help="Report removals without writing")
    args = parser.parse_args()

    cleaned_files = [clean_file(path) for path in args.paths]
    for cleaned in cleaned_files:
        for line_number, question_ref, persona_id in cleaned.removed:
            print(f"remove {cleaned.path}:{line_number} {question_ref} persona={persona_id}")
        print(
            f"{cleaned.path}: retain {len(cleaned.retained_lines)}, "
            f"remove {len(cleaned.removed)}"
        )

    if not args.dry_run:
        for cleaned in cleaned_files:
            write_atomically(cleaned)


if __name__ == "__main__":
    main()
