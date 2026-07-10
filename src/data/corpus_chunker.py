from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

_PERSIAN_TO_ARABIC = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

_PERSIAN_ORDINAL: dict[str, int] = {
    "یکم": 1,
    "اول": 1,
    "دوم": 2,
    "سوم": 3,
    "چهارم": 4,
    "پنجم": 5,
    "ششم": 6,
    "هفتم": 7,
    "هشتم": 8,
    "نهم": 9,
    "دهم": 10,
    "یازدهم": 11,
    "دوازدهم": 12,
    "سیزدهم": 13,
    "چهاردهم": 14,
    "پانزدهم": 15,
    "شانزدهم": 16,
    "هفدهم": 17,
}

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
_BOLD_LINE_RE = re.compile(r"^\*\*")


def _strip_html(text: str) -> str:
    return _HTML_TAG_RE.sub("", text)


def _extract_lesson_id(heading_text: str) -> str | None:
    m = re.search(r"درس\s+([۰۱۲۳۴۵۶۷۸۹]+)", heading_text)
    if m:
        num = int(m.group(1).translate(_PERSIAN_TO_ARABIC))
        return f"lesson-{num:02d}"
    m = re.search(r"درس\s+(\S+)", heading_text)
    if m:
        word = m.group(1).rstrip(":،")
        if word in _PERSIAN_ORDINAL:
            return f"lesson-{_PERSIAN_ORDINAL[word]:02d}"
    return None


def _parse_blocks(text: str) -> list[tuple[str, str, str]]:
    """Return list of (level, heading_text, body) for every heading block."""
    parts = _HEADING_RE.split(text)
    blocks: list[tuple[str, str, str]] = []
    i = 1
    while i + 1 < len(parts):
        level_hashes = parts[i]
        heading_text = parts[i + 1].strip()
        body = parts[i + 2] if i + 2 < len(parts) else ""
        blocks.append((level_hashes, heading_text, body))
        i += 3
    return blocks


def _clean_body(body: str) -> str:
    body = body.strip()
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body


def _parse_verse_entries(body: str) -> list[tuple[list[str], str]]:
    """Split a '### معنی ابیات' block into (bold_verse_lines, explanation) pairs."""
    lines = body.split("\n")
    entries: list[tuple[list[str], str]] = []
    current_bold: list[str] = []
    current_expl: list[str] = []

    for line in lines:
        if _BOLD_LINE_RE.match(line):
            if current_expl and any(ln.strip() for ln in current_expl):
                entries.append((current_bold, "\n".join(current_expl).strip()))
                current_bold = []
                current_expl = []
            current_bold.append(line)
        elif line.strip():
            if current_bold:
                current_expl.append(line)
        else:
            pass

    if current_bold:
        entries.append((current_bold, "\n".join(current_expl).strip()))

    return entries


def chunk_textbook(raw: str) -> list[dict]:
    blocks = _parse_blocks(raw)
    chunks: list[dict] = []
    current_lesson_id = "preamble"
    counters: dict[tuple[str, str], int] = defaultdict(int)

    for level, heading, body in blocks:
        if level == "###":
            lid = _extract_lesson_id(heading)
            if lid is not None:
                current_lesson_id = lid
                poem_text = _clean_body(body)
                if poem_text:
                    key = (current_lesson_id, "poem")
                    counters[key] += 1
                    chunks.append(
                        {
                            "chunk_id": f"textbook:{current_lesson_id}:poem:{counters[key]:03d}",
                            "source": "textbook",
                            "lesson_id": current_lesson_id,
                            "section_type": "poem",
                            "text": poem_text,
                        }
                    )
        elif level == "####":
            if heading.startswith("دانش ادبی"):
                body_text = _clean_body(body)
                if body_text:
                    key = (current_lesson_id, "literary-notes")
                    counters[key] += 1
                    chunks.append(
                        {
                            "chunk_id": f"textbook:{current_lesson_id}:literary-notes:{counters[key]:03d}",
                            "source": "textbook",
                            "lesson_id": current_lesson_id,
                            "section_type": "literary-notes",
                            "text": body_text,
                        }
                    )

    return chunks


def chunk_study_guide(raw: str) -> list[dict]:
    blocks = _parse_blocks(raw)
    chunks: list[dict] = []
    current_lesson_id = "preamble"
    counters: dict[tuple[str, str], int] = defaultdict(int)

    for level, heading, body in blocks:
        if level == "##":
            lid = _extract_lesson_id(heading)
            if lid is not None:
                current_lesson_id = lid
        elif level == "###":
            if heading.startswith("معنی لغت"):
                body_text = _clean_body(body)
                if body_text:
                    key = (current_lesson_id, "vocabulary")
                    counters[key] += 1
                    chunks.append(
                        {
                            "chunk_id": f"study-guide:{current_lesson_id}:vocabulary:{counters[key]:03d}",
                            "source": "study-guide",
                            "lesson_id": current_lesson_id,
                            "section_type": "vocabulary",
                            "text": body_text,
                        }
                    )
            elif heading.startswith("معنی ابیات"):
                entries = _parse_verse_entries(body)
                for bold_lines, explanation in entries:
                    combined = "\n".join(bold_lines)
                    if explanation:
                        combined += "\n" + explanation
                    combined = combined.strip()
                    if combined:
                        key = (current_lesson_id, "verse-explanation")
                        counters[key] += 1
                        chunks.append(
                            {
                                "chunk_id": f"study-guide:{current_lesson_id}:verse-explanation:{counters[key]:03d}",
                                "source": "study-guide",
                                "lesson_id": current_lesson_id,
                                "section_type": "verse-explanation",
                                "text": combined,
                            }
                        )

    return chunks


def chunk_gifted(raw: str) -> list[dict]:
    blocks = _parse_blocks(raw)
    chunks: list[dict] = []
    current_lesson_id = "preamble"
    counters: dict[tuple[str, str], int] = defaultdict(int)

    for level, heading, body in blocks:
        if level == "###":
            lid = _extract_lesson_id(heading)
            if lid is not None:
                current_lesson_id = lid
            body_text = _clean_body(body)
            if body_text:
                key = (current_lesson_id, "deep-analysis")
                counters[key] += 1
                chunks.append(
                    {
                        "chunk_id": f"gifted:{current_lesson_id}:deep-analysis:{counters[key]:03d}",
                        "source": "gifted",
                        "lesson_id": current_lesson_id,
                        "section_type": "deep-analysis",
                        "text": body_text,
                    }
                )

    return chunks


_FIXED_EXAM_SPLITS: dict[str, list[str]] = {
    "train": ["khordad-1402-arzeshyabi-ostani", "midterm-unknown_1", "keshvari-unknown_1"],
    "val": ["khordad1403-keshvari"],
    "test": ["khordad1404-keshvari", "khordad1404-khorasan"],
}


def generate_splits(
    questions_dir: Path,
    output_dir: Path,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    fixed_exam_splits: bool = False,
) -> None:
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    fixed_qids: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    fixed_stems: set[str] = set()
    if fixed_exam_splits:
        for split_name, stems in _FIXED_EXAM_SPLITS.items():
            for stem in stems:
                path = questions_dir / f"{stem}.json"
                if not path.exists():
                    logger.warning("Fixed exam file not found: %s", path)
                    continue
                data = json.loads(path.read_text(encoding="utf-8"))
                for q in data.get("questions", []):
                    fixed_qids[split_name].append(f"{stem}:{q['id']}")
                fixed_stems.add(stem)

    free_qids: list[str] = []
    for fpath in sorted(questions_dir.glob("*.json")):
        if fpath.stem in fixed_stems:
            continue
        data = json.loads(fpath.read_text(encoding="utf-8"))
        for q in data.get("questions", []):
            free_qids.append(f"{fpath.stem}:{q['id']}")

    rng = random.Random(seed)
    rng.shuffle(free_qids)

    n = len(free_qids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    split_qids: dict[str, list[str]] = {
        "train": fixed_qids["train"] + free_qids[:n_train],
        "val": fixed_qids["val"] + free_qids[n_train : n_train + n_val],
        "test": fixed_qids["test"] + free_qids[n_train + n_val :],
    }

    for split_name, qids in split_qids.items():
        output_path = splits_dir / f"{split_name}_qids.txt"
        output_path.write_text("\n".join(qids) + "\n", encoding="utf-8")
        logger.info("Wrote %d qids to %s", len(qids), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Structure-aware Persian corpus chunker")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw markdown files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/chunks/corpus.jsonl"),
        help="Output path for corpus.jsonl",
    )
    parser.add_argument(
        "--questions-dir",
        type=Path,
        default=Path("data/questions"),
        help="Directory containing question JSON files",
    )
    parser.add_argument(
        "--splits-only",
        action="store_true",
        help="Skip corpus chunking; only regenerate split files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for question-level shuffling (default: 42)",
    )
    parser.add_argument(
        "--ratios",
        nargs=3,
        type=float,
        default=[0.70, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test split ratios (default: 0.70 0.15 0.15)",
    )
    parser.add_argument(
        "--fixed-exam-splits",
        action="store_true",
        help="Pin real exam files to their designated splits; only randomize remaining files",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Ensure required output directories exist
    for d in ["data/splits", "data/ropg_kd", "data/dpo"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    if not args.splits_only:
        source_configs = [
            ("farsi-9th-grade-textbook.md", "textbook", chunk_textbook),
            ("farsi-9th-grade-study-guide.md", "study-guide", chunk_study_guide),
            ("farsi-9th-grade-gifted-textbook.md", "gifted", chunk_gifted),
        ]

        all_chunks: list[dict] = []

        for filename, source_name, chunk_fn in source_configs:
            path = args.raw_dir / filename
            if not path.exists():
                logger.warning("Source file not found: %s", path)
                continue
            raw = path.read_text(encoding="utf-8")
            raw = _strip_html(raw)
            chunks = chunk_fn(raw)
            logger.info("Source %s: %d chunks", source_name, len(chunks))
            all_chunks.extend(chunks)

        with open(args.output, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        logger.info("Wrote %d total chunks to %s", len(all_chunks), args.output)

    generate_splits(
        args.questions_dir,
        args.output.parent.parent,
        seed=args.seed,
        ratios=tuple(args.ratios),
        fixed_exam_splits=args.fixed_exam_splits,
    )


if __name__ == "__main__":
    main()
