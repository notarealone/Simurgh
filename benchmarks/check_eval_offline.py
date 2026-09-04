"""Exercise the evaluation rewrite's pure-Python contracts without model services.

The full evaluator needs GPU model weights and three remote endpoints, but its parsing, retry,
resume/merge, and reporting rules are deterministic. This standalone check keeps those rules
executable on a development machine with no network, GPU, or adapter files, using only small
fabricated records that retain the production record schema.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
for _p in (str(SRC_DIR), str(BENCHMARKS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ruff: noqa: E402  (deliberate mid-file imports; sys.path must be set first)
from eval_runner import (
    ARM_NAMES,
    COMPARISONS,
    PERSONAS,
    Case,
    build_expected_keys,
    collect_records,
    resolve_records,
    write_paired_deltas,
    write_summary,
)
from judge import (
    JudgeError,
    RetryPolicy,
    _score_with_retry,
    parse_judge_scores,
)

_RECORD_KEYS = {
    "replicate",
    "arm",
    "question_ref",
    "persona_id",
    "status",
    "failure_stage",
    "rank",
    "config_sha256",
    "embedder_model",
    "embedder_adapter_path",
    "rewriter_model",
    "rewriter_adapter_path",
    "generator_model",
    "primary_judge_model",
    "secondary_judge_model",
    "retrieval_instruction",
    "original_query",
    "rewritten_query",
    "hits",
    "answer",
    "primary_scores",
    "secondary_scores",
    "primary_attempts",
    "secondary_attempts",
    "primary_error",
    "secondary_error",
    "timestamp",
}


def _scores(persona_alignment: int) -> dict[str, int]:
    return {
        "context_utility": 3,
        "answer_correctness": 1,
        "faithfulness": 3,
        "persona_alignment": persona_alignment,
        "pedagogical_quality": 3,
    }


def _record(
    *,
    replicate: int = 1,
    arm: str = "base_embedder",
    question_ref: str = "exam:q1",
    persona_id: str = "crammer",
    status: str = "ok",
    failure_stage: str | None = None,
    rank: int = 0,
    config_sha256: str = "current",
    persona_alignment: int = 3,
    primary_scores: dict[str, int] | None = None,
    secondary_scores: dict[str, int] | None = None,
    answer: str | None = "answer",
) -> dict[str, object]:
    if status == "ok":
        if primary_scores is None:
            primary_scores = _scores(persona_alignment)
        if secondary_scores is None:
            secondary_scores = _scores(persona_alignment)
    else:
        answer = None
        primary_scores = None
        secondary_scores = None

    record: dict[str, object] = {
        "replicate": replicate,
        "arm": arm,
        "question_ref": question_ref,
        "persona_id": persona_id,
        "status": status,
        "failure_stage": failure_stage,
        "rank": rank,
        "config_sha256": config_sha256,
        "embedder_model": "Qwen/Qwen3-Embedding-0.6B",
        "embedder_adapter_path": None,
        "rewriter_model": None,
        "rewriter_adapter_path": None,
        "generator_model": "deepseek-v4-flash",
        "primary_judge_model": "gpt-5.6-luna",
        "secondary_judge_model": "gemini-3.7-flash",
        "retrieval_instruction": "profile",
        "original_query": "question",
        "rewritten_query": None,
        "hits": [{"chunk_id": "chunk-1", "text": "passage", "score": 1.0, "rank": 1}],
        "answer": answer,
        "primary_scores": primary_scores,
        "secondary_scores": secondary_scores,
        "primary_attempts": 1 if status == "ok" else 0,
        "secondary_attempts": 1 if status == "ok" else 0,
        "primary_error": None,
        "secondary_error": None,
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    assert set(record) == _RECORD_KEYS
    return record


def _write_rank_file(directory: Path, rank: int, records: list[dict[str, object]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rank{rank}.jsonl"
    content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text(content, encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _find_row(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = [
        row for row in rows if all(row.get(key) == value for key, value in criteria.items())
    ]
    assert len(matches) == 1, f"expected one row for {criteria}, got {len(matches)}"
    return matches[0]


def _expect_value_error(raw: str) -> None:
    try:
        parse_judge_scores(raw)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {raw!r}")


def _expect_runtime_error(action, description: str) -> None:
    try:
        action()
    except RuntimeError:
        return
    raise AssertionError(f"expected RuntimeError for {description}")


def check_score_parsing() -> None:
    valid = {
        "context_utility": 4,
        "answer_correctness": 1,
        "faithfulness": 3,
        "persona_alignment": 2,
        "pedagogical_quality": 4,
    }
    raw = json.dumps(valid)
    assert parse_judge_scores(raw).as_dict() == valid

    fenced = f"```json\n{raw}\n```"
    _expect_value_error(fenced)

    missing = dict(valid)
    missing.pop("faithfulness")
    _expect_value_error(json.dumps(missing))

    extra = dict(valid)
    extra["extra"] = 0
    _expect_value_error(json.dumps(extra))

    bool_answer = dict(valid)
    bool_answer["answer_correctness"] = True
    _expect_value_error(json.dumps(bool_answer))

    float_faithfulness = dict(valid)
    float_faithfulness["faithfulness"] = 3.0
    _expect_value_error(json.dumps(float_faithfulness))

    high_persona_alignment = dict(valid)
    high_persona_alignment["persona_alignment"] = 5
    _expect_value_error(json.dumps(high_persona_alignment))

    high_answer_correctness = dict(valid)
    high_answer_correctness["answer_correctness"] = 2
    _expect_value_error(json.dumps(high_answer_correctness))

    _expect_value_error("")
    print("OK judge score parsing")


def check_retry_behavior() -> None:
    policy = RetryPolicy(
        max_attempts=3,
        initial_backoff_seconds=0.0,
        backoff_multiplier=1.0,
        max_backoff_seconds=0.0,
    )
    valid = json.dumps(
        {
            "context_utility": 3,
            "answer_correctness": 1,
            "faithfulness": 3,
            "persona_alignment": 3,
            "pedagogical_quality": 3,
        }
    )
    flaky_calls = 0

    def flaky_call() -> str:
        nonlocal flaky_calls
        flaky_calls += 1
        if flaky_calls < 3:
            raise RuntimeError("temporary failure")
        return valid

    scores, attempts = _score_with_retry(flaky_call, policy, "offline")
    assert attempts == 3
    assert flaky_calls == 3
    assert scores.persona_alignment == 3

    always_policy = RetryPolicy(
        max_attempts=4,
        initial_backoff_seconds=0.0,
        backoff_multiplier=1.0,
        max_backoff_seconds=0.0,
    )
    always_calls = 0

    def always_fails() -> str:
        nonlocal always_calls
        always_calls += 1
        raise RuntimeError("permanent failure")

    result = None
    try:
        result = _score_with_retry(always_fails, always_policy, "offline")
    except JudgeError as exc:
        assert "4 attempts" in str(exc)
    else:
        raise AssertionError("always-failing callable unexpectedly returned scores")
    assert result is None
    assert always_calls == always_policy.max_attempts
    print("OK judge retry behavior")


def check_merge_semantics() -> None:
    expected_key = (1, "base_embedder", "exam:q1", "crammer")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        resolved_dir = root / "resolved"
        failure = _record(
            rank=0,
            status="failed",
            failure_stage="generation",
        )
        success = _record(rank=1, answer="recovered answer")
        _write_rank_file(resolved_dir, 0, [failure])
        _write_rank_file(resolved_dir, 1, [success])
        records, stale = collect_records(resolved_dir, world_size=2, digest="current")
        assert stale == 0
        resolved = resolve_records(records, [expected_key])
        assert len(resolved) == 1
        assert resolved[0]["status"] == "ok"
        assert resolved[0]["answer"] == "recovered answer"

        duplicate_dir = root / "duplicate"
        _write_rank_file(duplicate_dir, 0, [_record(rank=0)])
        _write_rank_file(duplicate_dir, 1, [_record(rank=1)])
        duplicate_records, stale = collect_records(duplicate_dir, world_size=2, digest="current")
        assert stale == 0
        _expect_runtime_error(
            lambda: resolve_records(duplicate_records, [expected_key]), "duplicate successes"
        )

        missing_dir = root / "missing"
        _write_rank_file(missing_dir, 0, [])
        missing_records, stale = collect_records(missing_dir, world_size=1, digest="current")
        assert stale == 0
        _expect_runtime_error(
            lambda: resolve_records(missing_records, [expected_key]), "missing expected key"
        )

        stale_dir = root / "stale"
        _write_rank_file(stale_dir, 0, [_record(config_sha256="foreign")])
        current_records, stale = collect_records(stale_dir, world_size=1, digest="current")
        assert current_records == []
        assert stale == 1
    print("OK merge and resume semantics")


def _report_config() -> dict[str, object]:
    return {
        "personas": ["crammer", "newcomer"],
        "replicates": [1],
        "execution": {"bootstrap_samples": 128, "bootstrap_seed": 42},
    }


def _report_records(
    base_values: tuple[int, int], trained_values: tuple[int, int]
) -> list[dict[str, object]]:
    pairs = (("exam:q1", "crammer"), ("exam:q2", "newcomer"))
    records: list[dict[str, object]] = []
    for (question_ref, persona_id), base_alignment, trained_alignment in zip(
        pairs, base_values, trained_values, strict=True
    ):
        records.append(
            _record(
                arm="base_embedder",
                question_ref=question_ref,
                persona_id=persona_id,
                persona_alignment=base_alignment,
            )
        )
        records.append(
            _record(
                arm="trained_embedder",
                question_ref=question_ref,
                persona_id=persona_id,
                persona_alignment=trained_alignment,
            )
        )
    return records


def check_reporting() -> None:
    train_split = PERSONAS["crammer"].split
    test_split = PERSONAS["newcomer"].split
    assert train_split != test_split
    config = _report_config()
    comparison = next(
        comparison
        for comparison in COMPARISONS
        if comparison[1] == "base_embedder" and comparison[2] == "trained_embedder"
    )[0]

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        records = _report_records((2, 4), (3, 4))
        summary_path = root / "summary.csv"
        paired_path = root / "paired_deltas.csv"
        write_summary(summary_path, records, config)
        write_paired_deltas(paired_path, records, config)

        summary_rows = _read_csv(summary_path)
        base_all = _find_row(
            summary_rows,
            arm="base_embedder",
            persona="all",
            replicate="all",
            judge="primary",
            metric="persona_alignment",
        )
        trained_all = _find_row(
            summary_rows,
            arm="trained_embedder",
            persona="all",
            replicate="all",
            judge="primary",
            metric="persona_alignment",
        )
        assert float(base_all["mean"]) == 3.0
        assert float(trained_all["mean"]) == 3.5

        for persona_label, expected_split in (
            ("all_train", train_split),
            ("all_test", test_split),
        ):
            split_row = _find_row(
                summary_rows,
                arm="base_embedder",
                persona=persona_label,
                persona_split=expected_split,
                replicate="all",
                judge="primary",
                metric="persona_alignment",
            )
            assert split_row["persona_split"] == expected_split

        paired_rows = _read_csv(paired_path)
        pooled = _find_row(
            paired_rows,
            comparison=comparison,
            persona="all",
            judge="primary",
            metric="persona_alignment",
        )
        assert float(pooled["delta"]) == 0.5
        assert int(pooled["n"]) == 2

        sign_records = _report_records((2, 4), (4, 2))
        sign_path = root / "paired_sign.csv"
        write_paired_deltas(sign_path, sign_records, config)
        sign_rows = _read_csv(sign_path)
        positive = _find_row(
            sign_rows,
            comparison=comparison,
            persona="crammer",
            judge="primary",
            metric="persona_alignment",
        )
        negative = _find_row(
            sign_rows,
            comparison=comparison,
            persona="newcomer",
            judge="primary",
            metric="persona_alignment",
        )
        pooled_sign = _find_row(
            sign_rows,
            comparison=comparison,
            persona="all",
            judge="primary",
            metric="persona_alignment",
        )
        positive_delta = float(positive["delta"])
        negative_delta = float(negative["delta"])
        pooled_delta = float(pooled_sign["delta"])
        assert positive_delta > 0
        assert negative_delta < 0
        assert (
            min(positive_delta, negative_delta)
            < pooled_delta
            < max(positive_delta, negative_delta)
        )
    print("OK summary and paired reporting")


def check_expected_keys() -> None:
    personas = ("crammer", "newcomer")
    cases = [
        Case(
            question_ref=f"exam:q{question_number}",
            persona_id=persona_id,
            query="question",
            gold_answer="answer",
            gold_explanation="explanation",
            profile_rendered="profile",
        )
        for question_number in range(1, 4)
        for persona_id in personas
    ]
    config = {
        "replicates": [1, 2],
        "arms": [{"name": arm_name} for arm_name in ARM_NAMES],
    }
    keys = build_expected_keys(config, cases)
    assert len(keys) == 60
    assert len(set(keys)) == 60
    print("OK expected-key arithmetic")


def main() -> int:
    check_score_parsing()
    check_retry_behavior()
    check_merge_semantics()
    check_reporting()
    check_expected_keys()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
