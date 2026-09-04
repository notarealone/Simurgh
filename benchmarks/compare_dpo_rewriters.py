"""Generate and blindly compare DPO-family query rewriters.

This benchmark evaluates the rewriter alone. It never loads a retriever, corpus, index, or
answer generator. Model outputs are cached before the independent Gemini-family judge runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.questions import load_question, render_question_value  # noqa: E402
from personalization.profiles import render_profile  # noqa: E402
from rag.llm import GeminiClient, OpenAICompatClient  # noqa: E402
from rag.rewriter import DPORewriter, build_rewrite_messages  # noqa: E402
from rl.dpo_train import load_pairs, validate_splits  # noqa: E402

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = (
    "You compare two query rewrites for a Persian educational search system. Judge which "
    "rewrite would retrieve the most useful study passages for the stated learner while "
    "preserving the complete question's meaning. The gold answer and explanation are context "
    "for relevance only; penalize a rewrite that leaks the answer. Return only strict JSON: "
    '{"winner":"A","reason":"short reason"}, '
    '{"winner":"B","reason":"short reason"}, or '
    '{"winner":"TIE","reason":"short reason"}.'
)
# Structured output pins the reply shape server-side; `_parse_judgment` still strips fenced
# markdown as defence-in-depth for endpoints that return a wrapper despite the schema.
_JUDGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "winner": {"type": "STRING", "enum": ["A", "B", "TIE"]},
        "reason": {"type": "STRING"},
    },
    "required": ["winner", "reason"],
    "property_ordering": ["winner", "reason"],
}
_SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class PromptRecord:
    question_ref: str
    persona_id: str
    query: str
    profile: str
    answer: str
    explanation: str

    @property
    def key(self) -> tuple[str, str]:
        return self.question_ref, self.persona_id


@dataclass(frozen=True)
class Candidate:
    name: str
    kind: str
    run_dir: Path | None = None


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int
    initial_backoff_seconds: float
    backoff_multiplier: float
    max_backoff_seconds: float

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RetryPolicy:
        policy = cls(
            max_attempts=int(config["max_attempts"]),
            initial_backoff_seconds=float(config["initial_backoff_seconds"]),
            backoff_multiplier=float(config["backoff_multiplier"]),
            max_backoff_seconds=float(config["max_backoff_seconds"]),
        )
        if policy.max_attempts < 1:
            raise ValueError("comparison.retry.max_attempts must be at least 1")
        if policy.initial_backoff_seconds < 0 or policy.backoff_multiplier < 1:
            raise ValueError("comparison retry backoff values are invalid")
        if policy.max_backoff_seconds < policy.initial_backoff_seconds:
            raise ValueError("comparison retry maximum must not be below its initial delay")
        return policy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} line {line_number} is invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path} line {line_number} must contain an object")
        records.append(record)
    return records


def _parse_question_ref(question_ref: str) -> tuple[str, str]:
    try:
        exam_stem, question_id = question_ref.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid question_ref {question_ref!r}") from exc
    if not exam_stem or not question_id:
        raise ValueError(f"Invalid question_ref {question_ref!r}")
    return exam_stem, question_id


def load_validation_prompts(config: dict[str, Any]) -> tuple[list[PromptRecord], dict[str, str]]:
    """Load every unique validation question/persona prompt and its judge-only gold fields."""
    train_path = Path(config["data"]["train_path"])
    val_path = Path(config["data"]["val_path"])
    train_pairs = load_pairs(train_path)
    val_pairs = load_pairs(val_path)
    split_report = validate_splits(train_pairs, val_pairs)
    questions_dir = Path(config["data"]["questions_dir"])

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for pair in val_pairs:
        key = pair["question_ref"], pair["persona_id"]
        previous = grouped.get(key)
        if previous is not None and previous["query"] != pair["query"]:
            raise ValueError(f"Validation key {key} has inconsistent query text")
        grouped[key] = pair

    prompts = []
    for question_ref, persona_id in sorted(grouped):
        pair = grouped[(question_ref, persona_id)]
        exam_stem, question_id = _parse_question_ref(question_ref)
        question = load_question(exam_stem, question_id, questions_dir)
        if question.query != pair["query"]:
            raise ValueError(
                f"Pair query for {question_ref} differs from data/questions rendered context"
            )
        answer = "" if question.answer is None else render_question_value(question.answer)
        explanation = question.explanation or ""
        prompts.append(
            PromptRecord(
                question_ref=question_ref,
                persona_id=persona_id,
                query=pair["query"],
                profile=render_profile(persona_id),
                answer=answer,
                explanation=explanation,
            )
        )
    expected = split_report["validation"]["question_persona_keys"]
    if len(prompts) != expected:
        raise RuntimeError(f"Loaded {len(prompts)} prompts, split report expected {expected}")
    if not prompts:
        raise ValueError(f"{val_path} yielded no validation prompts")
    return prompts, {"train": _sha256(train_path), "validation": _sha256(val_path)}


def _parse_assignment(raw: str, option: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"{option} must use NAME=VALUE, got {raw!r}")
    name, value = raw.split("=", 1)
    if not _SAFE_NAME.fullmatch(name) or not value:
        raise ValueError(f"Invalid {option} assignment: {raw!r}")
    return name, value


def resolve_candidates(run_args: list[str], include_base: bool, include_grok: bool) -> list[Candidate]:
    candidates = []
    names = set()
    for raw in run_args:
        name, value = _parse_assignment(raw, "--run")
        if name in {"base_qwen", "grok"}:
            raise ValueError(f"Candidate name {name!r} is reserved for its fixed baseline")
        if name in names:
            raise ValueError(f"Duplicate candidate name: {name}")
        run_dir = Path(value)
        if not (run_dir / "dpo_best").is_dir():
            raise FileNotFoundError(f"Candidate {name} has no promoted adapter: {run_dir / 'dpo_best'}")
        candidates.append(Candidate(name=name, kind="trained", run_dir=run_dir))
        names.add(name)
    if include_base:
        candidates.append(Candidate(name="base_qwen", kind="base"))
        names.add("base_qwen")
    if include_grok:
        candidates.append(Candidate(name="grok", kind="grok"))
        names.add("grok")
    if len(candidates) < 2:
        raise ValueError("Comparison requires at least two candidates")
    return candidates


def _run_manifest(candidate: Candidate) -> dict[str, Any] | None:
    if candidate.run_dir is None:
        return None
    manifest_path = candidate.run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing run manifest for {candidate.name}: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _candidate_metadata(
    candidate: Candidate,
    config: dict[str, Any],
    data_hashes: dict[str, str],
) -> dict[str, Any]:
    generation = config["generation"]
    metadata: dict[str, Any] = {
        "candidate": candidate.name,
        "kind": candidate.kind,
        "data_sha256": data_hashes,
        "prompt_contract": "build_rewrite_messages+qwen_chat_template+enable_thinking_false",
    }
    if candidate.kind in {"trained", "base"}:
        metadata["model"] = config["model"]["name"]
        metadata["generation"] = {
            "max_new_tokens": int(generation["max_new_tokens"]),
            "do_sample": False,
        }
    if candidate.kind == "trained":
        manifest = _run_manifest(candidate)
        if manifest is None:
            raise RuntimeError(f"Missing run manifest for {candidate.name}")
        if manifest.get("data_sha256") != data_hashes:
            raise ValueError(f"Candidate {candidate.name} was trained on different pair-file hashes")
        metadata.update(
            {
                "arm": manifest.get("arm"),
                "seed": manifest.get("seed"),
                "adapter_path": str(candidate.run_dir / "dpo_best"),
            }
        )
    elif candidate.kind == "grok":
        metadata["model"] = generation["grok_model"]
        metadata["generation"] = {
            "temperature": float(generation["grok_temperature"]),
            "max_new_tokens": int(generation["max_new_tokens"]),
        }
    return metadata


def _cache_paths(output_dir: Path, candidate: Candidate) -> tuple[Path, Path]:
    cache_dir = output_dir / "outputs"
    return cache_dir / f"{candidate.name}.jsonl", cache_dir / f"{candidate.name}.meta.json"


def validate_output_cache(
    output_path: Path,
    metadata_path: Path,
    prompts: list[PromptRecord],
    expected_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    if not output_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing output cache or metadata: {output_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Cache metadata mismatch for {output_path.name} field {key!r}: "
                f"expected {expected!r}, got {metadata.get(key)!r}"
            )
    if metadata.get("output_sha256") != _sha256(output_path):
        raise ValueError(f"Output hash mismatch for {output_path}")

    records = _read_jsonl(output_path)
    expected = {prompt.key: prompt for prompt in prompts}
    actual: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = record.get("question_ref"), record.get("persona_id")
        if key in actual:
            raise ValueError(f"Duplicate output key {key} in {output_path}")
        if key not in expected:
            raise ValueError(f"Unexpected output key {key} in {output_path}")
        prompt = expected[key]
        if record.get("query") != prompt.query or record.get("profile") != prompt.profile:
            raise ValueError(f"Prompt mismatch for {key} in {output_path}")
        rewrite = record.get("rewrite")
        if not isinstance(rewrite, str) or not rewrite.strip():
            raise ValueError(f"Empty rewrite for {key} in {output_path}")
        actual[key] = record
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise ValueError(f"Output cache {output_path} is missing {len(missing)} prompts")
    return [actual[prompt.key] for prompt in prompts]


def _local_rewrites(
    candidate: Candidate,
    config: dict[str, Any],
    prompts: list[PromptRecord],
) -> list[str]:
    generation_config = config["generation"]
    rewriter = DPORewriter(
        model_name=config["model"]["name"],
        adapter_path=str(candidate.run_dir / "dpo_best") if candidate.run_dir else None,
        device="cuda",
        max_seq_length=int(config["model"]["max_seq_length"]),
        generation={
            "max_new_tokens": int(generation_config["max_new_tokens"]),
            "do_sample": False,
        },
    )
    batch_size = int(generation_config["batch_size"])
    rewrites = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        rewrites.extend(
            rewriter.rewrite_batch(
                [prompt.profile for prompt in batch],
                [prompt.query for prompt in batch],
            )
        )
    del rewriter
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
    return rewrites


def _grok_rewrites(config: dict[str, Any], prompts: list[PromptRecord]) -> list[str]:
    from data.settings import REWRITER_API_KEY, REWRITER_BASE_URL

    if not REWRITER_BASE_URL or not REWRITER_API_KEY:
        raise RuntimeError("Grok generation requires REWRITER_BASE_URL and REWRITER_API_KEY")
    generation = config["generation"]
    client = OpenAICompatClient(
        base_url=REWRITER_BASE_URL,
        api_key=REWRITER_API_KEY,
        model=generation["grok_model"],
        temperature=float(generation["grok_temperature"]),
        max_tokens=int(generation["max_new_tokens"]),
    )
    rewrites = []
    for index, prompt in enumerate(prompts, 1):
        rewrite = client.chat(build_rewrite_messages(prompt.profile, prompt.query)).strip()
        if not rewrite:
            raise RuntimeError(f"Grok returned an empty rewrite for {prompt.key}")
        rewrites.append(rewrite)
        logger.info("Grok rewrite %d/%d", index, len(prompts))
    return rewrites


def generate_or_load_outputs(
    candidate: Candidate,
    config: dict[str, Any],
    prompts: list[PromptRecord],
    data_hashes: dict[str, str],
    output_dir: Path,
    *,
    force: bool,
) -> list[dict[str, Any]]:
    output_path, metadata_path = _cache_paths(output_dir, candidate)
    expected_metadata = _candidate_metadata(candidate, config, data_hashes)
    if output_path.is_file() and metadata_path.is_file() and not force:
        return validate_output_cache(output_path, metadata_path, prompts, expected_metadata)

    previous_hash = _sha256(output_path) if output_path.is_file() else None
    if candidate.kind in {"trained", "base"}:
        rewrites = _local_rewrites(candidate, config, prompts)
    else:
        rewrites = _grok_rewrites(config, prompts)
    records = [
        {
            "candidate": candidate.name,
            "question_ref": prompt.question_ref,
            "persona_id": prompt.persona_id,
            "query": prompt.query,
            "profile": prompt.profile,
            "rewrite": rewrite,
        }
        for prompt, rewrite in zip(prompts, rewrites, strict=True)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, records)
    output_hash = _sha256(output_path)
    if previous_hash and candidate.kind in {"trained", "base"} and previous_hash != output_hash:
        raise RuntimeError(
            f"Deterministic Qwen output changed for {candidate.name}: "
            f"{previous_hash} != {output_hash}"
        )
    _write_json(metadata_path, {**expected_metadata, "rows": len(records), "output_sha256": output_hash})
    return validate_output_cache(output_path, metadata_path, prompts, expected_metadata)


def _parse_pairs(raw_pairs: list[str], candidate_names: list[str]) -> list[tuple[str, str]]:
    if not raw_pairs:
        return list(itertools.combinations(candidate_names, 2))
    known = set(candidate_names)
    pairs = []
    for raw in raw_pairs:
        if ":" not in raw:
            raise ValueError(f"--pair must use LEFT:RIGHT, got {raw!r}")
        left, right = raw.split(":", 1)
        if left == right or left not in known or right not in known:
            raise ValueError(f"Invalid candidate pair: {raw!r}")
        pair = tuple(sorted((left, right)))
        if pair in pairs:
            raise ValueError(f"Duplicate candidate pair: {pair}")
        pairs.append(pair)
    return pairs


def _job_key(prompt: PromptRecord, left: str, right: str, seed: int) -> str:
    raw = f"{prompt.question_ref}\0{prompt.persona_id}\0{left}\0{right}\0{seed}"
    return hashlib.sha256(raw.encode()).hexdigest()


def build_job_manifest(
    prompts: list[PromptRecord],
    model_pairs: list[tuple[str, str]],
    comparison_seed: int,
) -> list[dict[str, Any]]:
    """Build exact 50/50 hidden orientation per model pair using stable hash order."""
    jobs = []
    half = len(prompts) // 2
    for left, right in model_pairs:
        ranked = sorted(
            prompts,
            key=lambda prompt: _job_key(prompt, left, right, comparison_seed),
        )
        left_as_a = {prompt.key for prompt in ranked[:half]}
        for prompt in prompts:
            candidate_a, candidate_b = (
                (left, right) if prompt.key in left_as_a else (right, left)
            )
            jobs.append(
                {
                    "job_key": _job_key(prompt, left, right, comparison_seed),
                    "question_ref": prompt.question_ref,
                    "persona_id": prompt.persona_id,
                    "model_left": left,
                    "model_right": right,
                    "candidate_a": candidate_a,
                    "candidate_b": candidate_b,
                    "comparison_seed": comparison_seed,
                }
            )
    keys = [job["job_key"] for job in jobs]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Comparison job manifest contains duplicate keys")
    return jobs


def _validate_orientation(jobs: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for job in jobs:
        pair = job["model_left"], job["model_right"]
        counts[pair][job["candidate_a"]] += 1
    report = {f"{left}:{right}": dict(counter) for (left, right), counter in counts.items()}
    for pair, counter in counts.items():
        values = list(counter.values())
        if len(values) != 2 or abs(values[0] - values[1]) > 1:
            raise RuntimeError(f"Unbalanced A/B orientation for {pair}: {dict(counter)}")
    return report


def _judge_prompt(prompt: PromptRecord, rewrite_a: str, rewrite_b: str) -> str:
    gold_sections = []
    if prompt.answer:
        gold_sections.append(f"Gold answer/reference:\n{prompt.answer}")
    if prompt.explanation:
        gold_sections.append(f"Gold explanation/rubric:\n{prompt.explanation}")
    gold = "\n\n".join(gold_sections) or "No gold explanation supplied."
    return (
        f"Learner profile:\n{prompt.profile}\n\n"
        f"Original complete question:\n{prompt.query}\n\n"
        f"Judge-only gold context:\n{gold}\n\n"
        f"Rewrite A:\n{rewrite_a}\n\n"
        f"Rewrite B:\n{rewrite_b}"
    )


def _parse_judgment(raw: str) -> tuple[str, str]:
    stripped = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    result = json.loads(stripped)
    if not isinstance(result, dict) or set(result) != {"winner", "reason"}:
        raise ValueError("Judge response must contain exactly winner and reason")
    winner = result["winner"]
    reason = result["reason"]
    if winner not in {"A", "B", "TIE"}:
        raise ValueError(f"Invalid judge winner: {winner!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Judge reason must be a nonempty string")
    return winner, reason.strip()


class JudgeUnavailable(RuntimeError):
    """The judge endpoint rejected a request in a way no retry can fix."""


def _judge_error_is_transient(exc: Exception) -> bool:
    """Decide whether *exc* is worth another attempt.

    google-genai raises ``ClientError``/``ServerError`` carrying an integer ``code``.
    Rate limits, timeouts and 5xx are transient; every other 4xx (wrong route, wrong
    model name, bad key, revoked quota) will fail identically on every retry and for
    every remaining job, so it must abort the run instead of burning the budget.
    """
    if isinstance(exc, ValueError):
        # A malformed or truncated reply. Cheap to ask once more.
        return True
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code in {408, 429} or code >= 500
    # Connection resets and read timeouts arrive without a status code.
    return True


def _judge_one(
    job: dict[str, Any],
    prompts_by_key: dict[tuple[str, str], PromptRecord],
    outputs: dict[str, dict[tuple[str, str], str]],
    client: GeminiClient,
    retry: RetryPolicy,
) -> dict[str, Any]:
    key = job["question_ref"], job["persona_id"]
    prompt = _judge_prompt(
        prompts_by_key[key],
        outputs[job["candidate_a"]][key],
        outputs[job["candidate_b"]][key],
    )
    delay = retry.initial_backoff_seconds
    last_error = ""
    for attempt in range(1, retry.max_attempts + 1):
        try:
            winner, reason = _parse_judgment(client.generate(prompt))
            return {**job, "status": "ok", "winner": winner, "reason": reason, "attempts": attempt}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if not _judge_error_is_transient(exc):
                raise JudgeUnavailable(
                    f"Judge endpoint failed permanently on attempt {attempt}: {last_error}"
                ) from exc
            if attempt < retry.max_attempts:
                time.sleep(delay)
                delay = min(delay * retry.backoff_multiplier, retry.max_backoff_seconds)
    return {
        **job,
        "status": "missing",
        "winner": None,
        "reason": None,
        "attempts": retry.max_attempts,
        "error": last_error,
    }


def run_judge(
    config: dict[str, Any],
    jobs: list[dict[str, Any]],
    prompts: list[PromptRecord],
    cached_records: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    base_url = os.environ.get("DPO_EVAL_BASE_URL")
    api_key = os.environ.get("DPO_EVAL_API_KEY")
    model = os.environ.get("DPO_EVAL_MODEL")
    missing_env = [
        name
        for name, value in (
            ("DPO_EVAL_BASE_URL", base_url),
            ("DPO_EVAL_API_KEY", api_key),
            ("DPO_EVAL_MODEL", model),
        )
        if not value
    ]
    if missing_env:
        raise RuntimeError("Missing required comparison environment: " + ", ".join(missing_env))
    if "gemini" not in model.lower():
        raise ValueError("DPO_EVAL_MODEL must identify the agreed Gemini-family judge")

    results_path = output_dir / "judgments.jsonl"
    existing = _read_jsonl(results_path) if results_path.is_file() else []
    existing_by_key = {record["job_key"]: record for record in existing}
    expected_keys = {job["job_key"] for job in jobs}
    unexpected = set(existing_by_key) - expected_keys
    if unexpected:
        raise ValueError(f"Judgment cache contains {len(unexpected)} jobs outside this manifest")

    prompts_by_key = {prompt.key: prompt for prompt in prompts}
    outputs = {
        name: {
            (record["question_ref"], record["persona_id"]): record["rewrite"]
            for record in records
        }
        for name, records in cached_records.items()
    }
    retry = RetryPolicy.from_config(config["comparison"]["retry"])
    client = GeminiClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_instruction=_JUDGE_SYSTEM,
        temperature=0.0,
        max_output_tokens=int(config["comparison"]["judge_max_tokens"]),
        # An absent key means the documented default; an explicit empty value hands the
        # choice to the model, which matters because the level a model accepts depends on
        # the model, and DPO_EVAL_MODEL is a runtime value.
        thinking_level=config["comparison"].get("judge_thinking_level", "minimal"),
        response_schema=_JUDGE_SCHEMA,
    )
    pending = [job for job in jobs if job["job_key"] not in existing_by_key]
    max_workers = int(config["comparison"]["max_workers"])
    output_dir.mkdir(parents=True, exist_ok=True)
    fatal: JudgeUnavailable | None = None
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_judge_one, job, prompts_by_key, outputs, client, retry): job
            for job in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except JudgeUnavailable as exc:
                # Every remaining job would fail the same way. Drop the queue instead of
                # spending the budget, and leave no half-empty judgment cache behind.
                fatal = exc
                executor.shutdown(wait=False, cancel_futures=True)
                break
            existing_by_key[result["job_key"]] = result
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            logger.info("Judgment %d/%d status=%s", index, len(pending), result["status"])
    if fatal is not None:
        raise fatal
    ordered = [existing_by_key[job["job_key"]] for job in jobs]
    _write_jsonl(results_path, ordered)
    return ordered


def _result_score(result: dict[str, Any], candidate: str) -> float:
    if result["winner"] == "TIE":
        return 0.5
    winner = result["candidate_a"] if result["winner"] == "A" else result["candidate_b"]
    return 1.0 if winner == candidate else 0.0


def _bootstrap_interval(values: list[float], samples: int, seed: int) -> list[float]:
    if not values:
        return [math.nan, math.nan]
    rng = random.Random(seed)
    means = []
    count = len(values)
    for _ in range(samples):
        means.append(sum(values[rng.randrange(count)] for _ in range(count)) / count)
    means.sort()
    return [means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]]


def _sign_flip_p_value(values: list[float]) -> float:
    wins = sum(value > 0.5 for value in values)
    losses = sum(value < 0.5 for value in values)
    non_ties = wins + losses
    if not non_ties:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(non_ties, k) for k in range(tail + 1)) / (2**non_ties)
    return min(1.0, 2.0 * probability)


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, p_values[key] * (total - rank)))
        adjusted[key] = running
    return adjusted


def summarize_results(
    config: dict[str, Any],
    candidates: list[Candidate],
    results: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    complete = [result for result in results if result["status"] == "ok"]
    missing = len(results) - len(complete)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in complete:
        grouped[(result["model_left"], result["model_right"])].append(result)

    bootstrap_samples = int(config["comparison"]["bootstrap_samples"])
    comparison_seed = int(results[0]["comparison_seed"]) if results else 0
    head_to_head = []
    raw_p_values = {}
    candidate_points: dict[str, list[float]] = defaultdict(list)
    persona_points: dict[tuple[str, str], list[float]] = defaultdict(list)
    for left, right in sorted(grouped):
        pair_results = grouped[(left, right)]
        left_scores = [_result_score(result, left) for result in pair_results]
        counts = Counter("tie" if score == 0.5 else "win" if score == 1.0 else "loss" for score in left_scores)
        pair_key = f"{left}:{right}"
        raw_p_values[pair_key] = _sign_flip_p_value(left_scores)
        head_to_head.append(
            {
                "left": left,
                "right": right,
                "n": len(left_scores),
                "missing": sum(
                    result["status"] != "ok"
                    for result in results
                    if (result["model_left"], result["model_right"]) == (left, right)
                ),
                "left_wins": counts["win"],
                "ties": counts["tie"],
                "left_losses": counts["loss"],
                "left_score": sum(left_scores) / len(left_scores),
                "left_score_ci95": _bootstrap_interval(
                    left_scores,
                    bootstrap_samples,
                    comparison_seed + int(hashlib.sha256(pair_key.encode()).hexdigest()[:8], 16),
                ),
                "sign_flip_p": raw_p_values[pair_key],
            }
        )
        for result, left_score in zip(pair_results, left_scores, strict=True):
            candidate_points[left].append(left_score)
            candidate_points[right].append(1.0 - left_score)
            persona_points[(left, result["persona_id"])].append(left_score)
            persona_points[(right, result["persona_id"])].append(1.0 - left_score)

    adjusted = _holm_adjust(raw_p_values)
    for row in head_to_head:
        row["holm_p"] = adjusted[f"{row['left']}:{row['right']}"]
    round_robin = {
        candidate.name: (
            sum(candidate_points[candidate.name]) / len(candidate_points[candidate.name])
            if candidate_points[candidate.name]
            else math.nan
        )
        for candidate in candidates
    }
    per_persona = [
        {
            "candidate": candidate,
            "persona_id": persona,
            "score": sum(values) / len(values),
            "n": len(values),
        }
        for (candidate, persona), values in sorted(persona_points.items())
    ]

    selected_variant = None
    if "wpo" in round_robin and "robust_dpo" in round_robin:
        wpo_score = round_robin["wpo"]
        robust_score = round_robin["robust_dpo"]
        if wpo_score != robust_score:
            selected_variant = "wpo" if wpo_score > robust_score else "robust_dpo"
        else:
            direct = next(
                row
                for row in head_to_head
                if {row["left"], row["right"]} == {"wpo", "robust_dpo"}
            )
            wpo_direct = direct["left_score"] if direct["left"] == "wpo" else 1 - direct["left_score"]
            selected_variant = "wpo" if wpo_direct >= 0.5 else "robust_dpo"

    summary = {
        "jobs": len(results),
        "complete": len(complete),
        "missing": missing,
        "round_robin": round_robin,
        "head_to_head": head_to_head,
        "per_persona": per_persona,
        "selected_variant": selected_variant,
        "selection_rule": (
            "round-robin score, then direct head-to-head, then WPO for the known off-policy gap"
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    with (output_dir / "head_to_head.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "left",
            "right",
            "n",
            "missing",
            "left_wins",
            "ties",
            "left_losses",
            "left_score",
            "left_score_ci95",
            "sign_flip_p",
            "holm_p",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(head_to_head)
    with (output_dir / "per_persona.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate", "persona_id", "score", "n"])
        writer.writeheader()
        writer.writerows(per_persona)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare promoted DPO-family rewriters")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME=RUN_DIR",
        help="Candidate run directory containing dpo_best and run_manifest.json",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="LEFT:RIGHT",
        help="Judge only this model pair; repeat as needed. Defaults to all pairs.",
    )
    parser.add_argument("--comparison-seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--without-base", action="store_true")
    parser.add_argument("--without-grok", action="store_true")
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def _validate_manifest_reuse(
    manifest_path: Path,
    current: dict[str, Any],
    *,
    require_judge_identity: bool,
) -> dict[str, Any] | None:
    if not manifest_path.is_file():
        return None
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in (
        "comparison_seed",
        "data_sha256",
        "candidates",
        "model_pairs",
        "prompts",
        "pairs",
        "jobs",
        "orientation",
        "output_hashes",
    ):
        if existing.get(key) != current.get(key):
            raise ValueError(f"Existing judge manifest has mismatched {key!r}")
    if require_judge_identity:
        for key in ("judge_base_url", "judge_model"):
            previous = existing.get(key)
            if previous is not None and previous != current.get(key):
                raise ValueError(f"Existing judgments use a different {key!r}: {previous!r}")
    return existing


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    if args.generate_only and args.prepare_only:
        raise ValueError("--generate-only and --prepare-only are mutually exclusive")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    comparison_seed = (
        int(config["comparison"]["screening_seed"])
        if args.comparison_seed is None
        else args.comparison_seed
    )
    output_dir = args.output_dir or (
        Path(config["comparison"]["output_root"]) / f"seed-{comparison_seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts, data_hashes = load_validation_prompts(config)
    candidates = resolve_candidates(args.run, not args.without_base, not args.without_grok)

    cached_records = {}
    for candidate in candidates:
        if args.prepare_only:
            output_path, metadata_path = _cache_paths(output_dir, candidate)
            cached_records[candidate.name] = validate_output_cache(
                output_path,
                metadata_path,
                prompts,
                _candidate_metadata(candidate, config, data_hashes),
            )
        else:
            cached_records[candidate.name] = generate_or_load_outputs(
                candidate,
                config,
                prompts,
                data_hashes,
                output_dir,
                force=args.force_regenerate,
            )
    if args.generate_only:
        logger.info("Generated or validated %d candidate caches", len(candidates))
        return

    model_pairs = _parse_pairs(args.pair, [candidate.name for candidate in candidates])
    jobs = build_job_manifest(prompts, model_pairs, comparison_seed)
    orientation = _validate_orientation(jobs)
    _write_jsonl(output_dir / "judge_jobs.jsonl", jobs)
    judge_manifest = {
        "comparison_seed": comparison_seed,
        "data_sha256": data_hashes,
        "candidates": [candidate.name for candidate in candidates],
        "model_pairs": [list(pair) for pair in model_pairs],
        "prompts": len(prompts),
        "pairs": len(model_pairs),
        "jobs": len(jobs),
        "orientation": orientation,
        "judge_base_url": os.environ.get("DPO_EVAL_BASE_URL"),
        "judge_model": os.environ.get("DPO_EVAL_MODEL"),
        "judge_api_key_present": bool(os.environ.get("DPO_EVAL_API_KEY")),
        "output_hashes": {
            candidate.name: json.loads(_cache_paths(output_dir, candidate)[1].read_text())["output_sha256"]
            for candidate in candidates
        },
    }
    manifest_path = output_dir / "judge_manifest.json"
    if not args.prepare_only:
        missing_judge_env = [
            key
            for key in ("DPO_EVAL_BASE_URL", "DPO_EVAL_API_KEY", "DPO_EVAL_MODEL")
            if not os.environ.get(key)
        ]
        if missing_judge_env:
            raise RuntimeError(
                "Missing required comparison environment: " + ", ".join(missing_judge_env)
            )
    existing_manifest = _validate_manifest_reuse(
        manifest_path,
        judge_manifest,
        require_judge_identity=not args.prepare_only,
    )
    if args.prepare_only and existing_manifest is not None:
        for key in ("judge_base_url", "judge_model", "judge_api_key_present"):
            judge_manifest[key] = existing_manifest.get(key)
    _write_json(manifest_path, judge_manifest)
    print(
        f"Prepared {len(prompts)} prompts, {len(model_pairs)} model pairs, "
        f"{len(jobs)} judge jobs in {output_dir}"
    )
    if args.prepare_only:
        return

    results = run_judge(config, jobs, prompts, cached_records, output_dir)
    summary = summarize_results(config, candidates, results, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
