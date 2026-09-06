"""Question-side rewriting: rungs 4.1 and 5.1 of the held-out evaluation.

A copy of ``eval_runner.py`` with one variable changed, and deliberately a copy: that file
produced ``docs/results/eval-ablation-v1.md`` under a published config digest, so it is
frozen. Everything here except the arm definitions, the two rewriting stages, and the
generator prompt is that file verbatim.

``eval_runner.py`` applies the rewrite to the *retrieval* query and answers the original
question. This runner inverts that: retrieval uses the original persona-instructed query --
byte-identical to the ``trained_embedder`` rung, same adapter, same index, same instruction
-- and the rewrite becomes the question the generator is asked.

===========================================  ================================================
trained_embedder_base_rewriter_question      rung 4.1: question rewritten by an untrained Qwen3-4B
trained_embedder_rewriter_question           rung 5.1: question rewritten by the promoted adapter
===========================================  ================================================

Two consequences are structural, not incidental:

* Retrieval is identical across both arms and to rung 3, so ``context_utility`` cannot move
  except through judge noise on identical passages. That flatness is a calibration reading,
  not a finding.
* The judge still receives the *original* question and its gold answer. A rewrite that drops
  the question's content therefore shows up as a correctness or faithfulness loss, which is
  exactly the failure this pair of arms exists to detect.

The index is built from raw ``corpus.jsonl`` text, never through ``rag.dense_store``: that
module normalizes Persian text and all 171 corpus chunks change under it, while
``rl.ropg_kd.load_corpus`` trained the adapter on the raw strings. Retrieving over
normalized text would score the adapter on inputs it never saw.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
for _p in (str(SRC_DIR), str(BENCHMARKS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ruff: noqa: E402  (deliberate mid-file imports; sys.path must be set first)
from compare_runs import holm, paired_stats
from data.questions import load_question
from data.settings import (
    GEMINI_API_KEY,
    GEMINI_ENDPOINT,
    GENERATOR_API_KEY,
    GENERATOR_BASE_URL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)
from judge import SCORE_FIELDS, GeminiJudge, JudgeError, LunaJudge, RetryPolicy
from personalization.profiles import PERSONAS, render_profile

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

ARM_NAMES = (
    "trained_embedder_base_rewriter_question",
    "trained_embedder_rewriter_question",
)
JUDGE_ROLES = ("primary", "secondary")

#: The only comparison this run can make within itself: both arms share one retrieval, so
#: the difference between them is the rewriter adapter and nothing else. Comparing either
#: against rung 3 or rung 4/5 is a cross-run join against the ablation's ``results.jsonl``,
#: and belongs in the analysis, not here.
COMPARISONS = (
    (
        "trained_embedder_rewriter_question - trained_embedder_base_rewriter_question",
        "trained_embedder_base_rewriter_question",
        "trained_embedder_rewriter_question",
    ),
)

ANSWER_SYSTEM = (
    "You are a Persian-language educational assistant helping a ninth-grade student. "
    "Answer using only the numbered passages you are given. If they do not contain the "
    "answer, say so honestly instead of guessing. Always reply in Persian. Cite the "
    "passages you used by their number, like [1]."
)
ANSWER_SYSTEM_PERSONALIZED = (
    ANSWER_SYSTEM
    + " Adapt the depth, vocabulary, and style of your explanation to the learner profile "
    "you are given."
)


class RemoteCallError(RuntimeError):
    """A remote generator call exhausted its retries."""


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved corpus chunk.

    Not ``rag.store.Hit``: that type's ``source`` and ``chunk_index`` mean file-of-origin
    and within-file ordinal, so a corpus ``chunk_id`` would have to be smuggled through a
    field that means something else — and importing ``rag.store`` drags hazm and sqlite3
    into a job that needs neither.
    """

    chunk_id: str
    text: str
    score: float
    rank: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class Case:
    """One ``(question, persona)`` pair: the unit of sharding and of GPU work."""

    question_ref: str
    persona_id: str
    query: str
    gold_answer: str
    gold_explanation: str
    profile_rendered: str


@dataclass(frozen=True)
class ArmContext:
    """What one arm retrieved, the query it retrieved with, and the question it will ask.

    ``retrieval_query`` and ``answer_query`` are one string in every ``eval_runner.py``
    arm and diverge here: retrieval keeps the original question, and ``answer_query``
    carries the rewrite that reaches the generator.
    """

    retrieval_query: str
    retrieval_instruction: str
    chunks: list[RetrievedChunk]
    answer_query: str = ""


# --------------------------------------------------------------------------------------
# Config, digests, inputs
# --------------------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: config must be a mapping")
    return config


def config_digest(config: dict[str, Any]) -> str:
    """SHA-256 over the *resolved* config, which is what a record's provenance means."""
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_test_qids(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def load_corpus(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_cases(config: dict[str, Any], limit: int | None = None) -> list[Case]:
    """Ordered ``(question, persona)`` cross product — the stable sharding order.

    ``--limit`` truncates the *question* list before the cross product, so a limited run
    still covers every persona.
    """
    questions_dir = Path(config["data"]["questions_dir"])
    refs = read_test_qids(config["data"]["test_qids_path"])
    if limit is not None:
        refs = refs[:limit]
    cases: list[Case] = []
    for ref in refs:
        exam_stem, qid = ref.split(":", 1)
        context = load_question(exam_stem, qid, questions_dir)
        answer = "" if context.answer is None else str(context.answer)
        explanation = context.explanation or ""
        for persona_id in config["personas"]:
            cases.append(
                Case(
                    question_ref=ref,
                    persona_id=persona_id,
                    query=context.query,
                    gold_answer=answer,
                    gold_explanation=explanation,
                    profile_rendered=render_profile(persona_id),
                )
            )
    return cases


def build_expected_keys(
    config: dict[str, Any], cases: Iterable[Case]
) -> list[tuple[int, str, str, str]]:
    """Every ``(replicate, arm, question_ref, persona_id)`` the run must produce."""
    arm_names = [arm["name"] for arm in config["arms"]]
    keys = [
        (replicate, arm_name, case.question_ref, case.persona_id)
        for replicate in config["replicates"]
        for arm_name in arm_names
        for case in cases
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("expected keys are not unique; check personas and replicates")
    return keys


# --------------------------------------------------------------------------------------
# Validation and preflight
# --------------------------------------------------------------------------------------


def _adapter_base_model(adapter_dir: Path) -> str | None:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8")).get("base_model_name_or_path")
    except json.JSONDecodeError:
        return None


def probe_models(config: dict[str, Any]) -> dict[str, str]:
    """One minimal call per remote model literal, reporting ``ok`` or the failure.

    A misspelled model name is otherwise discovered after the GPU stages, hours in. Every
    endpoint is probed even after one fails: three preflights to learn three problems costs
    three Kaggle sessions.

    Each probe retries under its role's configured policy. The Metis Gemini route returns
    intermittent nginx 504s, and a single-shot probe turns one of those into a red
    preflight for an endpoint the run itself would have retried through. The attempt count
    is reported so a flaky-but-passing endpoint still shows up.
    """
    from rag.llm import GeminiClient, OpenAICompatClient

    def probe(call: Callable[[], str], retry_config: dict[str, Any], label: str) -> str:
        try:
            _, attempts = _call_with_retry(call, RetryPolicy.from_config(retry_config), label)
        except RemoteCallError as exc:
            return str(exc)
        return "ok" if attempts == 1 else f"ok after {attempts} attempts"

    results: dict[str, str] = {}
    generator = config["generator"]["model"]
    primary = config["judges"]["primary"]["model"]
    secondary = config["judges"]["secondary"]["model"]
    # The generator may sit behind a different vendor prefix than the OpenAI-compatible
    # judge; GENERATOR_BASE_URL falls back to OPENAI_BASE_URL when it does not. Each probe
    # carries its role's configured temperature: some models reject any other value, and a
    # probe that fails on a knob the real call never sends is a false alarm.
    for model, base_url, api_key, role in (
        (generator, GENERATOR_BASE_URL, GENERATOR_API_KEY, config["generator"]),
        (primary, OPENAI_BASE_URL, OPENAI_API_KEY, config["judges"]["primary"]),
    ):
        client = OpenAICompatClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=float(role["temperature"]),
            max_tokens=16,
        )
        results[model] = probe(
            lambda call=client.chat: call([{"role": "user", "content": "ping"}]),
            role["retry"],
            model,
        )
    secondary_config = config["judges"]["secondary"]
    gemini = GeminiClient(
        base_url=GEMINI_ENDPOINT,
        api_key=GEMINI_API_KEY,
        model=secondary,
        temperature=0.0,
        max_output_tokens=16,
        thinking_level=secondary_config.get("thinking_level"),
    )
    results[secondary] = probe(
        lambda: gemini.generate("ping"), secondary_config["retry"], secondary
    )
    return results


def validate(config: dict[str, Any], *, probe: bool = True) -> dict[str, Any]:
    """Check everything checkable before a single byte of model weight is loaded.

    Collects every failure rather than raising on the first: a preflight that reports one
    problem per run costs one Kaggle session per problem.
    """
    errors: list[str] = []
    report: dict[str, Any] = {}

    questions_dir = Path(config["data"]["questions_dir"])
    qids_path = Path(config["data"]["test_qids_path"])
    if not qids_path.is_file():
        errors.append(f"test_qids_path not found: {qids_path}")
        refs: list[str] = []
    else:
        refs = read_test_qids(qids_path)
        if len(set(refs)) != len(refs):
            errors.append("test_qids contains duplicate entries")
        for ref in refs:
            if ref.count(":") != 1 or not all(part.strip() for part in ref.split(":")):
                errors.append(f"test_qid {ref!r} is not in exam_stem:qid form")
                continue
            exam_stem, qid = ref.split(":", 1)
            try:
                load_question(exam_stem, qid, questions_dir)
            except (FileNotFoundError, KeyError) as exc:
                errors.append(f"test_qid {ref!r} does not load: {exc}")
    report["n_questions"] = len(refs)

    personas = config.get("personas")
    if not isinstance(personas, list) or not personas:
        errors.append("personas must be a nonempty list")
        personas = []
    elif len(set(personas)) != len(personas):
        errors.append("personas contains duplicates")
    unknown = [p for p in personas if p not in PERSONAS]
    if unknown:
        errors.append(f"unknown personas {unknown}; valid: {sorted(PERSONAS)}")
    report["personas"] = [{"id": p, "split": PERSONAS[p].split} for p in personas if p in PERSONAS]

    replicates = config.get("replicates")
    if not isinstance(replicates, list) or not replicates:
        errors.append("replicates must be a nonempty list")
        replicates = []
    else:
        if len(set(replicates)) != len(replicates):
            errors.append("replicates contains duplicates")
        bad = [r for r in replicates if not isinstance(r, int) or isinstance(r, bool) or r < 1]
        if bad:
            errors.append(f"replicates must be positive integers; got {bad}")
    report["replicates"] = list(replicates)

    arms = config.get("arms") or []
    names = [arm.get("name") for arm in arms]
    if names != list(ARM_NAMES):
        errors.append(f"arms must be exactly {list(ARM_NAMES)} in order; got {names}")
    for arm in arms:
        if arm.get("embedder") != "ropg":
            errors.append(f"arm {arm.get('name')!r}: embedder must be ropg in this run")
        # `none` is accepted by the ablation runner and meaningless here: an arm with no
        # rewriter would ask the original question and duplicate rung 3 at full cost.
        if arm.get("rewriter") not in ("base", "dpo"):
            errors.append(f"arm {arm.get('name')!r}: rewriter must be base or dpo")
        if arm.get("profile") is not True:
            errors.append(
                f"arm {arm.get('name')!r}: profile must be true. The no-profile baseline "
                "belongs to configs/eval_ablation.yaml; this run varies only where the "
                "rewrite is applied."
            )
    report["arms"] = names

    # Unsloth's 4-bit loader rewrites the base id it records: a LoRA trained from
    # `Qwen/Qwen3-4B` with `load_in_4bit` reports `unsloth/qwen3-4b-unsloth-bnb-4bit`.
    # Match the model family, which is what the check is actually for; an exact repo id
    # would reject the very adapter `rag.rewriter.DPORewriter` then loads.
    adapters = {
        "ropg_adapter_path": ("qwen3-embedding-0.6b", config["artifacts"]["ropg_adapter_path"]),
        "dpo_adapter_path": ("qwen3-4b", config["artifacts"]["dpo_adapter_path"]),
    }
    report["adapters"] = {}
    for label, (family, raw_path) in adapters.items():
        adapter_dir = Path(raw_path)
        base_model = _adapter_base_model(adapter_dir)
        if base_model is None:
            errors.append(f"{label}: no readable adapter_config.json under {adapter_dir}")
        elif family not in base_model.lower():
            errors.append(
                f"{label}: adapter at {adapter_dir} records base model {base_model!r}, "
                f"which is not a {family} model"
            )
        report["adapters"][label] = {"path": str(adapter_dir), "base_model": base_model}

    corpus_path = Path(config["data"]["corpus_path"])
    if not corpus_path.is_file():
        errors.append(f"corpus_path not found: {corpus_path}")
        corpus: list[dict[str, Any]] = []
    else:
        corpus = load_corpus(corpus_path)
        chunk_ids = [row.get("chunk_id") for row in corpus]
        if any(not isinstance(cid, str) or not cid for cid in chunk_ids):
            errors.append("corpus has rows with a missing or non-string chunk_id")
        elif len(set(chunk_ids)) != len(chunk_ids):
            errors.append("corpus chunk_ids are not unique")
        if any(not str(row.get("text", "")).strip() for row in corpus):
            errors.append("corpus has rows with empty text")
    report["corpus_count"] = len(corpus)

    for name, value in (
        ("OPENAI_API_KEY", OPENAI_API_KEY),
        ("OPENAI_BASE_URL", OPENAI_BASE_URL),
        ("GEMINI_API_KEY", GEMINI_API_KEY),
        ("GEMINI_ENDPOINT", GEMINI_ENDPOINT),
    ):
        if not value:
            errors.append(f"{name} is empty; set it in the environment, never in the config")

    if int(config["retrieval"]["top_k"]) < 1:
        errors.append("retrieval.top_k must be at least 1")
    if int(config["execution"]["max_workers"]) < 1:
        errors.append("execution.max_workers must be at least 1")
    for label, block in (
        ("generator.retry", config["generator"]["retry"]),
        ("judges.primary.retry", config["judges"]["primary"]["retry"]),
        ("judges.secondary.retry", config["judges"]["secondary"]["retry"]),
    ):
        try:
            RetryPolicy.from_config(block, label)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{label}: {exc}")

    output_dir = Path(config["output_dir"])
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe_path = output_dir / ".write_probe"
        probe_path.write_text("", encoding="utf-8")
        probe_path.unlink()
    except OSError as exc:
        errors.append(f"output_dir {output_dir} is not writable: {exc}")

    if errors:
        raise ValueError("preflight failed:\n  - " + "\n  - ".join(errors))

    report["expected_keys"] = (
        report["n_questions"] * len(report["personas"]) * len(arms) * len(replicates)
    )
    report["probes"] = probe_models(config) if probe else {}
    return report


def print_preflight(report: dict[str, Any]) -> None:
    print(f"Questions:        {report['n_questions']}")
    personas = ", ".join(f"{p['id']} ({p['split']})" for p in report["personas"])
    print(f"Personas:         {personas}")
    print(f"Replicates:       {report['replicates']}")
    print(f"Arms:             {', '.join(report['arms'])}")
    for label, info in report["adapters"].items():
        print(f"{label:<17} {info['path']} -> {info['base_model']}")
    print(f"Corpus chunks:    {report['corpus_count']}")
    for model, status in report["probes"].items():
        print(f"Probe {model:<20} {status}")
    print(f"Expected keys:    {report['expected_keys']}")


# --------------------------------------------------------------------------------------
# Prompting and remote calls
# --------------------------------------------------------------------------------------


def build_answer_messages(
    profile_rendered: str | None, query: str, chunks: list[RetrievedChunk]
) -> list[dict[str, str]]:
    """Build the generator turn.

    Not ``rag.prompts.build_mcq_rag_prompt``: that helper takes options as a separate
    argument and forces multiple-choice framing, while ``data.questions.load_question``
    already renders passages, options, pairs, and items into one string — and several
    held-out questions are not multiple choice.

    The *rewritten* question is shown here, which is the one thing this runner changes
    relative to ``eval_runner.py``. The passages were retrieved with the original question,
    so a rewrite that drops the question's content cannot damage retrieval -- it can only
    change what the generator is asked about passages that are already fixed.
    """
    if chunks:
        passages = "\n\n".join(f"[{chunk.rank}] {chunk.text}" for chunk in chunks)
    else:
        passages = "(no passages were retrieved)"
    system = ANSWER_SYSTEM if profile_rendered is None else ANSWER_SYSTEM_PERSONALIZED
    prefix = "" if profile_rendered is None else f"Learner profile: {profile_rendered}\n\n"
    user = f"{prefix}Passages:\n{passages}\n\nQuestion:\n{query}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _call_with_retry(call: Callable[[], str], retry: RetryPolicy, label: str) -> tuple[str, int]:
    """Retry *call* until it returns non-empty text; return ``(text, attempts)``.

    An empty completion is a failed attempt, not an answer: judged as-is it would score a
    faithful zero and quietly depress whichever arm hit the endpoint on a bad minute.
    """
    delay = retry.initial_backoff_seconds
    last_error = "no attempt was made"
    for attempt in range(1, retry.max_attempts + 1):
        try:
            text = call()
            if text and text.strip():
                return text, attempt
            last_error = "empty completion"
        except Exception as exc:  # transport failures retry like empty replies
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retry.max_attempts:
            time.sleep(delay)
            delay = min(delay * retry.backoff_multiplier, retry.max_backoff_seconds)
    raise RemoteCallError(
        f"{label} failed after {retry.max_attempts} attempts; last error: {last_error}"
    )


# --------------------------------------------------------------------------------------
# GPU stages
# --------------------------------------------------------------------------------------


def _free(*objects: Any) -> None:
    import torch

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_index(embedder: Any, texts: list[str]) -> Any:
    import faiss

    vectors = embedder.encode(texts)
    index = faiss.IndexFlatIP(embedder.dim)
    index.add(vectors)
    if index.ntotal != len(texts):
        raise RuntimeError(f"index holds {index.ntotal} vectors for {len(texts)} chunks")
    return index


def _retrieve_for_arm(
    embedder: Any,
    index: Any,
    corpus: list[dict[str, Any]],
    arm: dict[str, Any],
    cases: list[Case],
    queries: list[str],
    top_k: int,
) -> dict[tuple[str, str], ArmContext]:
    """Retrieve one arm's passages for every case.

    Queries are grouped by instruction, because ``encode_query`` applies one instruction
    per call and cannot mix them within a batch, and deduplicated within each group. The
    dedup is what makes the unprofiled arm cost one encode per question instead of one per
    persona: with no profile in the query, four personas submit identical text.
    """
    grouped: dict[str, dict[str, list[int]]] = {}
    for position, (case, query) in enumerate(zip(cases, queries, strict=True)):
        instruction = case.profile_rendered if arm["profile"] else ""
        grouped.setdefault(instruction, {}).setdefault(query, []).append(position)

    contexts: dict[tuple[str, str], ArmContext] = {}
    for instruction, by_query in grouped.items():
        unique_queries = list(by_query)
        vectors = embedder.encode_query(unique_queries, instruction=instruction)
        scores, indices = index.search(vectors, top_k)
        for row, query in enumerate(unique_queries):
            chunks = [
                RetrievedChunk(
                    chunk_id=corpus[int(corpus_index)]["chunk_id"],
                    text=corpus[int(corpus_index)]["text"],
                    score=float(score),
                    rank=rank,
                )
                for rank, (corpus_index, score) in enumerate(
                    zip(indices[row], scores[row], strict=True), start=1
                )
                if int(corpus_index) >= 0
            ]
            for position in by_query[query]:
                case = cases[position]
                contexts[case.question_ref, case.persona_id] = ArmContext(
                    retrieval_query=query,
                    retrieval_instruction=instruction,
                    chunks=chunks,
                )
    return contexts


# torchrun's rendezvous variables. The rewriter child is a plain single-process job; left
# in its environment they make torch believe it is rank N of a group it never joins.
_DIST_ENV_VARS = frozenset(
    {
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "GROUP_WORLD_SIZE",
        "ROLE_RANK",
        "ROLE_NAME",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    }
)


def _rewrite_worker(request_path: Path, response_path: Path) -> int:
    """Child-process entry point for one rewriter pass. See ``_rewrite_all``."""
    from rag.rewriter import DPORewriter

    request = json.loads(request_path.read_text(encoding="utf-8"))
    rewriter_config = request["rewriter"]
    rewriter = DPORewriter(
        model_name=rewriter_config["model_name"],
        adapter_path=request["adapter_path"],
        device=request["device"],
        max_seq_length=rewriter_config["max_seq_length"],
        generation=rewriter_config["generation"],
    )
    batch_size = int(rewriter_config["batch_size"])
    profiles = request["profiles"]
    queries = request["queries"]
    rewrites: list[str] = []
    for start in range(0, len(queries), batch_size):
        stop = start + batch_size
        rewrites.extend(rewriter.rewrite_batch(profiles[start:stop], queries[start:stop]))
    response_path.write_text(json.dumps(rewrites, ensure_ascii=False), encoding="utf-8")
    return 0


def _rewrite_all(
    config: dict[str, Any], adapter_path: str | None, cases: list[Case], device: str
) -> list[str]:
    """Rewrite every case's query in a child process.

    The child, not this process, is what makes this correct. ``DPORewriter`` imports
    Unsloth, and importing Unsloth rewrites ``transformers``' Qwen3 attention forward pass
    globally to a variant that only Unsloth's own loader prepares (it reads an
    ``apply_qkv`` attribute it attaches itself). Any Qwen3 model loaded afterwards through
    plain ``transformers`` -- here the stage C sentence-transformers embedder -- then dies
    with ``'Qwen3Attention' object has no attribute 'apply_qkv'``. Confining the import to
    a subprocess keeps the patch out of the parent, and exiting also returns the
    rewriter's GPU memory unconditionally.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in _DIST_ENV_VARS and not key.startswith("TORCHELASTIC_")
    }
    with tempfile.TemporaryDirectory() as tmp:
        request_path = Path(tmp) / "request.json"
        response_path = Path(tmp) / "response.json"
        request_path.write_text(
            json.dumps(
                {
                    "rewriter": config["rewriter"],
                    "adapter_path": adapter_path,
                    "device": device,
                    "profiles": [case.profile_rendered for case in cases],
                    "queries": [case.query for case in cases],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--rewrite-worker",
                str(request_path),
                str(response_path),
            ],
            check=True,
            env=env,
        )
        rewrites = json.loads(response_path.read_text(encoding="utf-8"))
    if len(rewrites) != len(cases):
        raise RuntimeError(f"rewriter returned {len(rewrites)} queries for {len(cases)} cases")
    return rewrites


def run_gpu_stages(
    config: dict[str, Any], cases: list[Case], local_rank: int
) -> dict[str, dict[tuple[str, str], ArmContext]]:
    """Stages A-C: every arm's retrieval, four model loads, one at a time in memory."""
    corpus = load_corpus(config["data"]["corpus_path"])
    texts = [row["text"] for row in corpus]
    top_k = int(config["retrieval"]["top_k"])
    device = f"cuda:{local_rank}"
    arms_by_name = {arm["name"]: arm for arm in config["arms"]}
    contexts: dict[str, dict[tuple[str, str], ArmContext]] = {}

    from rag.embedder import Qwen3Embedder

    embedder_config = config["embedder"]

    def make_embedder(adapter_path: str | None) -> Qwen3Embedder:
        return Qwen3Embedder(
            model_name=embedder_config["model_name"],
            device=device,
            batch_size=int(embedder_config["batch_size"]),
            fp16=bool(embedder_config["fp16"]),
            adapter_path=adapter_path,
            max_seq_length=int(embedder_config["max_seq_length"]),
        )

    # No stage A: every arm here is a ROPG arm, which validation enforces.

    # Stage B: rewriters, before the trained embedder is loaded. Two separate loads because
    # DPORewriter exposes no adapter toggle and adding one is out of scope here.
    rewrites: dict[str, list[str]] = {}
    needed = {arm["rewriter"] for arm in config["arms"]}
    if "base" in needed:
        rewrites["base"] = _rewrite_all(config, None, cases, device)
    if "dpo" in needed:
        rewrites["dpo"] = _rewrite_all(
            config, config["artifacts"]["dpo_adapter_path"], cases, device
        )

    # Stage C: ROPG embedder. Run B was promoted with anchor mode `both`
    # (docs/results/ropg-runs-comparison-v1.md), so both towers are adapted and the served
    # index must be built with the adapter, not just the query side.
    #
    # One retrieval pass serves both arms. That is the design of this run: retrieval sees
    # the original question, so the arms differ only in the rewrite they hand the
    # generator. Retrieving per arm would spend a second encode pass to produce identical
    # chunks, and would let a later edit make them quietly differ.
    embedder = make_embedder(config["artifacts"]["ropg_adapter_path"])
    index = _build_index(embedder, texts)
    retrieved = _retrieve_for_arm(
        embedder,
        index,
        corpus,
        {"profile": True},
        cases,
        [case.query for case in cases],
        top_k,
    )
    _free(embedder, index)

    for arm in config["arms"]:
        rewritten = rewrites[arm["rewriter"]]
        contexts[arm["name"]] = {
            (case.question_ref, case.persona_id): replace(
                retrieved[case.question_ref, case.persona_id], answer_query=rewritten[position]
            )
            for position, case in enumerate(cases)
        }

    missing = set(arms_by_name) - set(contexts)
    if missing:
        raise RuntimeError(f"no retrieval was produced for arms {sorted(missing)}")
    return contexts


# --------------------------------------------------------------------------------------
# Stage D: remote generation and judging
# --------------------------------------------------------------------------------------


def _load_completed(rank_path: Path, digest: str) -> tuple[set[tuple[int, str, str, str]], int]:
    """Return keys already finished under *digest*, and the count of foreign-config rows."""
    completed: set[tuple[int, str, str, str]] = set()
    stale = 0
    if not rank_path.is_file():
        return completed, stale
    with rank_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("config_sha256") != digest:
                stale += 1
                continue
            if record.get("status") == "ok":
                completed.add(
                    (
                        record["replicate"],
                        record["arm"],
                        record["question_ref"],
                        record["persona_id"],
                    )
                )
    return completed, stale


def run_remote_stage(
    config: dict[str, Any],
    cases: list[Case],
    contexts: dict[str, dict[tuple[str, str], ArmContext]],
    rank_path: Path,
    digest: str,
    rank: int,
) -> int:
    """Generate and judge every pending key on this rank. Returns the number written."""
    completed, _stale = _load_completed(rank_path, digest)
    case_by_key = {(case.question_ref, case.persona_id): case for case in cases}
    arms_by_name = {arm["name"]: arm for arm in config["arms"]}

    pending = [key for key in build_expected_keys(config, cases) if key not in completed]
    if not pending:
        return 0

    from rag.llm import OpenAICompatClient

    generator_config = config["generator"]
    generator = OpenAICompatClient(
        base_url=GENERATOR_BASE_URL,
        api_key=GENERATOR_API_KEY,
        model=generator_config["model"],
        temperature=float(generator_config["temperature"]),
        max_tokens=int(generator_config["max_completion_tokens"]),
    )
    generator_retry = RetryPolicy.from_config(generator_config["retry"], "generator.retry")

    primary_config = config["judges"]["primary"]
    primary = LunaJudge(
        model=primary_config["model"],
        temperature=float(primary_config["temperature"]),
        reasoning_effort=primary_config.get("reasoning_effort"),
        max_completion_tokens=int(primary_config["max_completion_tokens"]),
        retry=RetryPolicy.from_config(primary_config["retry"], "judges.primary.retry"),
    )
    secondary_config = config["judges"]["secondary"]
    secondary = GeminiJudge(
        model=secondary_config["model"],
        temperature=float(secondary_config["temperature"]),
        thinking_level=secondary_config.get("thinking_level"),
        max_output_tokens=int(secondary_config["max_output_tokens"]),
        retry=RetryPolicy.from_config(secondary_config["retry"], "judges.secondary.retry"),
    )

    ropg_path = config["artifacts"]["ropg_adapter_path"]
    dpo_path = config["artifacts"]["dpo_adapter_path"]
    lock = threading.Lock()
    handle = rank_path.open("a", encoding="utf-8")

    def evaluate(key: tuple[int, str, str, str]) -> None:
        replicate, arm_name, question_ref, persona_id = key
        arm = arms_by_name[arm_name]
        case = case_by_key[question_ref, persona_id]
        context = contexts[arm_name][question_ref, persona_id]
        passages = [(chunk.chunk_id, chunk.text) for chunk in context.chunks]

        record: dict[str, Any] = {
            "replicate": replicate,
            "arm": arm_name,
            "question_ref": question_ref,
            "persona_id": persona_id,
            "status": "ok",
            "failure_stage": None,
            "rank": rank,
            "config_sha256": digest,
            "embedder_model": config["embedder"]["model_name"],
            "embedder_adapter_path": ropg_path if arm["embedder"] == "ropg" else None,
            "rewriter_model": config["rewriter"]["model_name"]
            if arm["rewriter"] != "none"
            else None,
            "rewriter_adapter_path": dpo_path if arm["rewriter"] == "dpo" else None,
            "generator_model": generator_config["model"],
            "primary_judge_model": primary_config["model"],
            "secondary_judge_model": secondary_config["model"],
            "retrieval_instruction": context.retrieval_instruction,
            "original_query": case.query,
            "rewritten_query": context.answer_query,
            # Recorded explicitly so a row is self-describing: in the ablation's
            # results.jsonl `rewritten_query` is what retrieval used, here it is what the
            # generator was asked, and the two files are joined in analysis.
            "retrieval_query": context.retrieval_query,
            "rewrite_target": "question",
            "hits": [chunk.as_dict() for chunk in context.chunks],
            "answer": None,
            "primary_scores": None,
            "secondary_scores": None,
            "primary_attempts": 0,
            "secondary_attempts": 0,
            "primary_error": None,
            "secondary_error": None,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        messages = build_answer_messages(
            case.profile_rendered if arm["profile"] else None, context.answer_query, context.chunks
        )
        try:
            answer, _attempts = _call_with_retry(
                lambda: generator.chat(messages), generator_retry, generator_config["model"]
            )
        except RemoteCallError as exc:
            record["status"] = "failed"
            record["failure_stage"] = "generation"
            record["primary_error"] = str(exc)
            _append(handle, lock, record)
            return
        record["answer"] = answer

        # The judge sees the learner profile and the *original* question, never the
        # rewrite. The exam question and its gold answer are the ground truth the answer is
        # accountable to; scoring against the rewrite would let a rewriter that changes the
        # question earn a good grade for answering a question nobody asked.
        judge_kwargs = {
            "profile_rendered": case.profile_rendered,
            "query": case.query,
            "gold_answer": case.gold_answer,
            "gold_explanation": case.gold_explanation,
            "passages": passages,
            "answer": answer,
        }
        try:
            scores, attempts = primary.score(**judge_kwargs)
        except JudgeError as exc:
            record["status"] = "failed"
            record["failure_stage"] = "judge_primary"
            record["primary_error"] = str(exc)
            _append(handle, lock, record)
            return
        record["primary_scores"] = scores.as_dict()
        record["primary_attempts"] = attempts

        # A secondary failure is not a row failure: the row still carries a primary score
        # and appears everywhere except judge agreement.
        try:
            scores, attempts = secondary.score(**judge_kwargs)
        except JudgeError as exc:
            record["secondary_error"] = str(exc)
        else:
            record["secondary_scores"] = scores.as_dict()
            record["secondary_attempts"] = attempts
        _append(handle, lock, record)

    try:
        with ThreadPoolExecutor(max_workers=int(config["execution"]["max_workers"])) as pool:
            list(pool.map(evaluate, pending))
    finally:
        handle.close()
    return len(pending)


def _append(handle: Any, lock: threading.Lock, record: dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False)
    with lock:
        handle.write(line + "\n")
        handle.flush()


# --------------------------------------------------------------------------------------
# Merge and reporting
# --------------------------------------------------------------------------------------


def collect_records(output_dir: Path, world_size: int, digest: str) -> tuple[list[dict], int]:
    records: list[dict[str, Any]] = []
    stale = 0
    for rank in range(world_size):
        path = output_dir / f"rank{rank}.jsonl"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("config_sha256") != digest:
                    stale += 1
                    continue
                records.append(record)
    return records, stale


def resolve_records(
    records: list[dict[str, Any]], expected: list[tuple[int, str, str, str]]
) -> list[dict[str, Any]]:
    """One record per expected key: the success if there is one, else the last failure."""
    by_key: dict[tuple[int, str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (record["replicate"], record["arm"], record["question_ref"], record["persona_id"])
        by_key.setdefault(key, []).append(record)

    resolved: list[dict[str, Any]] = []
    duplicated: list[tuple[int, str, str, str]] = []
    missing: list[tuple[int, str, str, str]] = []
    for key in expected:
        found = by_key.get(key, [])
        successes = [record for record in found if record.get("status") == "ok"]
        if len(successes) > 1:
            duplicated.append(key)
        elif successes:
            resolved.append(successes[0])
        elif found:
            resolved.append(found[-1])
        else:
            missing.append(key)
    if duplicated:
        raise RuntimeError(f"{len(duplicated)} keys have more than one success: {duplicated[:5]}")
    if missing:
        raise RuntimeError(f"{len(missing)} expected keys have no record at all: {missing[:5]}")

    unexpected = set(by_key) - set(expected)
    if unexpected:
        raise RuntimeError(
            f"{len(unexpected)} records do not belong to this run: {sorted(unexpected)[:5]}"
        )
    return resolved


def _scores(record: dict[str, Any], role: str) -> dict[str, int] | None:
    return record.get(f"{role}_scores")


def _persona_groups(personas: list[str]) -> list[tuple[str, str, set[str]]]:
    """``(persona_label, persona_split_label, member ids)`` for every summary grouping."""
    groups = [(p, PERSONAS[p].split, {p}) for p in personas]
    for split in ("train", "test"):
        members = {p for p in personas if PERSONAS[p].split == split}
        if members:
            groups.append((f"all_{split}", split, members))
    groups.append(("all", "all", set(personas)))
    return groups


def write_summary(path: Path, records: list[dict[str, Any]], config: dict[str, Any]) -> None:
    personas = list(config["personas"])
    replicate_groups: list[tuple[Any, set[int]]] = [(r, {r}) for r in config["replicates"]]
    replicate_groups.append(("all", set(config["replicates"])))

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "arm",
                "persona",
                "persona_split",
                "replicate",
                "judge",
                "metric",
                "mean",
                "std",
                "n_valid",
                "n_failed",
            ]
        )
        for arm in ARM_NAMES:
            for persona_label, split_label, members in _persona_groups(personas):
                for replicate_label, replicate_members in replicate_groups:
                    # Only the pooled-replicate row is emitted for the pooled persona
                    # groupings: a per-replicate, per-split cell answers no question the
                    # other rows do not, and multiplies the file for nothing.
                    if persona_label.startswith("all") and replicate_label != "all":
                        continue
                    rows = [
                        record
                        for record in records
                        if record["arm"] == arm
                        and record["persona_id"] in members
                        and record["replicate"] in replicate_members
                    ]
                    if not rows:
                        continue
                    for role in JUDGE_ROLES:
                        for metric in SCORE_FIELDS:
                            values = [
                                _scores(record, role)[metric]
                                for record in rows
                                if _scores(record, role) is not None
                            ]
                            n_valid = len(values)
                            array = np.asarray(values, dtype=float)
                            mean = float(array.mean()) if n_valid else ""
                            std = float(array.std(ddof=1)) if n_valid > 1 else 0.0
                            writer.writerow(
                                [
                                    arm,
                                    persona_label,
                                    split_label,
                                    replicate_label,
                                    role,
                                    metric,
                                    mean,
                                    std,
                                    n_valid,
                                    len(rows) - n_valid,
                                ]
                            )


def write_paired_deltas(path: Path, records: list[dict[str, Any]], config: dict[str, Any]) -> None:
    """Adjacent-rung paired deltas, Holm-corrected across the whole reported family.

    Pairing is what buys the power here: the same question and persona is scored under both
    arms, so per-question difficulty — the dominant variance term — cancels in the
    difference. Holm runs once over every row in the file rather than per comparison,
    because the family a reader sees is the file.
    """
    personas = list(config["personas"])
    persona_groups = [(p, PERSONAS[p].split, {p}) for p in personas]
    persona_groups.append(("all", "all", set(personas)))
    n_boot = int(config["execution"]["bootstrap_samples"])
    seed = int(config["execution"]["bootstrap_seed"])

    indexed: dict[tuple[str, str, str, int], dict[str, Any]] = {
        (r["arm"], r["question_ref"], r["persona_id"], r["replicate"]): r for r in records
    }
    rows: list[list[Any]] = []
    for comparison, before_arm, after_arm in COMPARISONS:
        for persona_label, split_label, members in persona_groups:
            for role in JUDGE_ROLES:
                for metric in SCORE_FIELDS:
                    before_values: list[float] = []
                    after_values: list[float] = []
                    for question_ref, persona_id, replicate in sorted(
                        {
                            (r["question_ref"], r["persona_id"], r["replicate"])
                            for r in records
                            if r["persona_id"] in members
                        }
                    ):
                        before = indexed.get((before_arm, question_ref, persona_id, replicate))
                        after = indexed.get((after_arm, question_ref, persona_id, replicate))
                        if before is None or after is None:
                            continue
                        before_scores = _scores(before, role)
                        after_scores = _scores(after, role)
                        if before_scores is None or after_scores is None:
                            continue
                        before_values.append(before_scores[metric])
                        after_values.append(after_scores[metric])
                    if not before_values:
                        continue
                    stats = paired_stats(
                        np.asarray(before_values, dtype=float),
                        np.asarray(after_values, dtype=float),
                        n_boot=n_boot,
                        rng=np.random.default_rng(seed),
                    )
                    rows.append(
                        [
                            comparison,
                            persona_label,
                            split_label,
                            role,
                            metric,
                            stats["before"],
                            stats["after"],
                            stats["delta"],
                            stats["lo"],
                            stats["hi"],
                            stats["p"],
                            None,
                            stats["n"],
                        ]
                    )

    adjusted = holm([row[10] for row in rows]) if rows else []
    for row, p_holm in zip(rows, adjusted, strict=True):
        row[11] = p_holm

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "comparison",
                "persona",
                "persona_split",
                "judge",
                "metric",
                "before",
                "after",
                "delta",
                "ci_lo",
                "ci_hi",
                "p",
                "p_holm",
                "n",
            ]
        )
        writer.writerows(rows)


def write_judge_agreement(path: Path, records: list[dict[str, Any]]) -> None:
    """How far the two judges agree, pooled and per arm.

    Per arm is not optional: disagreement concentrated in one arm is what reward hacking
    against the primary judge looks like from the outside.
    """
    groups: list[tuple[str, list[dict[str, Any]]]] = [("all", records)]
    groups.extend(
        (arm, [record for record in records if record["arm"] == arm]) for arm in ARM_NAMES
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["arm", "persona", "metric", "n", "exact_agreement_rate", "mean_abs_diff", "pearson_r"]
        )
        for arm_label, rows in groups:
            both = [
                record
                for record in rows
                if _scores(record, "primary") is not None
                and _scores(record, "secondary") is not None
            ]
            for metric in SCORE_FIELDS:
                if not both:
                    writer.writerow([arm_label, "all", metric, 0, "", "", ""])
                    continue
                a = np.asarray([_scores(r, "primary")[metric] for r in both], dtype=float)
                b = np.asarray([_scores(r, "secondary")[metric] for r in both], dtype=float)
                exact = float((a == b).mean())
                mad = float(np.abs(a - b).mean())
                # Degenerate on a constant vector, which happens whenever a metric
                # saturates. numpy warns and returns nan; report the nan rather than a 0.
                if a.std() == 0 or b.std() == 0:
                    corr = float("nan")
                else:
                    corr = float(np.corrcoef(a, b)[0, 1])
                writer.writerow([arm_label, "all", metric, len(both), exact, mad, corr])


def _package_version(name: str) -> str | None:
    import importlib.metadata

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_manifest(
    path: Path,
    config: dict[str, Any],
    digest: str,
    records: list[dict[str, Any]],
    expected: list[tuple[int, str, str, str]],
    world_size: int,
    stale_records: int,
) -> None:
    failures: dict[str, int] = {}
    for record in records:
        if record["status"] != "ok":
            stage = record["failure_stage"] or "unknown"
            failures[stage] = failures.get(stage, 0) + 1
    secondary_missing = sum(
        1 for record in records if record["status"] == "ok" and record["secondary_scores"] is None
    )
    per_rank: dict[str, int] = {}
    for record in records:
        key = str(record["rank"])
        per_rank[key] = per_rank.get(key, 0) + 1

    manifest = {
        "config": config,
        "config_sha256": digest,
        "corpus": {
            "path": config["data"]["corpus_path"],
            "sha256": sha256_file(config["data"]["corpus_path"]),
            "count": len(load_corpus(config["data"]["corpus_path"])),
        },
        "test_qids": {
            "path": config["data"]["test_qids_path"],
            "sha256": sha256_file(config["data"]["test_qids_path"]),
            "count": len(read_test_qids(config["data"]["test_qids_path"])),
        },
        "personas": [
            {"id": p, "split": PERSONAS[p].split, "rendered": PERSONAS[p].rendered}
            for p in config["personas"]
        ],
        "arms": config["arms"],
        "replicates": config["replicates"],
        "adapters": {
            label: {
                "path": str(Path(path_value)),
                "base_model": _adapter_base_model(Path(path_value)),
            }
            for label, path_value in (
                ("ropg", config["artifacts"]["ropg_adapter_path"]),
                ("dpo", config["artifacts"]["dpo_adapter_path"]),
            )
        },
        "models": {
            "generator": {
                "model": config["generator"]["model"],
                "temperature": config["generator"]["temperature"],
                "max_completion_tokens": config["generator"]["max_completion_tokens"],
            },
            "judge_primary": dict(config["judges"]["primary"]),
            "judge_secondary": dict(config["judges"]["secondary"]),
        },
        "world_size": world_size,
        "records_per_rank": per_rank,
        "expected_keys": len(expected),
        "merged_keys": len(records),
        "successful_keys": sum(1 for record in records if record["status"] == "ok"),
        "failures_by_stage": failures,
        "secondary_judge_missing": secondary_missing,
        "stale_records": stale_records,
        "versions": {
            name: _package_version(name)
            for name in (
                "torch",
                "transformers",
                "peft",
                "sentence-transformers",
                "faiss-gpu",
                "faiss-cpu",
                "openai",
                "google-genai",
            )
        },
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def merge(
    config: dict[str, Any],
    cases: list[Case],
    digest: str,
    world_size: int,
) -> dict[str, Any]:
    output_dir = Path(config["output_dir"])
    expected = build_expected_keys(config, cases)
    records, stale = collect_records(output_dir, world_size, digest)
    resolved = resolve_records(records, expected)
    resolved.sort(key=lambda r: (r["replicate"], r["arm"], r["question_ref"], r["persona_id"]))

    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for record in resolved:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    successful = [record for record in resolved if record["status"] == "ok"]
    write_summary(output_dir / "summary.csv", successful, config)
    write_paired_deltas(output_dir / "paired_deltas.csv", successful, config)
    write_judge_agreement(output_dir / "judge_agreement.csv", successful)
    write_manifest(
        output_dir / "run_manifest.json", config, digest, resolved, expected, world_size, stale
    )
    return {
        "expected": len(expected),
        "merged": len(resolved),
        "successful": len(successful),
        "stale": stale,
    }


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Intercepted ahead of argparse: the worker is an internal re-entry, not a user-facing
    # mode, and it must not touch the config, the rank environment, or the output dir.
    if argv[:1] == ["--rewrite-worker"]:
        return _rewrite_worker(Path(argv[1]), Path(argv[2]))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    digest = config_digest(config)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    report = validate(config)
    if args.preflight_only:
        if rank == 0:
            print_preflight(report)
        return 0

    cases = build_cases(config, args.limit)
    shard = cases[rank::world_size]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_path = output_dir / f"rank{rank}.jsonl"

    if world_size > 1:
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(local_rank)
        # gloo, not NCCL: this job is inference-only and its one collective is a barrier,
        # so no GPU tensor is ever communicated.
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=10),
        )

    print(f"[rank {rank}] {len(shard)} cases, config {digest[:12]}", flush=True)
    contexts = run_gpu_stages(config, shard, local_rank)
    written = run_remote_stage(config, shard, contexts, rank_path, digest, rank)
    print(f"[rank {rank}] wrote {written} records", flush=True)

    if world_size > 1:
        import torch.distributed as dist

        dist.barrier()

    if rank == 0:
        stats = merge(config, cases, digest, world_size)
        print(
            f"Merged {stats['merged']}/{stats['expected']} keys "
            f"({stats['successful']} successful, {stats['stale']} stale) -> {output_dir}"
        )

    if world_size > 1:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
