"""Offline contracts for the question-side rewriting runner.

``check_eval_offline.py`` already exercises the shared machinery — judge parsing, retry,
merge/resume, and the reporting writers — and both runners import that machinery from the
same modules. This file checks only what ``eval_rewrite_stage.py`` does *differently*, with
no GPU, no network, and no adapter files:

* the generator is asked the rewritten question while the judge is asked the original one;
* every arm-level rule the new config depends on is enforced;
* the new config differs from ``configs/eval_ablation.yaml`` in exactly the blocks it is
  allowed to differ in, because the two runs are compared by joining their results.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
for _p in (str(SRC_DIR), str(BENCHMARKS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ruff: noqa: E402  (deliberate mid-file imports; sys.path must be set first)
import yaml

import eval_rewrite_stage as runner
from eval_rewrite_stage import (
    ARM_NAMES,
    ArmContext,
    Case,
    RetrievedChunk,
    build_expected_keys,
    run_remote_stage,
)

ORIGINAL_QUESTION = "ORIGINAL-STEM: what does the passage say?"
PROFILE = "a ninth-grader who reads well"


class _Scores:
    """Stands in for ``judge.Scores``: the runner only calls ``as_dict``."""

    def as_dict(self) -> dict[str, int]:
        return {
            "context_utility": 3,
            "answer_correctness": 1,
            "faithfulness": 3,
            "persona_alignment": 3,
            "pedagogical_quality": 3,
        }


class _RecordingClient:
    """Captures the message list the runner builds instead of calling an endpoint."""

    def __init__(self, **_kwargs: Any) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return "answer"


class _RecordingJudge:
    def __init__(self, **_kwargs: Any) -> None:
        self.calls: list[dict[str, Any]] = []

    def score(self, **kwargs: Any) -> tuple[_Scores, int]:
        self.calls.append(kwargs)
        return _Scores(), 1


def _config(output_dir: Path) -> dict[str, Any]:
    return {
        "personas": ["crammer"],
        "replicates": [1],
        "arms": [
            {"name": name, "embedder": "ropg", "profile": True, "rewriter": kind}
            for name, kind in zip(ARM_NAMES, ("base", "dpo"), strict=True)
        ],
        "output_dir": str(output_dir),
        "embedder": {"model_name": "Qwen/Qwen3-Embedding-0.6B"},
        "rewriter": {"model_name": "Qwen/Qwen3-4B"},
        "artifacts": {"ropg_adapter_path": "ropg", "dpo_adapter_path": "dpo"},
        "generator": {
            "model": "generator",
            "temperature": 0.2,
            "max_completion_tokens": 16,
            "retry": {
                "max_attempts": 1,
                "initial_backoff_seconds": 0.0,
                "backoff_multiplier": 1.0,
                "max_backoff_seconds": 0.0,
            },
        },
        "judges": {
            "primary": {
                "model": "primary",
                "temperature": 1.0,
                "max_completion_tokens": 16,
                "retry": {
                    "max_attempts": 1,
                    "initial_backoff_seconds": 0.0,
                    "backoff_multiplier": 1.0,
                    "max_backoff_seconds": 0.0,
                },
            },
            "secondary": {
                "model": "secondary",
                "temperature": 0.0,
                "max_output_tokens": 16,
                "retry": {
                    "max_attempts": 1,
                    "initial_backoff_seconds": 0.0,
                    "backoff_multiplier": 1.0,
                    "max_backoff_seconds": 0.0,
                },
            },
        },
        "execution": {"max_workers": 1},
    }


def check_question_side_wiring() -> None:
    """The rewrite reaches the generator; the original question reaches the judge.

    This is the entire experiment. If the generator sees the original question the run is
    a duplicate of rung 3 at full cost, and if the judge sees the rewrite then a rewriter
    that changes the question is graded on the question it invented.
    """
    case = Case(
        question_ref="exam:q1",
        persona_id="crammer",
        query=ORIGINAL_QUESTION,
        gold_answer="gold",
        gold_explanation="because",
        profile_rendered=PROFILE,
    )
    chunk = RetrievedChunk(chunk_id="chunk-1", text="passage text", score=1.0, rank=1)
    contexts = {
        arm: {
            ("exam:q1", "crammer"): ArmContext(
                retrieval_query=ORIGINAL_QUESTION,
                retrieval_instruction=PROFILE,
                chunks=[chunk],
                answer_query=f"REWRITTEN-BY-{arm}",
            )
        }
        for arm in ARM_NAMES
    }

    client = _RecordingClient()
    primary = _RecordingJudge()
    secondary = _RecordingJudge()
    stub_llm = types.ModuleType("rag.llm")
    stub_llm.OpenAICompatClient = lambda **kwargs: client
    original_llm = sys.modules.get("rag.llm")
    sys.modules["rag.llm"] = stub_llm
    original_judges = (runner.LunaJudge, runner.GeminiJudge)
    runner.LunaJudge = lambda **kwargs: primary
    runner.GeminiJudge = lambda **kwargs: secondary
    try:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            config = _config(output_dir)
            rank_path = output_dir / "rank0.jsonl"
            written = run_remote_stage(
                config, [case], contexts, rank_path, digest="digest", rank=0
            )
            assert written == len(ARM_NAMES), written
            records = [
                json.loads(line)
                for line in rank_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    finally:
        if original_llm is None:
            del sys.modules["rag.llm"]
        else:
            sys.modules["rag.llm"] = original_llm
        runner.LunaJudge, runner.GeminiJudge = original_judges

    assert {record["arm"] for record in records} == set(ARM_NAMES)
    for record in records:
        assert record["status"] == "ok", record
        assert record["rewrite_target"] == "question"
        assert record["rewritten_query"] == f"REWRITTEN-BY-{record['arm']}"
        assert record["retrieval_query"] == ORIGINAL_QUESTION
        assert record["original_query"] == ORIGINAL_QUESTION

    prompts = ["".join(message["content"] for message in call) for call in client.calls]
    assert len(prompts) == len(ARM_NAMES)
    for arm, prompt in zip(sorted(ARM_NAMES), sorted(prompts), strict=True):
        del arm
        assert "REWRITTEN-BY-" in prompt, prompt
        assert ORIGINAL_QUESTION not in prompt, prompt
        assert PROFILE in prompt, "the learner profile must still reach the generator"
        assert "passage text" in prompt

    for judge in (primary, secondary):
        assert len(judge.calls) == len(ARM_NAMES)
        for call in judge.calls:
            assert call["query"] == ORIGINAL_QUESTION
            assert "REWRITTEN-BY-" not in call["query"]
            assert call["gold_answer"] == "gold"
            assert call["profile_rendered"] == PROFILE
    print("OK question-side prompt wiring")


def _rejection_message(mutate) -> str:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "eval_rewrite_stage.yaml").read_text())
    mutate(config)
    try:
        runner.validate(config, probe=False)
    except ValueError as error:
        return str(error)
    raise AssertionError("expected the config to be rejected")


def check_arm_rules() -> None:
    """Each arm rule rejects with its own message, on this config, not a fabricated one."""
    unprofiled = _rejection_message(lambda config: config["arms"][0].update({"profile": False}))
    assert "profile must be true" in unprofiled, unprofiled

    no_rewriter = _rejection_message(lambda config: config["arms"][1].update({"rewriter": "none"}))
    assert "rewriter must be base or dpo" in no_rewriter, no_rewriter

    base_embedder = _rejection_message(
        lambda config: config["arms"][0].update({"embedder": "base"})
    )
    assert "embedder must be ropg" in base_embedder, base_embedder

    renamed = _rejection_message(lambda config: config["arms"][0].update({"name": "something"}))
    assert "arms must be exactly" in renamed, renamed

    # The shipped config itself must survive validation, arm rules included. Data paths in
    # it are repo-relative, so this passes from the repository root and reports the real
    # blocking reason from anywhere else.
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "eval_rewrite_stage.yaml").read_text())
    try:
        report = runner.validate(config, probe=False)
    except ValueError as error:
        raise AssertionError(f"the shipped config does not validate: {error}") from error
    assert report["arms"] == list(ARM_NAMES), report["arms"]
    assert report["expected_keys"] == report["n_questions"] * len(report["personas"]) * len(
        ARM_NAMES
    ), report
    print("OK arm rules")


def check_config_parity() -> None:
    """Everything outside `arms` and `output_dir` must match the ablation config.

    The two runs are compared by joining their `results.jsonl` on (question_ref,
    persona_id). A drifted generator temperature or judge model would silently turn "where
    the rewrite is applied" into "and four other things".
    """
    ablation = yaml.safe_load((PROJECT_ROOT / "configs" / "eval_ablation.yaml").read_text())
    rewrite = yaml.safe_load((PROJECT_ROOT / "configs" / "eval_rewrite_stage.yaml").read_text())
    allowed = {"arms", "output_dir"}
    assert set(ablation) == set(rewrite), set(ablation) ^ set(rewrite)
    for key in sorted(set(ablation) - allowed):
        assert ablation[key] == rewrite[key], f"{key} differs between the two configs"
    assert rewrite["output_dir"] != ablation["output_dir"], (
        "a shared output_dir would let the two runs write into one another's rank files"
    )
    assert [arm["name"] for arm in rewrite["arms"]] == list(ARM_NAMES)
    print("OK config parity with the ablation")


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
    config = {"replicates": [1], "arms": [{"name": name} for name in ARM_NAMES]}
    keys = build_expected_keys(config, cases)
    assert len(keys) == 3 * len(personas) * len(ARM_NAMES)
    assert len(set(keys)) == len(keys)
    print("OK expected-key arithmetic")


def check_shared_retrieval() -> None:
    """One retrieval pass feeds both arms, and each arm keeps its own rewrite, in order.

    The GPU work is stubbed; the wiring under test is pure bookkeeping and is exactly where
    an off-by-one would be invisible — every row would still look well-formed while
    carrying another case's rewritten question.
    """
    personas = ("crammer", "newcomer")
    cases = [
        Case(
            question_ref=f"exam:q{number}",
            persona_id=persona_id,
            query=f"original-{number}",
            gold_answer="gold",
            gold_explanation="because",
            profile_rendered=PROFILE,
        )
        for number in range(1, 4)
        for persona_id in personas
    ]
    retrieval_calls: list[list[str]] = []

    def fake_retrieve(_embedder, _index, _corpus, arm, cases_arg, queries, _top_k):
        assert arm["profile"] is True
        retrieval_calls.append(list(queries))
        return {
            (case.question_ref, case.persona_id): ArmContext(
                retrieval_query=query,
                retrieval_instruction=case.profile_rendered,
                chunks=[
                    RetrievedChunk(chunk_id=f"chunk-{query}", text="passage", score=1.0, rank=1)
                ],
            )
            for case, query in zip(cases_arg, queries, strict=True)
        }

    def fake_rewrite(_config, adapter_path, cases_arg, _device):
        kind = "dpo" if adapter_path else "base"
        return [f"{kind}-rewrite-of-{case.query}" for case in cases_arg]

    stub_embedder = types.ModuleType("rag.embedder")
    stub_embedder.Qwen3Embedder = lambda **kwargs: object()
    original_embedder = sys.modules.get("rag.embedder")
    sys.modules["rag.embedder"] = stub_embedder
    originals = (runner._build_index, runner._retrieve_for_arm, runner._rewrite_all, runner._free)
    runner._build_index = lambda _embedder, _texts: object()
    runner._retrieve_for_arm = fake_retrieve
    runner._rewrite_all = fake_rewrite
    runner._free = lambda *objects: None
    try:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            corpus_path = output_dir / "corpus.jsonl"
            corpus_path.write_text(
                json.dumps({"chunk_id": "chunk-1", "text": "passage"}) + "\n", encoding="utf-8"
            )
            config = _config(output_dir)
            config["data"] = {"corpus_path": str(corpus_path)}
            config["retrieval"] = {"top_k": 5}
            config["embedder"].update({"batch_size": 1, "fp16": False, "max_seq_length": 16})
            contexts = runner.run_gpu_stages(config, cases, local_rank=0)
    finally:
        if original_embedder is None:
            del sys.modules["rag.embedder"]
        else:
            sys.modules["rag.embedder"] = original_embedder
        runner._build_index, runner._retrieve_for_arm, runner._rewrite_all, runner._free = (
            originals
        )

    assert len(retrieval_calls) == 1, f"{len(retrieval_calls)} retrieval passes; expected one"
    assert retrieval_calls[0] == [case.query for case in cases], (
        "retrieval was handed something other than the original questions"
    )
    assert set(contexts) == set(ARM_NAMES)
    base_arm, dpo_arm = ARM_NAMES
    for case in cases:
        key = (case.question_ref, case.persona_id)
        base, dpo = contexts[base_arm][key], contexts[dpo_arm][key]
        assert base.retrieval_query == case.query
        assert dpo.retrieval_query == case.query
        assert base.answer_query == f"base-rewrite-of-{case.query}"
        assert dpo.answer_query == f"dpo-rewrite-of-{case.query}"
        assert [chunk.chunk_id for chunk in base.chunks] == [
            chunk.chunk_id for chunk in dpo.chunks
        ], "the two arms received different passages"
    print("OK shared retrieval and per-arm rewrites")


def main() -> int:
    check_question_side_wiring()
    check_arm_rules()
    check_config_parity()
    check_expected_keys()
    check_shared_retrieval()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
