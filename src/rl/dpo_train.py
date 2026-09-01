"""Train the persona-conditioned Qwen3 query rewriter with DPO-family objectives.

The module owns the data, token-budget, objective, and distributed-training contract used
by both the command-line entry point and the generated Kaggle notebook.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import logging
import math
import os
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = ("question_ref", "persona_id", "query", "chosen", "rejected")
_PINNED_PACKAGES = {
    "unsloth": "2026.8.22",
    "trl": "0.24.0",
    "transformers": "4.57.6",
    "datasets": "4.3.0",
    "peft": "0.18.0",
    "accelerate": "1.14.0",
}
_REQUIRED_DPO_PARAMETERS = {
    "max_prompt_length",
    "max_completion_length",
    "use_weighting",
    "precompute_ref_log_probs",
    "rpo_alpha",
}
# Every field here is part of an arm's identity, so `resolve_arm` demands an exact match
# against the config. `rpo_alpha` adds TRL's supervised NLL term on the chosen response;
# without it the objective only widens the chosen/rejected margin, which the seed-42
# screening run satisfied by pushing both log-probabilities down (see
# docs/results/dpo-arms-seed42-v1.md). TRL scales the anchor by the WPO weight in the `wpo`
# arm, so the same alpha is a weaker anchor there; that asymmetry is left explicit.
_ARM_CONTRACT = {
    "dpo": {
        "loss_type": "sigmoid",
        "use_weighting": False,
        "label_smoothing": 0.0,
        "rpo_alpha": 1.0,
    },
    "wpo": {
        "loss_type": "sigmoid",
        "use_weighting": True,
        "label_smoothing": 0.0,
        "rpo_alpha": 1.0,
    },
    "robust_dpo": {
        "loss_type": "robust",
        "use_weighting": False,
        "label_smoothing": 0.1,
        "rpo_alpha": 1.0,
    },
}
_INSTALL_COMMAND = (
    "pip uninstall -y torchao && "
    "pip install unsloth==2026.8.22 trl==0.24.0 transformers==4.57.6 "
    "datasets==4.3.0 peft==0.18.0 accelerate==1.14.0"
)


def _is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _verify_training_packages() -> dict[str, str]:
    versions: dict[str, str] = {}
    mismatches: list[str] = []
    for package, expected in _PINNED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        versions[package] = actual
        if actual != expected:
            mismatches.append(f"{package}=={actual} (expected {expected})")

    try:
        torchao_version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        versions["torchao"] = "not installed"
    else:
        versions["torchao"] = torchao_version
        mismatches.append(
            f"torchao=={torchao_version} is installed, but this bitsandbytes QLoRA path "
            "requires the optional TorchAO PEFT backend to be absent"
        )

    if mismatches:
        raise RuntimeError(
            "Incompatible DPO training environment (package mismatch: "
            + ", ".join(mismatches)
            + "). Install:\n  "
            + _INSTALL_COMMAND
        )
    return versions


def _verify_dpo_config_interface(dpo_config: type) -> None:
    parameters = set(inspect.signature(dpo_config).parameters)
    missing_parameters = sorted(_REQUIRED_DPO_PARAMETERS - parameters)
    if missing_parameters:
        raise RuntimeError(
            "Incompatible DPO training environment (DPOConfig lacks: "
            + ", ".join(missing_parameters)
            + "). Install:\n  "
            + _INSTALL_COMMAND
        )


def verify_training_environment() -> dict[str, str]:
    """Verify the package tuple and TRL interface before allocating a model."""
    versions = _verify_training_packages()
    try:
        from trl import DPOConfig
    except Exception as exc:
        raise RuntimeError(
            "Cannot import TRL's DPOConfig. Install the pinned training tuple with:\n"
            f"  {_INSTALL_COMMAND}"
        ) from exc

    _verify_dpo_config_interface(DPOConfig)
    logger.info("Training packages: %s", ", ".join(f"{k}={v}" for k, v in versions.items()))
    return versions


def load_pairs(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate one versioned preference-pair JSONL file."""
    from data.gen_dpo_data import DPO_OUTPUT_FORMAT_VERSION
    from personalization.profiles import train_personas

    pair_path = Path(path)
    if not pair_path.is_file():
        raise FileNotFoundError(f"Preference-pair file not found: {pair_path}")

    known_personas = {persona.id for persona in train_personas()}
    pairs: list[dict[str, Any]] = []
    seen_rows: dict[str, int] = {}
    for line_number, line in enumerate(pair_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{pair_path} line {line_number} is invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{pair_path} line {line_number} must contain a JSON object")

        version = record.get("format_version")
        if version != DPO_OUTPUT_FORMAT_VERSION:
            raise ValueError(
                f"{pair_path} line {line_number} has format version {version!r}, expected "
                f"{DPO_OUTPUT_FORMAT_VERSION}; regenerate with src/data/gen_dpo_data.py"
            )
        for field in _REQUIRED_FIELDS:
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{pair_path} line {line_number} field {field!r} must be a nonempty string"
                )
        if record["persona_id"] not in known_personas:
            raise ValueError(
                f"{pair_path} line {line_number} has unknown training persona "
                f"{record['persona_id']!r}; expected one of {sorted(known_personas)}"
            )
        if record["chosen"].strip() == record["rejected"].strip():
            raise ValueError(
                f"{pair_path} line {line_number} has identical chosen and rejected completions"
            )

        row_key = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if row_key in seen_rows:
            raise ValueError(
                f"{pair_path} line {line_number} duplicates line {seen_rows[row_key]} exactly"
            )
        seen_rows[row_key] = line_number
        pairs.append(record)

    if not pairs:
        raise ValueError(f"Preference-pair file is empty: {pair_path}")
    return pairs


def _split_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(pairs),
        "questions": len({pair["question_ref"] for pair in pairs}),
        "question_persona_keys": len(
            {(pair["question_ref"], pair["persona_id"]) for pair in pairs}
        ),
        "personas": dict(sorted(Counter(pair["persona_id"] for pair in pairs).items())),
    }


def validate_splits(
    train_pairs: list[dict[str, Any]], val_pairs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Reject question leakage and return the split report stored in the manifest."""
    train_questions = {pair["question_ref"] for pair in train_pairs}
    val_questions = {pair["question_ref"] for pair in val_pairs}
    overlap = sorted(train_questions & val_questions)
    if overlap:
        sample = ", ".join(overlap[:5])
        raise ValueError(
            f"Train/validation question_ref overlap: {len(overlap)} questions; first: {sample}"
        )
    return {
        "train": _split_summary(train_pairs),
        "validation": _split_summary(val_pairs),
        "question_overlap": 0,
        "score_analysis": "unavailable: pair files do not store judge scores",
        "pair_type_analysis": "unavailable: pair files do not store pair provenance",
    }


def _render_pair(pair: dict[str, Any], tokenizer: PreTrainedTokenizerBase) -> dict[str, str]:
    from personalization.profiles import render_profile
    from rag.rewriter import build_rewrite_messages

    messages = build_rewrite_messages(render_profile(pair["persona_id"]), pair["query"])
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return {"prompt": prompt, "chosen": pair["chosen"], "rejected": pair["rejected"]}


def pairs_to_dataset(pairs: list[dict[str, Any]], tokenizer: PreTrainedTokenizerBase):
    """Convert validated rows to the explicit-prompt schema expected by DPOTrainer."""
    from datasets import Dataset

    return Dataset.from_list([_render_pair(pair, tokenizer) for pair in pairs])


def _percentile_nearest_rank(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _length_summary(values: list[int], limit: int) -> dict[str, int | float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p95": _percentile_nearest_rank(values, 0.95),
        "max": max(values),
        "limit": limit,
        "over_limit": sum(value > limit for value in values),
    }


def _completion_token_length(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None or not token_ids or token_ids[-1] != eos_token_id:
        return len(token_ids) + 1
    return len(token_ids)


def measure_token_lengths(
    dataset: Any,
    tokenizer: PreTrainedTokenizerBase,
    training_config: dict[str, Any],
    *,
    split_name: str,
) -> dict[str, Any]:
    """Measure TRL's prompt/completion/full lengths and reject every truncation."""
    prompt_lengths: list[int] = []
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []
    chosen_full_lengths: list[int] = []
    rejected_full_lengths: list[int] = []

    for row in dataset:
        prompt_length = len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"])
        chosen_length = _completion_token_length(row["chosen"], tokenizer)
        rejected_length = _completion_token_length(row["rejected"], tokenizer)
        prompt_lengths.append(prompt_length)
        chosen_lengths.append(chosen_length)
        rejected_lengths.append(rejected_length)
        chosen_full_lengths.append(prompt_length + chosen_length)
        rejected_full_lengths.append(prompt_length + rejected_length)

    prompt_limit = int(training_config["max_prompt_length"])
    completion_limit = int(training_config["max_completion_length"])
    full_limit = int(training_config["max_length"])
    summary = {
        "prompt": _length_summary(prompt_lengths, prompt_limit),
        "chosen_completion_with_eos": _length_summary(chosen_lengths, completion_limit),
        "rejected_completion_with_eos": _length_summary(rejected_lengths, completion_limit),
        "prompt_plus_chosen": _length_summary(chosen_full_lengths, full_limit),
        "prompt_plus_rejected": _length_summary(rejected_full_lengths, full_limit),
    }
    violations = sum(int(stats["over_limit"]) for stats in summary.values())
    zero_completions = sum(length <= 1 for length in chosen_lengths + rejected_lengths)
    if zero_completions:
        raise ValueError(f"{split_name} has {zero_completions} zero-token completions")
    if violations:
        details = ", ".join(
            f"{name}={stats['over_limit']}" for name, stats in summary.items() if stats["over_limit"]
        )
        raise ValueError(f"{split_name} exceeds configured token budgets: {details}")
    logger.info("%s token lengths: %s", split_name, json.dumps(summary, sort_keys=True))
    return {"rows": len(dataset), "violations": 0, **summary}


def resolve_arm(config: dict[str, Any], arm: str | None) -> tuple[str, dict[str, Any]]:
    """Resolve one objective and reject config changes that alter its fixed definition."""
    selected = arm or config.get("arm") or "dpo"
    if selected not in _ARM_CONTRACT:
        raise ValueError(f"Unknown arm {selected!r}; expected one of {sorted(_ARM_CONTRACT)}")
    configured_arms = config.get("arms")
    if not isinstance(configured_arms, dict):
        raise ValueError("Config must define the fixed arms mapping")
    configured = configured_arms.get(selected)
    expected = _ARM_CONTRACT[selected]
    if configured != expected:
        raise ValueError(
            f"Config for arm {selected!r} contradicts the experiment contract: "
            f"expected {expected}, got {configured}"
        )
    return selected, dict(expected)


def _derive_batch(training_config: dict[str, Any], world_size: int) -> tuple[int, int]:
    per_device = int(training_config["per_device_train_batch_size"])
    target = int(training_config["target_global_batch_size"])
    denominator = world_size * per_device
    if denominator <= 0 or target <= 0 or target % denominator:
        raise ValueError(
            "training.target_global_batch_size must be a positive multiple of "
            f"world_size * per_device_train_batch_size ({world_size} * {per_device})"
        )
    accumulation = target // denominator
    if accumulation < 1:
        raise ValueError("Derived gradient accumulation must be at least 1")
    return accumulation, per_device * world_size * accumulation


def _load_tokenizer(model_name: str, max_length: int):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_length)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _prepare_data(config: dict[str, Any], tokenizer: PreTrainedTokenizerBase):
    data_config = config.get("data", {})
    if not data_config.get("val_path"):
        raise ValueError(
            "data.val_path is required; a row-level fallback would leak repeated questions"
        )
    train_path = Path(data_config["train_path"])
    val_path = Path(data_config["val_path"])
    train_pairs = load_pairs(train_path)
    val_pairs = load_pairs(val_path)
    split_summary = validate_splits(train_pairs, val_pairs)
    train_dataset = pairs_to_dataset(train_pairs, tokenizer)
    val_dataset = pairs_to_dataset(val_pairs, tokenizer)
    token_summary = {
        "train": measure_token_lengths(
            train_dataset, tokenizer, config["training"], split_name="train"
        ),
        "validation": measure_token_lengths(
            val_dataset, tokenizer, config["training"], split_name="validation"
        ),
    }
    return train_dataset, val_dataset, split_summary, token_summary, train_path, val_path


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_manifest(
    *,
    config: dict[str, Any],
    arm: str,
    arm_config: dict[str, Any],
    seed: int,
    package_versions: dict[str, str],
    world_size: int,
    accumulation: int,
    effective_batch: int,
    split_summary: dict[str, Any],
    token_summary: dict[str, Any],
    train_path: Path,
    val_path: Path,
) -> dict[str, Any]:
    return {
        "config": config,
        "arm": arm,
        "arm_config": arm_config,
        "seed": seed,
        "package_versions": package_versions,
        "world_size": world_size,
        "gradient_accumulation_steps": accumulation,
        "effective_global_batch_size": effective_batch,
        "splits": split_summary,
        "token_lengths": token_summary,
        "data_sha256": {"train": _sha256(train_path), "validation": _sha256(val_path)},
    }


def run_preflight(config: dict[str, Any], *, arm: str | None = None, seed: int | None = None):
    """Validate environment, objective, data, and token budgets without loading the model."""
    package_versions = verify_training_environment()
    selected_arm, arm_config = resolve_arm(config, arm)
    selected_seed = int(config.get("seed", 42) if seed is None else seed)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    accumulation, effective_batch = _derive_batch(config["training"], world_size)
    tokenizer = _load_tokenizer(config["model"]["name"], config["model"]["max_seq_length"])
    _, _, split_summary, token_summary, train_path, val_path = _prepare_data(config, tokenizer)
    report = _run_manifest(
        config=config,
        arm=selected_arm,
        arm_config=arm_config,
        seed=selected_seed,
        package_versions=package_versions,
        world_size=world_size,
        accumulation=accumulation,
        effective_batch=effective_batch,
        split_summary=split_summary,
        token_summary=token_summary,
        train_path=train_path,
        val_path=val_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report


def train(
    config: dict[str, Any],
    *,
    arm: str | None = None,
    seed: int | None = None,
    max_steps: int | None = None,
    resume_from_checkpoint: str | None = None,
) -> Path:
    """Train one preference arm and return its run directory."""
    package_versions = _verify_training_packages()

    # Unsloth must patch Transformers, TRL, and PEFT before any of them are imported.
    from unsloth import FastLanguageModel  # noqa: I001

    import torch
    from trl import DPOConfig, DPOTrainer

    _verify_dpo_config_interface(DPOConfig)
    logger.info(
        "Training packages: %s",
        ", ".join(f"{k}={v}" for k, v in package_versions.items()),
    )
    selected_arm, arm_config = resolve_arm(config, arm)
    selected_seed = int(config.get("seed", 42) if seed is None else seed)
    training_config = config["training"]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("DPO training requires a CUDA GPU; use --preflight-only on CPU")
    torch.cuda.set_device(local_rank)
    _seed_everything(selected_seed)

    accumulation, effective_batch = _derive_batch(training_config, world_size)
    run_dir = Path(training_config["run_root"]) / selected_arm / f"seed-{selected_seed}"
    if _is_rank_zero() and run_dir.exists() and any(run_dir.iterdir()) and not resume_from_checkpoint:
        raise FileExistsError(
            f"Run directory is not empty: {run_dir}. Pass --resume-from-checkpoint explicitly."
        )
    if resume_from_checkpoint and not Path(resume_from_checkpoint).is_dir():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_from_checkpoint}")

    model_config = config["model"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config["name"],
        max_seq_length=int(model_config["max_seq_length"]),
        dtype=None,
        load_in_4bit=bool(model_config["load_in_4bit"]),
        # One replica per rank. Unsloth's default planner would shard a single-process run
        # across every visible GPU, which DDP replication and generation both reject.
        device_map={"": local_rank},
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, val_dataset, split_summary, token_summary, train_path, val_path = _prepare_data(
        config, tokenizer
    )
    lora_config = config["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora_config["r"]),
        lora_alpha=int(lora_config["alpha"]),
        lora_dropout=float(lora_config["dropout"]),
        target_modules=list(lora_config["target_modules"]),
        bias=lora_config["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=selected_seed,
    )

    manifest = _run_manifest(
        config=config,
        arm=selected_arm,
        arm_config=arm_config,
        seed=selected_seed,
        package_versions=package_versions,
        world_size=world_size,
        accumulation=accumulation,
        effective_batch=effective_batch,
        split_summary=split_summary,
        token_summary=token_summary,
        train_path=train_path,
        val_path=val_path,
    )
    if _is_rank_zero():
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "run_manifest.json", manifest)

    configured_max_steps = -1 if max_steps is None else int(max_steps)
    if configured_max_steps == 0 or configured_max_steps < -1:
        raise ValueError("max_steps must be a positive integer when supplied")
    args = DPOConfig(
        output_dir=str(run_dir),
        num_train_epochs=float(training_config["epochs"]),
        max_steps=configured_max_steps,
        per_device_train_batch_size=int(training_config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=accumulation,
        learning_rate=float(training_config["learning_rate"]),
        lr_scheduler_type=training_config["lr_scheduler_type"],
        warmup_ratio=float(training_config["warmup_ratio"]),
        max_grad_norm=float(training_config["max_grad_norm"]),
        optim=training_config["optim"],
        fp16=bool(training_config["fp16"]),
        bf16=False,
        gradient_checkpointing=bool(training_config["gradient_checkpointing"]),
        beta=float(training_config["beta"]),
        loss_type=arm_config["loss_type"],
        use_weighting=arm_config["use_weighting"],
        label_smoothing=float(arm_config["label_smoothing"]),
        rpo_alpha=float(arm_config["rpo_alpha"]),
        max_prompt_length=int(training_config["max_prompt_length"]),
        max_completion_length=int(training_config["max_completion_length"]),
        max_length=int(training_config["max_length"]),
        truncation_mode=training_config["truncation_mode"],
        precompute_ref_log_probs=bool(training_config["precompute_ref_log_probs"]),
        eval_strategy=training_config["eval_strategy"],
        save_strategy=training_config["save_strategy"],
        save_total_limit=int(training_config["save_total_limit"]),
        load_best_model_at_end=bool(training_config["load_best_model_at_end"]),
        metric_for_best_model=training_config["metric_for_best_model"],
        greater_is_better=bool(training_config["greater_is_better"]),
        logging_steps=int(training_config["logging_steps"]),
        report_to=training_config["report_to"],
        ddp_find_unused_parameters=bool(training_config["ddp_find_unused_parameters"]),
        seed=selected_seed,
        data_seed=selected_seed,
    )
    logger.info(
        "Arm=%s loss=%s weighting=%s smoothing=%s rpo_alpha=%s world=%d "
        "accumulation=%d global_batch=%d",
        selected_arm,
        arm_config["loss_type"],
        arm_config["use_weighting"],
        arm_config["label_smoothing"],
        arm_config["rpo_alpha"],
        world_size,
        accumulation,
        effective_batch,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    best_dir = run_dir / "dpo_best"
    trainer.save_model(str(best_dir))
    if _is_rank_zero():
        tokenizer.save_pretrained(best_dir)
        metrics = {
            "train": train_result.metrics,
            "log_history": trainer.state.log_history,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
        }
        _write_json(run_dir / "training_metrics.json", metrics)
        manifest["result"] = {
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "promoted_adapter": str(best_dir),
        }
        _write_json(run_dir / "run_manifest.json", manifest)
        logger.info("Training complete: %s", best_dir)
    return run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DPO-family query rewriter")
    parser.add_argument("--config", type=Path, required=True, help="Path to train_dpo YAML")
    parser.add_argument("--arm", choices=sorted(_ARM_CONTRACT))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.preflight_only:
        run_preflight(config, arm=args.arm, seed=args.seed)
        return
    train(
        config,
        arm=args.arm,
        seed=args.seed,
        max_steps=args.max_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()
