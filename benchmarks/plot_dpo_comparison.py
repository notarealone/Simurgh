"""Compare two DPO screening runs and emit the figures for a results document.

Every number is recomputed from the committed run artifacts (`run_manifest.json`,
`training_metrics.json`, `comparisons/seed-42/*`), so the figures cannot drift from the
tournament they describe.

Cross-run tournament scores are not directly comparable: round-robin is relative to the
field, and the two runs judged different validation prompt sets. This script therefore
anchors every cross-run claim to `base_qwen`, whose greedy rewrites are byte-identical
across the two runs on almost every shared prompt, and restricts to the shared
`(question_ref, persona_id)` keys.

Usage:
    python benchmarks/plot_dpo_comparison.py \
        --old models/dpo/oldData-dpo-seed-42 --new models/dpo/dpo-seed-42 \
        --out docs/results/figures/dpo-v4-vs-v3
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ("dpo", "wpo", "robust_dpo")
CANDIDATES = ("base_qwen", "dpo", "wpo", "robust_dpo", "grok")
PERSONAS = ("crammer", "scholar", "steady")
BETA_KEY = "beta"
BOOTSTRAP_SAMPLES = 5000
PERMUTATIONS = 20000
CI_SEED = 42
COLORS = {"old": "#9aa5b1", "new": "#2f6f9f", "grok": "#b5651d", "base_qwen": "#444444"}
CLOZE = re.compile(r"\.{3,}|_{3,}|\u2026")
Key = tuple[str, str]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


class Run:
    """One screening run: three arms, one tournament, one pair-file generation."""

    def __init__(self, root: Path, label: str) -> None:
        self.root = root
        self.label = label
        comparison = root / "comparisons" / "seed-42"
        self.summary = json.loads((comparison / "summary.json").read_text(encoding="utf-8"))
        self.judgments = read_jsonl(comparison / "judgments.jsonl")
        self.outputs = {
            candidate: {
                (row["question_ref"], row["persona_id"]): row["rewrite"]
                for row in read_jsonl(comparison / "outputs" / f"{candidate}.jsonl")
            }
            for candidate in CANDIDATES
        }
        self.manifests = {
            arm: json.loads(
                (root / "runs" / arm / "seed-42" / "run_manifest.json").read_text(encoding="utf-8")
            )
            for arm in ARMS
        }
        self.metrics = {
            arm: json.loads(
                (root / "runs" / arm / "seed-42" / "training_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            for arm in ARMS
        }
        self.beta = self.manifests["dpo"]["config"]["training"][BETA_KEY]

    @property
    def keys(self) -> set[Key]:
        return set(self.outputs["base_qwen"])

    def round_robin(self) -> dict[str, float]:
        return self.summary["round_robin"]

    def eval_points(self, arm: str) -> list[dict[str, float]]:
        return [entry for entry in self.metrics[arm]["log_history"] if "eval_loss" in entry]

    def promoted_epoch(self, arm: str) -> float:
        """Epoch of the checkpoint the trainer promoted, read from the run manifest."""
        checkpoint = Path(self.manifests[arm]["result"]["best_model_checkpoint"]).name
        state = json.loads(
            (self.root / "runs" / arm / "seed-42" / checkpoint / "trainer_state.json").read_text(
                encoding="utf-8"
            )
        )
        return round(state["epoch"], 6)

    def promoted_eval(self, arm: str) -> dict[str, float]:
        target = self.promoted_epoch(arm)
        return min(self.eval_points(arm), key=lambda entry: abs(entry["epoch"] - target))

    def pairwise(self, arm: str, opponent: str, keys: set[Key]) -> dict[Key, float]:
        """Per-prompt score of `arm` against `opponent`; 1.0 win, 0.5 tie, 0.0 loss."""
        scores: dict[Key, float] = {}
        for row in self.judgments:
            key = (row["question_ref"], row["persona_id"])
            if key not in keys or {row["model_left"], row["model_right"]} != {arm, opponent}:
                continue
            winner = (
                row["candidate_a"]
                if row["winner"] == "A"
                else (row["candidate_b"] if row["winner"] == "B" else None)
            )
            scores[key] = 0.5 if winner is None else float(winner == arm)
        return scores


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def wilson_free_ci(scores: list[float], rng: random.Random) -> tuple[float, float]:
    """Bootstrap CI over per-prompt scores; ties are honest 0.5s, so resample directly."""
    draws = sorted(mean([rng.choice(scores) for _ in scores]) for _ in range(BOOTSTRAP_SAMPLES))
    return draws[int(0.025 * BOOTSTRAP_SAMPLES)], draws[int(0.975 * BOOTSTRAP_SAMPLES) - 1]


def paired_delta(
    old: dict[Key, float], new: dict[Key, float], rng: random.Random
) -> dict[str, Any]:
    """Paired per-prompt delta with a sign-flip permutation p-value."""
    keys = sorted(set(old) & set(new))
    diffs = [new[key] - old[key] for key in keys]
    observed = mean(diffs)
    extreme = sum(
        abs(mean([diff if rng.random() < 0.5 else -diff for diff in diffs]))
        >= abs(observed) - 1e-12
        for _ in range(PERMUTATIONS)
    )
    draws = sorted(mean([rng.choice(diffs) for _ in diffs]) for _ in range(BOOTSTRAP_SAMPLES))
    return {
        "n": len(keys),
        "delta": observed,
        "ci": (draws[int(0.025 * BOOTSTRAP_SAMPLES)], draws[int(0.975 * BOOTSTRAP_SAMPLES) - 1]),
        "p": (extreme + 1) / (PERMUTATIONS + 1),
    }


def jaccard(left: str, right: str) -> float:
    a, b = set(left.split()), set(right.split())
    return len(a & b) / max(1, len(a | b))


def repeated_trigram(text: str) -> bool:
    tokens = text.split()
    grams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    return bool(grams) and len(set(grams)) < len(grams)


def persona_divergence(rewrites: dict[Key, str]) -> tuple[int, float]:
    """How differently a candidate rewrites the same question for different personas."""
    by_question: dict[str, dict[str, str]] = defaultdict(dict)
    for (question_ref, persona_id), text in rewrites.items():
        by_question[question_ref][persona_id] = text
    groups = [group for group in by_question.values() if len(group) >= 2]
    identical = sum(1 for group in groups if len(set(group.values())) == 1)
    overlaps = [
        jaccard(a, b)
        for group in groups
        for index, a in enumerate(group.values())
        for b in list(group.values())[index + 1 :]
    ]
    return identical, mean(overlaps)


def figure_win_rate(old: Run, new: Run, shared: set[Key], out_dir: Path) -> dict[str, Any]:
    rng = random.Random(CI_SEED)
    stats: dict[str, Any] = {}
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    width = 0.36
    for index, arm in enumerate(ARMS):
        entry = {}
        for tag, run in (("old", old), ("new", new)):
            scores = run.pairwise(arm, "base_qwen", shared)
            values = list(scores.values())
            low, high = wilson_free_ci(values, rng)
            entry[tag] = {"scores": scores, "mean": mean(values), "ci": (low, high)}
            offset = -width / 2 if tag == "old" else width / 2
            ax.bar(
                index + offset,
                entry[tag]["mean"],
                width,
                color=COLORS[tag],
                label={"old": "v3 data + old trainer", "new": "v4 data + new trainer"}[tag]
                if index == 0
                else None,
            )
            ax.errorbar(
                index + offset,
                entry[tag]["mean"],
                yerr=[[entry[tag]["mean"] - low], [high - entry[tag]["mean"]]],
                fmt="none",
                ecolor="black",
                capsize=3,
                lw=1,
            )
        entry["paired"] = paired_delta(entry["old"]["scores"], entry["new"]["scores"], rng)
        stats[arm] = entry
        ax.annotate(
            f"{entry['paired']['delta']:+.3f}\np={entry['paired']['p']:.3f}",
            (index, max(entry["old"]["mean"], entry["new"]["mean"]) + 0.055),
            ha="center",
            fontsize=8,
        )
    ax.axhline(0.5, color="black", ls="--", lw=1)
    ax.text(-0.42, 0.512, "parity with base_qwen", fontsize=7, ha="left")
    ax.set_xticks(range(len(ARMS)), ARMS)
    ax.set_ylim(0.0, 0.78)
    ax.set_ylabel("win rate vs. base_qwen")
    ax.set_title(f"Anchored win rate against the untrained policy ({len(shared)} shared prompts)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "win_rate_vs_base.png", dpi=160)
    plt.close(fig)
    return stats


def figure_displacement(old: Run, new: Run, out_dir: Path) -> dict[str, Any]:
    """Sequence log-probability shift against the frozen reference, in nats."""
    stats: dict[str, Any] = {}
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    positions, labels = [], []
    for index, arm in enumerate(ARMS):
        for offset, (tag, run) in zip((-0.19, 0.19), (("old", old), ("new", new)), strict=True):
            entry = run.promoted_eval(arm)
            chosen = entry["eval_rewards/chosen"] / run.beta
            rejected = entry["eval_rewards/rejected"] / run.beta
            stats[f"{tag}:{arm}"] = {
                "epoch": entry["epoch"],
                "dlogp_chosen": chosen,
                "dlogp_rejected": rejected,
                "margin": entry["eval_rewards/margins"],
                "accuracy": entry["eval_rewards/accuracies"],
            }
            ax.bar(index + offset - 0.085, chosen, 0.17, color=COLORS[tag])
            ax.bar(
                index + offset + 0.085, rejected, 0.17, color=COLORS[tag], alpha=0.45, hatch="//"
            )
            positions.append(index + offset)
            labels.append(f"{tag}\nep{entry['epoch']:.0f}")
    ax.axhline(0.0, color="black", lw=1)
    ax.set_xticks(positions, labels, fontsize=7)
    for index, arm in enumerate(ARMS):
        ax.annotate(arm, (index, -34), ha="center", fontsize=9)
    ax.set_ylabel(r"$\Delta \log p$ vs. frozen reference (nats)")
    ax.set_title("Solid = chosen rewrite, hatched = rejected rewrite (promoted checkpoint)")
    ax.set_ylim(-40, 52)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["old"]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["new"]),
    ]
    ax.legend(
        handles, ["v3 data + old trainer", "v4 data + new trainer"], loc="upper left", fontsize=8
    )
    fig.tight_layout()
    fig.savefig(out_dir / "likelihood_displacement.png", dpi=160)
    plt.close(fig)
    return stats


def figure_style_space(old: Run, new: Run, shared: set[Key], out_dir: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for tag, run in (("old", old), ("new", new)):
        teacher, base = run.outputs["grok"], run.outputs["base_qwen"]
        for candidate in ("base_qwen", *ARMS):
            texts = run.outputs[candidate]
            point = (
                mean([jaccard(texts[key], base[key]) for key in shared]),
                mean([jaccard(texts[key], teacher[key]) for key in shared]),
            )
            stats[f"{tag}:{candidate}"] = {"to_base": point[0], "to_grok": point[1]}
            if candidate == "base_qwen" and tag == "new":
                continue
            marker = "s" if candidate == "base_qwen" else "o"
            ax.scatter(*point, color=COLORS[tag], marker=marker, s=70, zorder=3)
            ax.annotate(candidate, (point[0] + 0.008, point[1] - 0.006), fontsize=8)
    for arm in ARMS:
        start, end = stats[f"old:{arm}"], stats[f"new:{arm}"]
        ax.annotate(
            "",
            xy=(end["to_base"], end["to_grok"]),
            xytext=(start["to_base"], start["to_grok"]),
            arrowprops={"arrowstyle": "->", "color": "#777777", "lw": 1},
        )
    ax.margins(0.09)
    ax.set_xlabel("token overlap with base_qwen (Jaccard)")
    ax.set_ylabel("token overlap with grok teacher (Jaccard)")
    ax.set_title("Old training drifted off base; new training moved toward the teacher")
    handles = [
        plt.Line2D([], [], color=COLORS["old"], marker="o", ls="", label="v3 data + old trainer"),
        plt.Line2D([], [], color=COLORS["new"], marker="o", ls="", label="v4 data + new trainer"),
        plt.Line2D([], [], color=COLORS["old"], marker="s", ls="", label="base_qwen (both runs)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "style_space.png", dpi=160)
    plt.close(fig)
    return stats


def figure_per_persona(old: Run, new: Run, shared: set[Key], out_dir: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    fig, axes = plt.subplots(1, len(ARMS), figsize=(9.6, 3.4), sharey=True)
    for ax, arm in zip(axes, ARMS, strict=True):
        for offset, (tag, run) in zip((-0.19, 0.19), (("old", old), ("new", new)), strict=True):
            scores = run.pairwise(arm, "base_qwen", shared)
            grouped: dict[str, list[float]] = defaultdict(list)
            for (_, persona_id), value in scores.items():
                grouped[persona_id].append(value)
            for index, persona in enumerate(PERSONAS):
                value = mean(grouped[persona])
                stats[f"{tag}:{arm}:{persona}"] = {"score": value, "n": len(grouped[persona])}
                ax.bar(index + offset, value, 0.36, color=COLORS[tag])
        ax.axhline(0.5, color="black", ls="--", lw=1)
        ax.set_xticks(range(len(PERSONAS)), PERSONAS, fontsize=8)
        ax.set_title(arm, fontsize=10)
    axes[0].set_ylabel("win rate vs. base_qwen")
    axes[0].set_ylim(0.0, 0.75)
    axes[-1].legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=COLORS["old"]),
            plt.Rectangle((0, 0), 1, 1, color=COLORS["new"]),
        ],
        labels=["v3 + old", "v4 + new"],
        fontsize=7,
        loc="upper right",
    )
    fig.suptitle("Per-persona anchored win rate", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "per_persona_win_rate.png", dpi=160)
    plt.close(fig)
    return stats


def figure_eval_trajectory(old: Run, new: Run, out_dir: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for ax, (tag, run, title) in zip(
        axes,
        (
            ("old", old, "v3 data + old trainer (select on eval_loss)"),
            ("new", new, "v4 data + new trainer (select on accuracy)"),
        ),
        strict=True,
    ):
        for arm, style in zip(ARMS, ("-o", "-s", "-^"), strict=True):
            points = run.eval_points(arm)
            xs = [entry["epoch"] for entry in points]
            ys = [entry["eval_rewards/accuracies"] for entry in points]
            stats[f"{tag}:{arm}"] = dict(zip(xs, ys, strict=True))
            ax.plot(xs, ys, style, label=arm, lw=1.4, ms=5)
            promoted = run.promoted_epoch(arm)
            selected = min(points, key=lambda entry: abs(entry["epoch"] - promoted))
            ax.scatter(
                [selected["epoch"]],
                [selected["eval_rewards/accuracies"]],
                s=220,
                facecolors="none",
                edgecolors="red",
                lw=1.6,
                zorder=4,
            )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("epoch")
        ax.set_xticks([1, 2, 3])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("validation preference accuracy")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].annotate(
        "red ring = promoted checkpoint\n(v4 uses a different validation set,\nso the vertical offset is not like-for-like)",
        (1.05, 0.60),
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "eval_accuracy_trajectory.png", dpi=160)
    plt.close(fig)
    return stats


def figure_text_shape(old: Run, new: Run, shared: set[Key], out_dir: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    queries = {
        (row["question_ref"], row["persona_id"]): row["query"]
        for row in read_jsonl(new.root / "comparisons" / "seed-42" / "outputs" / "base_qwen.jsonl")
    }
    cloze_keys = [key for key in shared if CLOZE.search(queries[key])]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for tag, run in (("old", old), ("new", new)):
        for candidate in ("base_qwen", *ARMS, "grok"):
            texts = [run.outputs[candidate][key] for key in sorted(shared)]
            lengths = sorted(len(text.split()) for text in texts)
            identical, overlap = persona_divergence(
                {key: run.outputs[candidate][key] for key in shared}
            )
            stats[f"{tag}:{candidate}"] = {
                "words_median": statistics.median(lengths),
                "words_mean": mean([float(value) for value in lengths]),
                "repeated_trigram_rate": mean([float(repeated_trigram(text)) for text in texts]),
                "cloze_marker_kept": mean(
                    [float(bool(CLOZE.search(run.outputs[candidate][key]))) for key in cloze_keys]
                ),
                "identical_across_personas": identical,
                "persona_overlap": overlap,
            }
            if candidate in ("base_qwen", "grok") and tag == "old":
                continue
            color = COLORS.get(candidate, COLORS[tag])
            ls = {"old": ":", "new": "-"}[tag]
            label = candidate if candidate in ("base_qwen", "grok") else f"{candidate} ({tag})"
            axes[0].plot(
                lengths,
                [index / len(lengths) for index in range(1, len(lengths) + 1)],
                ls=ls,
                color=color,
                lw=1.4,
                label=label,
            )
    axes[0].set_xlabel("rewrite length (whitespace words)")
    axes[0].set_ylabel("empirical CDF")
    axes[0].set_title("Length distribution", fontsize=10)
    axes[0].legend(fontsize=6.5, loc="lower right")
    axes[0].grid(alpha=0.25)

    metrics = ("repeated_trigram_rate", "cloze_marker_kept")
    titles = ("repeated trigram", f"cloze marker kept (n={len(cloze_keys)})")
    xs = range(len(ARMS) + 2)
    order = ("base_qwen", *ARMS, "grok")
    for metric_index, (metric, title) in enumerate(zip(metrics, titles, strict=True)):
        for offset, tag in zip((-0.19, 0.19), ("old", "new"), strict=True):
            values = [stats[f"{tag}:{candidate}"][metric] for candidate in order]
            axes[1].bar(
                [x + offset + metric_index * (len(order) + 1) for x in xs],
                values,
                0.36,
                color=COLORS[tag],
            )
        axes[1].annotate(
            title, (metric_index * (len(order) + 1) + 2, 0.92), ha="center", fontsize=8
        )
    axes[1].set_xticks(
        [x + metric_index * (len(order) + 1) for metric_index in range(2) for x in xs],
        list(order) * 2,
        rotation=60,
        fontsize=6.5,
        ha="right",
    )
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("rate")
    axes[1].set_title("Degeneracy and meaning preservation", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "text_shape.png", dpi=160)
    plt.close(fig)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=Path("models/dpo/oldData-dpo-seed-42"))
    parser.add_argument("--new", type=Path, default=Path("models/dpo/dpo-seed-42"))
    parser.add_argument("--out", type=Path, default=Path("docs/results/figures/dpo-v4-vs-v3"))
    args = parser.parse_args()

    old, new = Run(args.old, "old"), Run(args.new, "new")
    shared = old.keys & new.keys
    identical_base = sum(
        old.outputs["base_qwen"][key] == new.outputs["base_qwen"][key] for key in shared
    )
    identical_grok = sum(old.outputs["grok"][key] == new.outputs["grok"][key] for key in shared)
    args.out.mkdir(parents=True, exist_ok=True)

    report = {
        "prompts": {
            "old": len(old.keys),
            "new": len(new.keys),
            "shared": len(shared),
            "base_qwen_identical_on_shared": identical_base,
            "grok_identical_on_shared": identical_grok,
        },
        "round_robin": {"old": old.round_robin(), "new": new.round_robin()},
        "win_rate_vs_base": figure_win_rate(old, new, shared, args.out),
        "displacement": figure_displacement(old, new, args.out),
        "style_space": figure_style_space(old, new, shared, args.out),
        "per_persona": figure_per_persona(old, new, shared, args.out),
        "eval_trajectory": figure_eval_trajectory(old, new, args.out),
        "text_shape": figure_text_shape(old, new, shared, args.out),
    }
    for entry in report["win_rate_vs_base"].values():
        entry["old"].pop("scores")
        entry["new"].pop("scores")
    (args.out / "stats.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
