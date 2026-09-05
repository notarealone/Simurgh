"""Figures for the Stage 1 ROPG run comparison.

Every number is recomputed from the four `training_log.json` files under `models/ropg/`,
reusing the paired bootstrap, sign-flip permutation test, and Holm correction from
`compare_runs.py`, so the figures cannot drift from the reports that script emits.

The four runs share one validation set of 276 `(question, persona)` rows and identical
epoch-0 per-query vectors, which is what makes the paired tests and the baseline-relative
panels legitimate. The script asserts that parity instead of assuming it.

Usage:
    python benchmarks/plot_ropg_comparison.py \
        --runs models/ropg --out docs/results/figures/ropg-runs-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_runs import holm, paired_stats

# The checkpoint directory name differs per run because the runs were launched from
# separate Kaggle sessions; the selected arm (B) kept the default name.
LOG_PATHS = {
    "A": "ropg_kd_runA_model/ropg_kd_runA_checkpoints/training_log.json",
    "B": "ropg_kd_runB_model/ropg_kd_checkpoints/training_log.json",
    "C": "ropg_kd_runC_model/ropg_kd_runC_checkpoints/training_log.json",
    "D": "ropg_kd_runD_model/ropg_kd_runD_checkpoints/training_log.json",
}
RUNS = ("A", "B", "C", "D")
PERSONAS = ("crammer", "scholar", "steady")
HEADLINE = ("recall@5", "ndcg@5", "mrr", "hit@5", "hit@1", "judged@5")
N_BOOT = 10000
SEED = 42
COLORS = {"A": "#9aa5b1", "B": "#2f6f9f", "C": "#4f9d69", "D": "#b5651d"}
BASELINE_COLOR = "#444444"
SELECTOR = "recall@5"  # src/rl/ropg_kd.py::is_better


class Run:
    """One ROPG arm: its config, its epoch metrics, and its selected checkpoint."""

    def __init__(self, root: Path, label: str) -> None:
        self.label = label
        self.path = root / LOG_PATHS[label]
        log = json.loads(self.path.read_text(encoding="utf-8"))
        self.config = log["config"]
        self.seed = log["seed"]
        self.best_epoch = log["best_epoch"]
        self.epochs = {entry["epoch"]: entry for entry in log["epoch_metrics"]}

    @property
    def baseline(self) -> dict[str, Any]:
        return self.epochs[0]

    @property
    def best(self) -> dict[str, Any]:
        return self.epochs[self.best_epoch]

    @property
    def anchor(self) -> str:
        return str(self.config.get("anchor", {}).get("mode", "none"))

    @property
    def data(self) -> str:
        return Path(str(self.config["data"]["train_data"])).name

    def swap_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        return {"per_query": entry["persona_swap"]["per_query"]}


def overall(entry: dict[str, Any], metric: str) -> float:
    """Epoch metrics are `{overall, crammer, scholar, steady}` dicts; losses are scalars."""
    value = entry[metric]
    return float(value["overall"] if isinstance(value, dict) else value)


def per_query(entry: dict[str, Any], metric: str) -> np.ndarray:
    return np.asarray(entry["per_query"][metric], dtype=float)


def compare(
    entry_a: dict[str, Any],
    entry_b: dict[str, Any],
    metrics: list[str],
    mask: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Paired B-minus-A stats per metric, Holm-corrected across the metric family."""
    rows: dict[str, dict[str, float]] = {}
    for metric in metrics:
        # One generator per metric, seeded identically, matching compare_runs.py so the
        # figures and the Markdown reports print the same intervals.
        rng = np.random.default_rng(SEED)
        before, after = per_query(entry_a, metric), per_query(entry_b, metric)
        if mask is not None:
            before, after = before[mask], after[mask]
        rows[metric] = paired_stats(before, after, N_BOOT, rng)
    for metric, adjusted in zip(rows, holm([r["p"] for r in rows.values()]), strict=True):
        rows[metric]["p_holm"] = adjusted
    return rows


def finish(fig: plt.Figure, out: Path, name: str) -> Path:
    path = out / name
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def annotate_significance(ax: plt.Axes, x: float, y: float, p_holm: float) -> None:
    if p_holm < 0.05:
        ax.annotate(
            "*", (x, y), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=13
        )


def plot_metric_ladder(runs: dict[str, Run], out: Path) -> Path:
    """nDCG@K and Hit@K against K: where each arm's advantage actually lives."""
    ks = [1, 2, 3, 4, 5]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, family in zip(axes, ("ndcg", "hit"), strict=True):
        ax.plot(
            ks,
            [overall(runs["A"].baseline, f"{family}@{k}") for k in ks],
            marker="o",
            color=BASELINE_COLOR,
            linestyle="--",
            label="baseline (epoch 0)",
        )
        for label in RUNS:
            ax.plot(
                ks,
                [overall(runs[label].best, f"{family}@{k}") for k in ks],
                marker="o",
                color=COLORS[label],
                linewidth=2.2 if label == "B" else 1.4,
                label=f"Run {label}",
            )
        ax.set_xticks(ks)
        ax.set_xlabel("K")
        ax.set_ylabel(f"{family}@K".replace("ndcg", "nDCG").replace("hit", "Hit"))
        ax.grid(alpha=0.3)
    axes[0].set_title("Graded ranking quality (nDCG@K)")
    axes[1].set_title("Any-relevant coverage (Hit@K)")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Best checkpoint per arm, 276 shared validation queries — "
        "Run C leads at K≤4, Run B at K=5",
        fontsize=11,
    )
    return finish(fig, out, "metric_ladder.png")


def plot_headline_deltas(runs: dict[str, Run], out: Path) -> tuple[Path, dict[str, Any]]:
    """Left: each arm minus baseline. Right: Run B minus each rival. CIs are paired."""
    metrics = list(HEADLINE)
    vs_base = {label: compare(runs[label].baseline, runs[label].best, metrics) for label in RUNS}
    b_vs = {
        label: compare(runs[label].best, runs["B"].best, metrics) for label in RUNS if label != "B"
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    offsets = np.linspace(-0.26, 0.26, 4)
    ys = np.arange(len(metrics))
    for offset, label in zip(offsets, RUNS, strict=True):
        stats = [vs_base[label][m] for m in metrics]
        axes[0].errorbar(
            [s["delta"] for s in stats],
            ys + offset,
            xerr=[
                [s["delta"] - s["lo"] for s in stats],
                [s["hi"] - s["delta"] for s in stats],
            ],
            fmt="o",
            markersize=5,
            color=COLORS[label],
            elinewidth=1.4,
            capsize=2,
            label=f"Run {label}",
        )
    axes[0].set_title("Trained checkpoint minus untrained baseline")
    axes[0].legend(fontsize=8, loc="upper left")

    for offset, label in zip(
        np.linspace(-0.2, 0.2, 3), [r for r in RUNS if r != "B"], strict=True
    ):
        stats = [b_vs[label][m] for m in metrics]
        axes[1].errorbar(
            [s["delta"] for s in stats],
            ys + offset,
            xerr=[
                [s["delta"] - s["lo"] for s in stats],
                [s["hi"] - s["delta"] for s in stats],
            ],
            fmt="o",
            markersize=5,
            color=COLORS[label],
            elinewidth=1.4,
            capsize=2,
            label=f"B − {label}",
        )
        for metric, y, stat in zip(metrics, ys + offset, stats, strict=True):
            if stat["p_holm"] < 0.05:
                axes[1].annotate(
                    "*",
                    (stat["hi"], y),
                    textcoords="offset points",
                    xytext=(4, -4),
                    fontsize=12,
                    color=COLORS[label],
                )
            del metric
    axes[1].set_title("Run B minus each rival arm  (* Holm p < 0.05)")
    axes[1].legend(fontsize=8, loc="upper left")

    for ax in axes:
        ax.axvline(0.0, color="#888888", linewidth=1)
        ax.set_yticks(ys)
        ax.set_yticklabels(metrics)
        ax.invert_yaxis()
        ax.set_xlabel("paired difference (95% bootstrap CI)")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(
        "Recall@5 is the only metric that separates the arms; judged@5 moves with it",
        fontsize=11,
    )
    path = finish(fig, out, "headline_deltas.png")
    return path, {"vs_baseline": vs_base, "b_minus": b_vs}


def plot_epoch_trajectories(runs: dict[str, Run], out: Path) -> Path:
    """Loss falls for three epochs while retrieval peaks at epoch 1 or 2."""
    panels = [
        ("train_loss", "Training loss", False),
        ("val_loss", "Validation loss", False),
        (SELECTOR, "Recall@5 (checkpoint selector)", True),
        ("ndcg@5", "nDCG@5", True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, (metric, title, retrieval) in zip(axes, panels, strict=True):
        for label in RUNS:
            run = runs[label]
            xs, values = [], []
            for epoch in sorted(run.epochs):
                entry = run.epochs[epoch]
                if metric not in entry:
                    continue
                xs.append(epoch)
                values.append(overall(entry, metric))
            ax.plot(
                xs,
                values,
                marker="o",
                markersize=4,
                color=COLORS[label],
                linewidth=2.2 if label == "B" else 1.4,
                label=f"Run {label}",
            )
            if retrieval:
                ax.scatter(
                    [run.best_epoch],
                    [overall(run.best, metric)],
                    s=110,
                    facecolors="none",
                    edgecolors=COLORS[label],
                    linewidths=1.8,
                    zorder=5,
                )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_xticks(sorted(runs["B"].epochs))
        ax.grid(alpha=0.3)
    axes[1].annotate(
        "not comparable\nacross filtered /\nunfiltered arms",
        xy=(0.97, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=7.5,
        color="#7a4b00",
    )
    axes[2].annotate(
        "circles = selected\ncheckpoint",
        xy=(0.97, 0.06),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#333333",
    )
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Training loss more than halves by epoch 3 while Recall@5 peaks at epoch 1 "
        "(Run A: epoch 2) — the extra epochs buy nothing",
        fontsize=11,
    )
    return finish(fig, out, "epoch_trajectories.png")


def plot_confounds(runs: dict[str, Run], out: Path) -> tuple[Path, dict[str, float]]:
    """Two things that could explain the ranking: validation loss and judged coverage."""
    points = [
        (label, epoch, runs[label].epochs[epoch])
        for label in RUNS
        for epoch in sorted(runs[label].epochs)
        if epoch > 0
    ]
    val_loss = np.array([overall(e, "val_loss") for _, _, e in points])
    recall = np.array([overall(e, SELECTOR) for _, _, e in points])
    judged = np.array([overall(e, "judged@5") for _, _, e in points])
    r_loss = float(np.corrcoef(val_loss, recall)[0, 1])
    r_judged = float(np.corrcoef(judged, recall)[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, xs, xlabel, r in (
        (axes[0], val_loss, "validation loss (triplet KD)", r_loss),
        (axes[1], judged, "judged@5 (evaluation coverage)", r_judged),
    ):
        for label in RUNS:
            idx = [i for i, (lab, _, _) in enumerate(points) if lab == label]
            ax.scatter(
                xs[idx], recall[idx], s=52, color=COLORS[label], label=f"Run {label}", zorder=3
            )
            for i in idx:
                ax.annotate(
                    f"e{points[i][1]}",
                    (xs[i], recall[i]),
                    textcoords="offset points",
                    xytext=(6, -2),
                    fontsize=7,
                    color="#555555",
                )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Recall@5")
        ax.grid(alpha=0.3)
        ax.set_title(f"Pearson r = {r:+.2f}  (12 trained checkpoints)", fontsize=10)
    axes[0].axhline(
        overall(runs["A"].baseline, SELECTOR), color=BASELINE_COLOR, linestyle="--", linewidth=1
    )
    axes[0].annotate(
        "untrained baseline",
        xy=(0.02, overall(runs["A"].baseline, SELECTOR)),
        xycoords=("axes fraction", "data"),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=7.5,
        color=BASELINE_COLOR,
    )
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Validation loss does not predict retrieval quality; judged@5 coverage does — "
        "the ranking's main caveat",
        fontsize=11,
    )
    path = finish(fig, out, "loss_vs_retrieval.png")
    return path, {"r_val_loss_recall": r_loss, "r_judged_recall": r_judged}


def label_density(val_path: Path) -> dict[str, dict[str, float]]:
    """Per-persona label sharpness in the shared validation file.

    The three personas share the same 92 question texts; only the teacher's relevance
    scores differ, so any per-persona gap in retrieval quality comes from the labels or
    from noise, never from the query. Recall/Hit/MRR binarize the teacher's top
    `relevance_top_m` (3) chunks, so the denominator is identical for every persona —
    what differs is how confidently the teacher separated those three from the rest.
    Counting chunks that clear 0.5 measures exactly that.
    """
    rows = [
        json.loads(line)
        for line in val_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scores: dict[str, list[list[float]]] = {persona: [] for persona in PERSONAS}
    queries: dict[str, set[str]] = {persona: set() for persona in PERSONAS}
    for row in rows:
        scores[row["persona_id"]].append([doc["teacher_score"] for doc in row["docs"]])
        queries[row["persona_id"]].add(row["query"])
    shared = len(set.intersection(*queries.values()))
    return {
        "relevant_per_query": {
            persona: float(np.mean([sum(1 for s in row if s >= 0.5) for row in per_query]))
            for persona, per_query in scores.items()
        },
        "mean_teacher_score": {
            persona: float(np.mean([np.mean(row) for row in per_query]))
            for persona, per_query in scores.items()
        },
        "shared_query_texts": {"count": float(shared), "of": float(len(queries["crammer"]))},
    }


def plot_per_persona(
    runs: dict[str, Run], out: Path, val_path: Path
) -> tuple[Path, dict[str, Any]]:
    """The headline gains are not shared equally by the three personas."""
    metrics = ("recall@5", "ndcg@5", "mrr")
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.1))
    table: dict[str, Any] = {}
    width = 0.19
    xs = np.arange(len(PERSONAS))
    for ax, metric in zip(axes[:3], metrics, strict=True):
        baseline = [runs["A"].baseline[metric][p] for p in PERSONAS]
        ax.bar(
            xs - 2 * width,
            baseline,
            width,
            color=BASELINE_COLOR,
            label="baseline",
        )
        for i, label in enumerate(RUNS):
            values = [runs[label].best[metric][p] for p in PERSONAS]
            ax.bar(
                xs + (i - 1) * width,
                values,
                width,
                color=COLORS[label],
                label=f"Run {label}",
            )
            table.setdefault(metric, {})[label] = dict(zip(PERSONAS, values, strict=True))
        table.setdefault(metric, {})["baseline"] = dict(zip(PERSONAS, baseline, strict=True))
        lo = min(baseline + [runs[r].best[metric][p] for r in RUNS for p in PERSONAS])
        ax.set_ylim(max(0.0, lo - 0.06), None)
        ax.set_xticks(xs)
        ax.set_xticklabels(PERSONAS)
        ax.set_title(metric, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[2].legend(fontsize=7.5, ncol=2, loc="upper left")
    axes[0].annotate(
        "crammer: Runs A and D fall below\nthe untrained encoder",
        xy=(0.03, 0.96),
        xycoords="axes fraction",
        va="top",
        fontsize=8,
        color="#8a2b2b",
    )

    density = label_density(val_path)
    table["label_density"] = density
    axes[3].bar(
        xs,
        [density["relevant_per_query"][p] for p in PERSONAS],
        0.55,
        color="#7a5c9e",
    )
    axes[3].set_xticks(xs)
    axes[3].set_xticklabels(PERSONAS)
    axes[3].set_ylabel("chunks with teacher_score ≥ 0.5")
    axes[3].set_title("label sharpness (same 92 questions)", fontsize=10)
    axes[3].grid(axis="y", alpha=0.3)
    for x, persona in enumerate(PERSONAS):
        axes[3].annotate(
            f"{density['relevant_per_query'][persona]:.2f}",
            (x, density["relevant_per_query"][persona]),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
        )
    fig.suptitle(
        "Per-persona best-checkpoint scores (n = 92 each) — crammer gains least, and its "
        "teacher labels are the least sharp",
        fontsize=11,
    )
    path = finish(fig, out, "per_persona.png")
    return path, table


def plot_persona_swap(runs: dict[str, Run], out: Path) -> tuple[Path, dict[str, Any]]:
    """Swapped minus matched: the personalization control, overall and per persona."""
    overall_stats = {
        label: compare(runs[label].best, runs[label].swap_entry(runs[label].best), ["ndcg@5"])[
            "ndcg@5"
        ]
        for label in RUNS
    }
    baseline_stats = compare(
        runs["A"].baseline, runs["A"].swap_entry(runs["A"].baseline), ["ndcg@5"]
    )["ndcg@5"]
    pids = np.array(runs["A"].baseline["per_query"]["persona_ids"])
    per_persona: dict[str, dict[str, dict[str, float]]] = {}
    for label, entry in [("baseline", runs["A"].baseline)] + [(r, runs[r].best) for r in RUNS]:
        run = runs["A"] if label == "baseline" else runs[label]
        per_persona[label] = {
            persona: compare(entry, run.swap_entry(entry), ["ndcg@5"], pids == persona)["ndcg@5"]
            for persona in PERSONAS
        }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    labels = ["baseline"] + [f"Run {r}" for r in RUNS]
    stats = [baseline_stats] + [overall_stats[r] for r in RUNS]
    colors = [BASELINE_COLOR] + [COLORS[r] for r in RUNS]
    ys = np.arange(len(labels))
    for y, stat, color in zip(ys, stats, colors, strict=True):
        axes[0].errorbar(
            [stat["delta"]],
            [y],
            xerr=[[stat["delta"] - stat["lo"]], [stat["hi"] - stat["delta"]]],
            fmt="o",
            markersize=6,
            color=color,
            elinewidth=1.6,
            capsize=3,
        )
    axes[0].axvline(0.0, color="#888888", linewidth=1)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("swapped − matched nDCG@5 (95% CI)")
    axes[0].set_title("Overall: every interval covers zero", fontsize=10)
    axes[0].grid(axis="x", alpha=0.3)

    width = 0.16
    xs = np.arange(len(PERSONAS))
    for i, label in enumerate(["baseline", *RUNS]):
        color = BASELINE_COLOR if label == "baseline" else COLORS[label]
        axes[1].bar(
            xs + (i - 2) * width,
            [per_persona[label][p]["delta"] for p in PERSONAS],
            width,
            color=color,
            label="baseline" if label == "baseline" else f"Run {label}",
        )
    axes[1].axhline(0.0, color="#888888", linewidth=1)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(PERSONAS)
    axes[1].set_ylabel("swapped − matched nDCG@5")
    axes[1].set_title(
        "Per persona (n = 92): the baseline's ±0.07 prefix bias cancels", fontsize=10
    )
    axes[1].legend(fontsize=7.5, ncol=2)
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle(
        "Persona-mismatch control — no arm reads the persona prefix, and training flattens "
        "the base encoder's prefix bias",
        fontsize=11,
    )
    path = finish(fig, out, "persona_swap.png")
    return path, {
        "overall": {"baseline": baseline_stats, **overall_stats},
        "per_persona": per_persona,
    }


def plot_per_query(runs: dict[str, Run], out: Path) -> tuple[Path, dict[str, Any]]:
    """Query-level distribution of Run B's gain: an average hides the losers."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    summary: dict[str, Any] = {}
    for ax, metric in zip(axes, ("ndcg@5", SELECTOR), strict=True):
        delta = per_query(runs["B"].best, metric) - per_query(runs["B"].baseline, metric)
        better = int((delta > 1e-9).sum())
        worse = int((delta < -1e-9).sum())
        tie = int(delta.size - better - worse)
        summary[metric] = {
            "better": better,
            "tie": tie,
            "worse": worse,
            "mean": float(delta.mean()),
            "median": float(np.median(delta)),
        }
        ax.hist(delta, bins=31, color=COLORS["B"], edgecolor="white", linewidth=0.4)
        ax.axvline(0.0, color="#888888", linewidth=1)
        ax.axvline(
            float(delta.mean()),
            color="#b5651d",
            linestyle="--",
            linewidth=1.4,
            label=f"mean {delta.mean():+.3f}",
        )
        ax.set_xlabel(f"per-query {metric}: Run B − baseline")
        ax.set_ylabel("queries")
        ax.set_title(f"better {better} / tie {tie} / worse {worse}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        "Run B's gain is a majority effect, not a uniform one: 60 of 276 queries rank worse "
        "than untrained",
        fontsize=11,
    )
    path = finish(fig, out, "per_query_delta.png")
    return path, summary


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", type=Path, default=Path("models/ropg"))
    ap.add_argument("--out", type=Path, default=Path("docs/results/figures/ropg-runs-v1"))
    ap.add_argument(
        "--val",
        type=Path,
        default=Path("data/ropg_kd/val.jsonl"),
        help="validation labels, for the per-persona label-density panel",
    )
    args = ap.parse_args()

    runs = {label: Run(args.runs, label) for label in RUNS}
    args.out.mkdir(parents=True, exist_ok=True)

    # Paired tests and every baseline-relative panel are void unless the untrained
    # evaluations agree exactly, which is the same guard compare_runs.py applies.
    reference = runs["A"].baseline["per_query"]
    for label in RUNS[1:]:
        if runs[label].baseline["per_query"] != reference:
            raise ValueError(
                f"run {label}: epoch-0 per-query vectors differ from run A. The runs did not "
                "score the same rows, so no cross-arm figure here is valid."
            )

    stats: dict[str, Any] = {
        "n_queries": len(reference["persona_ids"]),
        "runs": {
            label: {
                "log": str(runs[label].path),
                "seed": runs[label].seed,
                "anchor": runs[label].anchor,
                "train_data": runs[label].data,
                "best_epoch": runs[label].best_epoch,
                "epochs": len(runs[label].epochs) - 1,
                "best": {m: overall(runs[label].best, m) for m in HEADLINE},
            }
            for label in RUNS
        },
        "baseline": {m: overall(runs["A"].baseline, m) for m in HEADLINE},
    }

    written = [plot_metric_ladder(runs, args.out), plot_epoch_trajectories(runs, args.out)]
    for path, payload, key in (
        (*plot_headline_deltas(runs, args.out), "paired"),
        (*plot_confounds(runs, args.out), "confounds"),
        (*plot_per_persona(runs, args.out, args.val), "per_persona"),
        (*plot_persona_swap(runs, args.out), "persona_swap"),
        (*plot_per_query(runs, args.out), "per_query"),
    ):
        written.append(path)
        stats[key] = payload

    (args.out / "stats.json").write_text(json.dumps(stats, indent=1, sort_keys=True), "utf-8")
    for path in written:
        print(f"wrote {path}")
    print(f"wrote {args.out / 'stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
