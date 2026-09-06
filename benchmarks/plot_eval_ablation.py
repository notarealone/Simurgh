"""Figures for the held-out five-arm ablation evaluation.

Every number is recomputed from the run's `results.jsonl`, so the figures cannot drift
from `summary.csv` / `paired_deltas.csv` that `eval_runner.py` wrote beside it. The paired
bootstrap, sign-flip permutation test, and Holm correction are imported from
`compare_runs.py` for the same reason.

The five arms score the same 94 test questions x 4 personas, one replicate, under one
config digest, which is what makes every panel here a paired comparison. The script
asserts that pairing (same key set per arm, one digest across the file) before plotting.

Usage:
    python benchmarks/plot_eval_ablation.py \
        --run results/ablation/complete_eval_results/ablation \
        --out docs/results/figures/eval-ablation-v1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_runs import holm, paired_stats
from judge import SCORE_FIELDS

# Ladder order: each arm adds exactly one component to the arm before it.
ARMS = (
    "base_embedder_no_profile",
    "base_embedder",
    "trained_embedder",
    "trained_embedder_base_rewriter",
    "trained_embedder_rewriter",
)
LABELS = {
    "base_embedder_no_profile": "naive RAG",
    "base_embedder": "+persona\nprompt",
    "trained_embedder": "+ROPG\nembedder",
    "trained_embedder_base_rewriter": "+base\nrewriter",
    "trained_embedder_rewriter": "+WPO\nrewriter",
}
RUNGS = tuple(pairwise(ARMS))
REWRITER_ARMS = ("trained_embedder_base_rewriter", "trained_embedder_rewriter")
PERSONAS = ("crammer", "scholar", "steady", "newcomer")
HELD_OUT = "newcomer"
JUDGES = ("primary", "secondary")
# 0-4 rubric fields first, the binary field last: they do not share a y-axis.
GRADED = tuple(f for f in SCORE_FIELDS if f != "answer_correctness")
METRICS = (*GRADED, "answer_correctness")
N_BOOT = 10000
SEED = 42
COLORS = {
    "base_embedder_no_profile": "#9aa5b1",
    "base_embedder": "#c98a3c",
    "trained_embedder": "#2f6f9f",
    "trained_embedder_base_rewriter": "#7a6ea8",
    "trained_embedder_rewriter": "#4f9d69",
}
JUDGE_TITLE = {
    "primary": "primary judge (gpt-5.6-luna)",
    "secondary": "secondary judge (gemini-3.5-flash-lite)",
}
# Persian word forms, matched against rewritten queries. "grade nine" and "exam" are
# persona vocabulary; they belong to the profile, not to any test question.
PERSONA_WORDS = {"grade nine": "نهم", "exam": "امتحان"}
CONTENT_TOKEN = re.compile(r"[\u0600-\u06FF\u200c]{3,}")
FIDELITY_CUT = 0.3


class Run:
    """One evaluation run: per-key scores, retrieval hits, and rewritten queries."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
        records = [
            json.loads(line)
            for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        digests = {r["config_sha256"] for r in records}
        if len(digests) != 1:
            raise ValueError(
                f"{len(digests)} config digests in results.jsonl; rows are not comparable"
            )
        statuses = Counter(r["status"] for r in records)
        if set(statuses) != {"ok"}:
            raise ValueError(f"results.jsonl is not all successful rows: {dict(statuses)}")

        self.digest = digests.pop()
        self.keys: list[str] = sorted({f"{r['question_ref']}|{r['persona_id']}" for r in records})
        index = {key: i for i, key in enumerate(self.keys)}
        n = len(self.keys)

        self.scores: dict[tuple[str, str, str], np.ndarray] = {
            (arm, judge, metric): np.full(n, np.nan)
            for arm in ARMS
            for judge in JUDGES
            for metric in METRICS
        }
        self.hits: dict[str, list[tuple[str, ...]]] = {arm: [()] * n for arm in ARMS}
        self.rewrites: dict[str, list[str | None]] = {arm: [None] * n for arm in ARMS}
        self.originals: list[str] = [""] * n
        seen: dict[str, set[str]] = {arm: set() for arm in ARMS}

        for record in records:
            arm, key = record["arm"], f"{record['question_ref']}|{record['persona_id']}"
            i = index[key]
            seen[arm].add(key)
            for judge in JUDGES:
                scored = record[f"{judge}_scores"] or {}
                for metric in METRICS:
                    if metric in scored:
                        self.scores[(arm, judge, metric)][i] = float(scored[metric])
            self.hits[arm][i] = tuple(hit["chunk_id"] for hit in record["hits"])
            self.rewrites[arm][i] = record["rewritten_query"]
            self.originals[i] = record["original_query"]

        for arm in ARMS:
            if seen[arm] != set(self.keys):
                raise ValueError(
                    f"arm {arm} covers {len(seen[arm])} of {n} keys; pairing is invalid"
                )
        self.personas = np.array([key.rsplit("|", 1)[1] for key in self.keys])

    def mask(self, persona: str | None) -> np.ndarray:
        if persona is None:
            return np.ones(len(self.keys), dtype=bool)
        if persona == "train":
            return self.personas != HELD_OUT
        return self.personas == persona

    def mean(self, arm: str, judge: str, metric: str, persona: str | None = None) -> float:
        values = self.scores[(arm, judge, metric)][self.mask(persona)]
        return float(np.nanmean(values))


def compare(
    run: Run, before: str, after: str, judge: str, persona: str | None = None
) -> dict[str, dict[str, float]]:
    """Paired after-minus-before stats per metric, Holm-corrected across the metric family."""
    rows: dict[str, dict[str, float]] = {}
    keep = run.mask(persona)
    for metric in METRICS:
        # One generator per metric, seeded identically, matching compare_runs.py so the
        # figures and the runner's own CSVs print the same intervals.
        rng = np.random.default_rng(SEED)
        a = run.scores[(before, judge, metric)][keep]
        b = run.scores[(after, judge, metric)][keep]
        both = ~np.isnan(a) & ~np.isnan(b)
        rows[metric] = paired_stats(a[both], b[both], N_BOOT, rng)
    for metric, adjusted in zip(rows, holm([r["p"] for r in rows.values()]), strict=True):
        rows[metric]["p_holm"] = adjusted
    return rows


def finish(fig: plt.Figure, out: Path, name: str) -> Path:
    path = out / name
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def short(metric: str) -> str:
    return metric.replace("_", "\n")


def plot_ladder(run: Run, out: Path) -> tuple[Path, dict[str, Any]]:
    """Every arm on every rubric field, per judge: the headline table as bars."""
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2), gridspec_kw={"width_ratios": [4, 1]})
    stats: dict[str, Any] = {}
    for row, judge in enumerate(JUDGES):
        for col, metrics in enumerate((GRADED, ("answer_correctness",))):
            ax = axes[row][col]
            x = np.arange(len(metrics))
            width = 0.16
            for k, arm in enumerate(ARMS):
                values = [run.mean(arm, judge, m) for m in metrics]
                ax.bar(
                    x + (k - 2) * width,
                    values,
                    width,
                    color=COLORS[arm],
                    label=LABELS[arm].replace("\n", " "),
                    edgecolor="white",
                    linewidth=0.5,
                )
                for xi, value in zip(x, values, strict=True):
                    ax.annotate(
                        f"{value:.2f}",
                        (xi + (k - 2) * width, value),
                        textcoords="offset points",
                        xytext=(0, 2),
                        ha="center",
                        fontsize=5.6,
                        rotation=90,
                    )
                    stats[f"{judge}|{arm}|{metrics[list(x).index(xi)]}"] = value
            ax.set_xticks(x, [short(m) for m in metrics], fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylim(0, 4.35)
                ax.set_ylabel("mean score (0-4)")
                ax.set_title(JUDGE_TITLE[judge], fontsize=10, loc="left")
            else:
                ax.set_ylim(0, 1.0)
                ax.set_ylabel("mean (0/1)")
                ax.set_title("binary field", fontsize=10, loc="left")
    axes[0][0].legend(
        fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.62, 1.24), frameon=False
    )
    fig.suptitle(
        "Ablation ladder on the held-out test split (94 questions x 4 personas, 1 replicate)",
        fontsize=11,
        y=1.04,
    )
    fig.tight_layout()
    return finish(fig, out, "ladder.png"), stats


def plot_rung_deltas(run: Run, out: Path) -> tuple[Path, dict[str, Any]]:
    """Each rung's paired gain over the rung below, with CIs and Holm stars."""
    fig, axes = plt.subplots(2, len(RUNGS), figsize=(14, 6.4), sharey="row")
    stats: dict[str, Any] = {}
    for row, judge in enumerate(JUDGES):
        for col, (before, after) in enumerate(RUNGS):
            ax = axes[row][col]
            rows = compare(run, before, after, judge)
            stats[f"{judge}|{after} - {before}"] = rows
            y = np.arange(len(METRICS))[::-1]
            for yi, metric in zip(y, METRICS, strict=True):
                s = rows[metric]
                significant = s["p_holm"] < 0.05
                color = COLORS[after] if s["delta"] >= 0 else "#b03a2e"
                ax.errorbar(
                    s["delta"],
                    yi,
                    xerr=[[s["delta"] - s["lo"]], [s["hi"] - s["delta"]]],
                    fmt="o",
                    ms=5,
                    color=color,
                    ecolor=color,
                    elinewidth=1.6,
                    capsize=3,
                    alpha=1.0 if significant else 0.55,
                )
                if significant:
                    ax.annotate(
                        "*",
                        (s["hi"], yi),
                        textcoords="offset points",
                        xytext=(4, -3),
                        fontsize=13,
                        color=color,
                    )
            ax.axvline(0, color="#444444", lw=1)
            ax.set_yticks(y, [m.replace("_", " ") for m in METRICS], fontsize=8)
            ax.grid(axis="x", alpha=0.3)
            ax.set_axisbelow(True)
            ax.set_xlim(-0.35, 0.72)
            if row == 0:
                ax.set_title(
                    f"rung {col + 2}: {LABELS[after]}".replace("\n", " "), fontsize=9.5, loc="left"
                )
            if col == 0:
                ax.set_ylabel(JUDGE_TITLE[judge].split(" (")[0], fontsize=9)
            if row == 1:
                ax.set_xlabel("paired delta vs rung below")
    fig.suptitle(
        "Rung-by-rung paired gains, 95% bootstrap CI, * = Holm-adjusted p < 0.05 within the judge's 5-metric family",
        fontsize=10.5,
        y=1.0,
    )
    fig.tight_layout()
    return finish(fig, out, "rung_deltas.png"), stats


def plot_per_persona(run: Run, out: Path) -> tuple[Path, dict[str, Any]]:
    """Who each rung's gain belongs to, and whether the held-out persona behaves differently."""
    metrics = ("persona_alignment", "pedagogical_quality", "context_utility")
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4.3), sharey=True)
    stats: dict[str, Any] = {}
    for ax, metric in zip(axes, metrics, strict=True):
        y = np.arange(len(PERSONAS))[::-1]
        height = 0.16
        for k, arm in enumerate(ARMS):
            values = [run.mean(arm, "primary", metric, persona) for persona in PERSONAS]
            ax.barh(
                y + (k - 2) * height,
                values,
                height,
                color=COLORS[arm],
                label=LABELS[arm].replace("\n", " "),
                edgecolor="white",
                linewidth=0.5,
            )
            for persona, value in zip(PERSONAS, values, strict=True):
                stats[f"{arm}|{metric}|{persona}"] = value
        ax.set_yticks(
            y,
            [p + ("\n(held out)" if p == HELD_OUT else "") for p in PERSONAS],
            fontsize=8.5,
        )
        ax.set_xlim(2.0, 4.0)
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_title(metric.replace("_", " "), fontsize=10, loc="left")
        ax.set_xlabel("primary-judge mean (0-4)")
    fig.suptitle("Per-persona scores: three training personas and the held-out one", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        fontsize=8,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        frameon=False,
    )
    return finish(fig, out, "per_persona.png"), stats


def plot_retrieval_shift(run: Run, out: Path) -> tuple[Path, dict[str, Any]]:
    """What each component did to retrieval, independent of the judges."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    stats: dict[str, Any] = {}

    overlaps = []
    for before, after in RUNGS:
        shared = [
            len(set(a) & set(b)) / 5.0
            for a, b in zip(run.hits[before], run.hits[after], strict=True)
        ]
        overlaps.append(float(np.mean(shared)))
        stats[f"overlap|{after} vs {before}"] = overlaps[-1]
    ax = axes[0]
    x = np.arange(len(RUNGS))
    ax.bar(x, overlaps, 0.55, color=[COLORS[after] for _, after in RUNGS], edgecolor="white")
    for xi, value in zip(x, overlaps, strict=True):
        ax.annotate(
            f"{value:.2f}",
            (xi, value),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(x, [LABELS[after] for _, after in RUNGS], fontsize=7.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("mean fraction of top-5 shared")
    ax.set_title("how far each rung moved the top 5", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    ax = axes[1]
    zeros = []
    for arm in ARMS:
        values = run.scores[(arm, "primary", "context_utility")]
        zeros.append(float(np.nanmean(values == 0)))
        stats[f"zero_context_utility|{arm}"] = zeros[-1]
    ax.bar(np.arange(len(ARMS)), zeros, 0.55, color=[COLORS[a] for a in ARMS], edgecolor="white")
    for xi, value in enumerate(zeros):
        ax.annotate(
            f"{value:.1%}",
            (xi, value),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(np.arange(len(ARMS)), [LABELS[a] for a in ARMS], fontsize=7.5)
    ax.set_ylabel("share of rows")
    ax.set_title("rows the primary judge scored context_utility = 0", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    ax = axes[2]
    distinct = []
    for arm in ARMS:
        by_question: dict[str, set[tuple[str, ...]]] = {}
        for key, hits in zip(run.keys, run.hits[arm], strict=True):
            by_question.setdefault(key.rsplit("|", 1)[0], set()).add(tuple(sorted(hits)))
        distinct.append(float(np.mean([len(v) for v in by_question.values()])))
        stats[f"distinct_hitsets|{arm}"] = distinct[-1]
    ax.bar(
        np.arange(len(ARMS)), distinct, 0.55, color=[COLORS[a] for a in ARMS], edgecolor="white"
    )
    for xi, value in enumerate(distinct):
        ax.annotate(
            f"{value:.2f}",
            (xi, value),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
        )
    ax.axhline(1.0, color="#444444", lw=1, ls=":")
    ax.set_xticks(np.arange(len(ARMS)), [LABELS[a] for a in ARMS], fontsize=7.5)
    ax.set_ylim(0, 4.3)
    ax.set_ylabel("distinct top-5 sets per question")
    ax.set_title("persona divergence of retrieval (max 4)", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.suptitle(
        "Retrieval diagnostics, measured on the returned chunk ids alone", fontsize=11, y=1.02
    )
    fig.tight_layout()
    return finish(fig, out, "retrieval_shift.png"), stats


def token_fidelity(run: Run, arm: str) -> np.ndarray:
    """Fraction of the question's Persian content tokens the rewrite kept."""
    out = np.full(len(run.keys), np.nan)
    for i, (original, rewritten) in enumerate(zip(run.originals, run.rewrites[arm], strict=True)):
        if rewritten is None:
            continue
        source = set(CONTENT_TOKEN.findall(original))
        if not source:
            continue
        out[i] = len(source & set(CONTENT_TOKEN.findall(rewritten))) / len(source)
    return out


def plot_rewriter_mechanism(run: Run, out: Path) -> tuple[Path, dict[str, Any]]:
    """Why the rewriting rungs cost context utility: the rewrites drop the question."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    stats: dict[str, Any] = {}
    fidelity = {arm: token_fidelity(run, arm) for arm in REWRITER_ARMS}

    ax = axes[0]
    bins = np.linspace(0, 1, 21)
    for arm in REWRITER_ARMS:
        values = fidelity[arm][~np.isnan(fidelity[arm])]
        ax.hist(
            values, bins=bins, alpha=0.6, color=COLORS[arm], label=LABELS[arm].replace("\n", " ")
        )
        stats[f"token_fidelity_mean|{arm}"] = float(values.mean())
        stats[f"token_fidelity_below_cut|{arm}"] = float(np.mean(values < FIDELITY_CUT))
    ax.axvline(FIDELITY_CUT, color="#444444", lw=1, ls=":")
    ax.set_xlabel("fraction of the question's content tokens kept")
    ax.set_ylabel("rows")
    ax.set_title("rewrite fidelity to the question", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    ax = axes[1]
    x = np.arange(2)
    width = 0.35
    for k, arm in enumerate(REWRITER_ARMS):
        scores = run.scores[(arm, "primary", "context_utility")]
        low = fidelity[arm] < FIDELITY_CUT
        high = fidelity[arm] >= FIDELITY_CUT
        values = [float(np.nanmean(scores[low])), float(np.nanmean(scores[high]))]
        counts = [int(np.sum(low)), int(np.sum(high))]
        ax.bar(x + (k - 0.5) * width, values, width, color=COLORS[arm], edgecolor="white")
        for xi, value, count in zip(x, values, counts, strict=True):
            ax.annotate(
                f"{value:.2f}\nn={count}",
                (xi + (k - 0.5) * width, value),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=7.5,
            )
        stats[f"context_utility_by_fidelity|{arm}"] = {
            "below": values[0],
            "at_or_above": values[1],
        }
    baseline = run.mean("trained_embedder", "primary", "context_utility")
    ax.axhline(baseline, color=COLORS["trained_embedder"], lw=1.3, ls="--")
    ax.annotate(
        f"no rewriter: {baseline:.2f}",
        (-0.42, baseline),
        textcoords="offset points",
        xytext=(0, 5),
        ha="left",
        fontsize=8,
        color=COLORS["trained_embedder"],
    )
    stats["context_utility|trained_embedder"] = baseline
    ax.set_xticks(x, [f"fidelity < {FIDELITY_CUT}", f"fidelity >= {FIDELITY_CUT}"], fontsize=8.5)
    ax.set_ylim(0, 4.3)
    ax.set_ylabel("primary-judge context_utility")
    ax.set_title("the cost sits on the low-fidelity rewrites", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    ax = axes[2]
    x = np.arange(len(PERSONA_WORDS))
    for k, arm in enumerate(REWRITER_ARMS):
        rates = []
        for word in PERSONA_WORDS.values():
            present = [word in (r or "") for r in run.rewrites[arm] if r is not None]
            rates.append(float(np.mean(present)))
        ax.bar(x + (k - 0.5) * width, rates, width, color=COLORS[arm], edgecolor="white")
        for xi, rate in zip(x, rates, strict=True):
            ax.annotate(
                f"{rate:.1%}",
                (xi + (k - 0.5) * width, rate),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=8,
            )
        stats[f"persona_word_rate|{arm}"] = dict(zip(PERSONA_WORDS, rates, strict=True))
    # Persian glyphs are not shaped by matplotlib, so the tick labels stay English;
    # PERSONA_WORDS holds the forms actually matched.
    ax.set_xticks(x, [f"{en!r}".strip(chr(39)) for en in PERSONA_WORDS], fontsize=8.5)
    ax.set_ylim(0, 0.32)
    ax.set_ylabel("share of rewrites containing the word")
    ax.set_title("persona vocabulary injected into the query", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.suptitle("What the two rewriters actually did to the query", fontsize=11, y=1.02)
    fig.tight_layout()
    return finish(fig, out, "rewriter_mechanism.png"), stats


def plot_per_question(run: Run, out: Path) -> tuple[Path, dict[str, Any]]:
    """Win / tie / loss per key, so the means are not read as uniform effects."""
    metrics = ("context_utility", "persona_alignment", "pedagogical_quality")
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4.0), sharey=True)
    stats: dict[str, Any] = {}
    for ax, metric in zip(axes, metrics, strict=True):
        y = np.arange(len(RUNGS))[::-1]
        for yi, (before, after) in zip(y, RUNGS, strict=True):
            a = run.scores[(before, "primary", metric)]
            b = run.scores[(after, "primary", metric)]
            both = ~np.isnan(a) & ~np.isnan(b)
            d = b[both] - a[both]
            counts = (int(np.sum(d > 0)), int(np.sum(d == 0)), int(np.sum(d < 0)))
            stats[f"{metric}|{after} - {before}"] = {
                "win": counts[0],
                "tie": counts[1],
                "loss": counts[2],
            }
            left = 0.0
            for value, color, name in zip(
                counts, ("#2f6f9f", "#d8dde3", "#b03a2e"), ("win", "tie", "loss"), strict=True
            ):
                ax.barh(
                    yi,
                    value,
                    0.6,
                    left=left,
                    color=color,
                    edgecolor="white",
                    label=name if yi == y[0] else None,
                )
                if value:
                    ax.annotate(
                        str(value),
                        (left + value / 2, yi),
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="#222222" if name == "tie" else "white",
                    )
                left += value
        ax.set_yticks(y, [f"{LABELS[after]}".replace("\n", " ") for _, after in RUNGS], fontsize=8)
        ax.set_xlabel("rows (of 376)")
        ax.set_title(metric.replace("_", " "), fontsize=10, loc="left")
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Per-row wins, ties and losses against the rung below (primary judge)", fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        frameon=False,
    )
    return finish(fig, out, "per_question.png"), stats


def plot_judge_agreement(run: Run, out: Path) -> tuple[Path, dict[str, Any]]:
    """How far the two judges agree, per metric and on the rung deltas themselves."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    stats: dict[str, Any] = {}

    exact, corr = [], []
    for metric in METRICS:
        a = np.concatenate([run.scores[(arm, "primary", metric)] for arm in ARMS])
        b = np.concatenate([run.scores[(arm, "secondary", metric)] for arm in ARMS])
        both = ~np.isnan(a) & ~np.isnan(b)
        exact.append(float(np.mean(a[both] == b[both])))
        corr.append(float(np.corrcoef(a[both], b[both])[0, 1]))
        stats[f"agreement|{metric}"] = {
            "exact": exact[-1],
            "pearson_r": corr[-1],
            "n": int(both.sum()),
        }
    x = np.arange(len(METRICS))
    ax = axes[0]
    ax.bar(x - 0.2, exact, 0.4, color="#2f6f9f", label="exact agreement", edgecolor="white")
    ax.bar(x + 0.2, corr, 0.4, color="#c98a3c", label="Pearson r", edgecolor="white")
    ax.set_xticks(x, [short(m) for m in METRICS], fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("per-metric judge agreement (pooled over arms)", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    ax = axes[1]
    xs, ys, labels = [], [], []
    for before, after in RUNGS:
        for metric in METRICS:
            p = compare(run, before, after, "primary")[metric]["delta"]
            s = compare(run, before, after, "secondary")[metric]["delta"]
            xs.append(p)
            ys.append(s)
            labels.append(f"{after} - {before}|{metric}")
    r = float(np.corrcoef(xs, ys)[0, 1])
    stats["delta_agreement_pearson_r"] = r
    stats["delta_sign_agreement"] = float(np.mean(np.sign(xs) == np.sign(ys)))
    ax.axhline(0, color="#999999", lw=0.8)
    ax.axvline(0, color="#999999", lw=0.8)
    lim = max(max(np.abs(xs)), max(np.abs(ys))) * 1.15
    ax.plot([-lim, lim], [-lim, lim], color="#444444", lw=1, ls=":")
    ax.scatter(xs, ys, s=26, color="#4f9d69", alpha=0.85, edgecolor="white")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("primary-judge rung delta")
    ax.set_ylabel("secondary-judge rung delta")
    ax.set_title(f"the judges agree on the deltas (r = {r:.2f})", fontsize=10, loc="left")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    ax = axes[2]
    diffs = []
    for metric in METRICS:
        a = np.concatenate([run.scores[(arm, "primary", metric)] for arm in ARMS])
        b = np.concatenate([run.scores[(arm, "secondary", metric)] for arm in ARMS])
        both = ~np.isnan(a) & ~np.isnan(b)
        diffs.append(float(np.mean(b[both] - a[both])))
        stats[f"judge_bias|{metric}"] = diffs[-1]
    colors = ["#b03a2e" if d < 0 else "#2f6f9f" for d in diffs]
    ax.bar(x, diffs, 0.5, color=colors, edgecolor="white")
    for xi, value in zip(x, diffs, strict=True):
        ax.annotate(
            f"{value:+.2f}",
            (xi, value),
            textcoords="offset points",
            xytext=(0, 4 if value >= 0 else -11),
            ha="center",
            fontsize=8,
        )
    ax.axhline(0, color="#444444", lw=1)
    span = max(abs(min(diffs)), abs(max(diffs)))
    ax.set_ylim(min(diffs) - 0.14 * span, max(diffs) + 0.14 * span)
    ax.set_xticks(x, [short(m) for m in METRICS], fontsize=7.5)
    ax.set_ylabel("secondary minus primary")
    ax.set_title("systematic level difference between judges", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.suptitle(
        "Judge cross-validation: level disagreement is large, delta agreement is high",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    return finish(fig, out, "judge_agreement.png"), stats


def plot_headline(run: Run, out: Path) -> tuple[Path, dict[str, Any]]:
    """The two comparisons a reader wants: full system vs naive, and vs the best rung."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
    stats: dict[str, Any] = {}
    pairs = (
        ("base_embedder_no_profile", "trained_embedder_rewriter", "full system vs naive RAG"),
        (
            "trained_embedder",
            "trained_embedder_rewriter",
            "full system vs best rung (ROPG embedder)",
        ),
    )
    for ax, (before, after, title) in zip(axes, pairs, strict=True):
        y = np.arange(len(METRICS))[::-1]
        offset = 0.16
        for k, judge in enumerate(JUDGES):
            rows = compare(run, before, after, judge)
            stats[f"{judge}|{after} - {before}"] = rows
            color = "#2f6f9f" if judge == "primary" else "#c98a3c"
            for yi, metric in zip(y, METRICS, strict=True):
                s = rows[metric]
                ax.errorbar(
                    s["delta"],
                    yi + (0.5 - k) * offset,
                    xerr=[[s["delta"] - s["lo"]], [s["hi"] - s["delta"]]],
                    fmt="o",
                    ms=5,
                    color=color,
                    ecolor=color,
                    elinewidth=1.6,
                    capsize=3,
                    label=JUDGE_TITLE[judge].split(" (")[0] if yi == y[0] else None,
                    alpha=1.0 if s["p_holm"] < 0.05 else 0.5,
                )
                if s["p_holm"] < 0.05:
                    ax.annotate(
                        "*",
                        (s["hi"], yi + (0.5 - k) * offset),
                        textcoords="offset points",
                        xytext=(4, -3),
                        fontsize=13,
                        color=color,
                    )
        ax.axvline(0, color="#444444", lw=1)
        ax.set_yticks(y, [m.replace("_", " ") for m in METRICS], fontsize=9)
        ax.set_xlabel("paired delta")
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(
        "Headline comparisons, 95% bootstrap CI, * = Holm-adjusted p < 0.05 within the panel's 5-metric family",
        fontsize=10.5,
        y=1.02,
    )
    fig.tight_layout()
    return finish(fig, out, "headline.png"), stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", required=True, help="directory holding results.jsonl and run_manifest.json"
    )
    parser.add_argument("--out", required=True, help="figure output directory")
    args = parser.parse_args()

    run = Run(Path(args.run))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "run": str(Path(args.run)),
        "config_sha256": run.digest,
        "keys": len(run.keys),
        "arms": list(ARMS),
        "n_boot": N_BOOT,
        "seed": SEED,
    }
    for name, plot in (
        ("ladder", plot_ladder),
        ("headline", plot_headline),
        ("rung_deltas", plot_rung_deltas),
        ("per_persona", plot_per_persona),
        ("retrieval_shift", plot_retrieval_shift),
        ("rewriter_mechanism", plot_rewriter_mechanism),
        ("per_question", plot_per_question),
        ("judge_agreement", plot_judge_agreement),
    ):
        path, section = plot(run, out)
        stats[name] = section
        print(f"wrote {path}")

    (out / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {out / 'stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
