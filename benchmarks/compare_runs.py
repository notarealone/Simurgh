"""Paired significance testing over the per-query vectors in ``training_log.json``.

``rl.ropg_kd.evaluate_retrieval`` returns per-query metric vectors and the training
loop persists them for the untrained baseline and every epoch, precisely so that two
evaluations can be compared with a **paired** test. This is the consumer. Until it
runs, no delta in the results tables is claimable: with *n* = 92 per persona the
differences at stake are about one standard error, and the untrained baseline's own
per-persona nDCG@5 spread (0.5368 / 0.5591) already sits under one — before any
gradient is taken.

Pairing is what buys the power. Every evaluation scores the *same* 276 (question,
persona) rows, so per-query difficulty — by far the largest source of variance —
cancels in the difference. An unpaired comparison of two means throws that away.

Two statistics, because they answer different questions:

* **Bootstrap percentile CI** on the mean difference — the effect size and its
  uncertainty. Resamples queries with replacement.
* **Sign-flip permutation p-value** — the significance. Under the null "the paired
  differences are symmetric about zero", flipping their signs at random is exactly
  as likely as what was observed. This is a better-calibrated p-value than reading
  one off the bootstrap distribution, and costs the same.

Reported p-values are also **Holm-corrected** across the metric family, because a
run reports a dozen metrics and the largest of twelve noise draws looks impressive
on its own.

Usage::

    # untrained baseline (epoch 0) vs the run's best epoch
    uv run python benchmarks/compare_runs.py runB_training_log.json

    # one arm against another, at each one's best epoch
    uv run python benchmarks/compare_runs.py runA_log.json runC_log.json

    # pin specific epochs, and write the table into the thesis
    uv run python benchmarks/compare_runs.py runB.json --b-epoch 1 \\
        --out docs/results/stage1-runB-epoch1.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Not per-query, so not paired-testable: one scalar per epoch. Reported for context
# only. It is also the one number that is NOT comparable across arms — a filtered
# arm computes its val loss over a different (smaller) set of triplets.
SCALAR_KEYS = ("val_loss", "train_loss", "epoch")


def load_log(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        log = json.load(fh)
    if "epoch_metrics" not in log:
        raise ValueError(f"{path}: not a training_log.json (no 'epoch_metrics' key)")
    return log


def pick_entry(log: dict[str, Any], path: str, epoch: int | None) -> dict[str, Any]:
    """Return the epoch entry to compare, defaulting to the run's own best epoch."""
    entries = log["epoch_metrics"]
    want = log.get("best_epoch", 0) if epoch is None else epoch
    for entry in entries:
        if entry.get("epoch") == want:
            if "per_query" not in entry:
                raise ValueError(
                    f"{path}: epoch {want} has no per_query vectors. Retrieval eval was "
                    "disabled for that run (no corpus, or no val groups), so there is "
                    "nothing to pair."
                )
            return entry
    have = [e.get("epoch") for e in entries]
    raise ValueError(f"{path}: no epoch {want} in the log (have {have})")


def check_alignment(a: dict[str, Any], b: dict[str, Any], label_a: str, label_b: str) -> None:
    """Refuse to pair two evaluations that did not score the same rows in the same order.

    ``evaluate_retrieval`` builds its vectors in group order and skips a group only
    when none of its judged chunks are in the corpus — a decision that depends on the
    data alone, never on the model. So identical ``persona_ids`` is a sound proxy for
    "same rows, same order". If it fails, the two runs used different val data and the
    pairing would silently compare unrelated questions.
    """
    pa, pb = a["per_query"]["persona_ids"], b["per_query"]["persona_ids"]
    if len(pa) != len(pb):
        raise ValueError(
            f"{label_a} scored {len(pa)} queries but {label_b} scored {len(pb)}. "
            "These runs did not use the same val set — pairing is invalid."
        )
    if pa != pb:
        n_diff = sum(1 for x, y in zip(pa, pb, strict=True) if x != y)
        raise ValueError(
            f"{label_a} and {label_b} disagree on persona_ids at {n_diff}/{len(pa)} "
            "positions. Same length but different rows or a different order — pairing "
            "is invalid."
        )


def paired_stats(
    before: np.ndarray, after: np.ndarray, n_boot: int, rng: np.random.Generator
) -> dict[str, float]:
    """Mean paired difference with a bootstrap CI and a sign-flip permutation p-value."""
    d = after - before
    n = d.size
    observed = float(d.mean())

    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # Sign-flip permutation. The +1 in both terms is the standard guard against
    # reporting p = 0 from a finite number of permutations.
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_boot, n))
    null = (d * signs).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_boot + 1)

    return {
        "before": float(before.mean()),
        "after": float(after.mean()),
        "delta": observed,
        "lo": float(lo),
        "hi": float(hi),
        "p": float(p),
        "n": n,
    }


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment. Controls family-wise error across metrics.

    Uniformly more powerful than plain Bonferroni at the same guarantee, and it makes
    no independence assumption — which matters here, where nDCG@1..5 and Hit@1..5 are
    heavily correlated by construction.
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adjusted[i] = running
    return adjusted


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def compare(
    entry_a: dict[str, Any],
    entry_b: dict[str, Any],
    subset: str | None,
    n_boot: int,
    seed: int,
) -> list[tuple[str, dict[str, float]]]:
    """Paired stats for every metric both entries carry, optionally within one persona."""
    pq_a, pq_b = entry_a["per_query"], entry_b["per_query"]
    metrics = [k for k in pq_a if k != "persona_ids" and k in pq_b]

    mask = None
    if subset is not None:
        mask = np.array([pid == subset for pid in pq_a["persona_ids"]])
        if not mask.any():
            return []

    rows = []
    for name in metrics:
        # One generator per metric, seeded identically, so a metric's numbers do not
        # shift when an unrelated metric is added to the log.
        rng = np.random.default_rng(seed)
        before = np.asarray(pq_a[name], dtype=float)
        after = np.asarray(pq_b[name], dtype=float)
        if mask is not None:
            before, after = before[mask], after[mask]
        rows.append((name, paired_stats(before, after, n_boot, rng)))

    adjusted = holm([s["p"] for _, s in rows])
    for (_, s), padj in zip(rows, adjusted, strict=True):
        s["p_holm"] = padj
    return rows


def render_table(rows: list[tuple[str, dict[str, float]]], label_a: str, label_b: str) -> str:
    if not rows:
        return "_(no overlapping metrics)_\n"
    out = [
        f"| Metric | {label_a} | {label_b} | Δ | 95% CI | p | p (Holm) | |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, s in rows:
        out.append(
            f"| `{name}` | {s['before']:.4f} | {s['after']:.4f} | {s['delta']:+.4f} | "
            f"[{s['lo']:+.4f}, {s['hi']:+.4f}] | {s['p']:.4f} | {s['p_holm']:.4f} | "
            f"{stars(s['p_holm'])} |"
        )
    return "\n".join(out) + "\n"


def describe(log: dict[str, Any], entry: dict[str, Any], path: str) -> str:
    cfg = log.get("config", {})
    anchor = cfg.get("anchor", {}) or {}
    epoch = entry.get("epoch")
    tag = "baseline (untrained)" if epoch == 0 else f"epoch {epoch}"
    return (
        f"`{Path(path).name}` {tag} — mode={cfg.get('mode')}, "
        f"anchor={anchor.get('mode', 'none')}, "
        f"data={Path(str(cfg.get('data', {}).get('train_data', '?'))).name}, "
        f"seed={log.get('seed')}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "logs", nargs="+", help="one training_log.json (epoch 0 vs best), or two (arm vs arm)"
    )
    ap.add_argument(
        "--a-epoch",
        type=int,
        default=None,
        help="epoch from the first log (default: 0 for one log, best for two)",
    )
    ap.add_argument(
        "--b-epoch",
        type=int,
        default=None,
        help="epoch from the second log (default: that run's best)",
    )
    ap.add_argument(
        "--n-boot",
        type=int,
        default=10000,
        help="bootstrap / permutation resamples (default 10000)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--swap",
        action="store_true",
        help="pair persona-matched retrieval against the rotated-persona control "
        "within one log (needs eval.persona_swap enabled for that run)",
    )
    ap.add_argument("--no-personas", action="store_true", help="skip the per-persona breakdown")
    ap.add_argument("--out", type=Path, default=None, help="also write the Markdown to this path")
    args = ap.parse_args()

    if len(args.logs) > 2:
        ap.error("compare at most two logs")

    if args.swap and len(args.logs) != 1:
        ap.error("--swap compares matched vs swapped inside ONE log; pass a single log")

    single = len(args.logs) == 1
    path_a = args.logs[0]
    path_b = args.logs[0] if single else args.logs[1]
    log_a = load_log(path_a)
    log_b = log_a if single else load_log(path_b)

    # One log means "did training beat the untrained encoder?", which is the question
    # the epoch-0 row exists to answer. Two logs means "did arm B beat arm A?".
    a_epoch = args.a_epoch if args.a_epoch is not None else (0 if single else None)
    entry_a = pick_entry(log_a, path_a, a_epoch)
    entry_b = pick_entry(log_b, path_b, args.b_epoch)

    if entry_a is entry_b:
        ap.error("the two selected entries are the same epoch of the same log")

    if args.swap:
        # Matched vs rotated-persona, same epoch, same rows, same corpus embedding. The
        # delta is the personalisation signal in isolation: everything except the
        # `Instruct:` prefix is held fixed, so a null result here means the encoder is
        # not reading the persona at all and every headline gain is generic retrieval.
        entry_a = pick_entry(log_a, path_a, args.a_epoch if args.a_epoch is not None else None)
        swap = entry_a.get("persona_swap")
        if not swap:
            raise ValueError(
                f"{path_a}: epoch {entry_a.get('epoch')} has no persona_swap block. That "
                "run predates the control, or was configured with eval.persona_swap: false."
            )
        entry_b = {"epoch": entry_a.get("epoch"), "per_query": swap["per_query"]}
        label_a = describe(log_a, entry_a, path_a) + " — persona-MATCHED"
        label_b = describe(log_a, entry_a, path_a) + " — persona-SWAPPED (rotated)"
    else:
        label_a = describe(log_a, entry_a, path_a)
        label_b = describe(log_b, entry_b, path_b)
    check_alignment(entry_a, entry_b, label_a, label_b)

    lines = [
        "# Paired comparison",
        "",
        f"- **A:** {label_a}",
        f"- **B:** {label_b}",
        f"- {entry_a['per_query']['persona_ids'].__len__()} paired queries, "
        f"{args.n_boot} resamples, seed {args.seed}",
        "- Δ = B − A. CI is a bootstrap percentile interval; p is a sign-flip permutation",
        "  test, Holm-corrected across the metric family. Stars use the Holm value.",
        "",
    ]

    if not single and not args.swap:
        # Two arms trained on the same val set must agree exactly at epoch 0 — the
        # adapter is the identity there. A mismatch means something leaked into the
        # eval path (different corpus, different val file, changed relevance rule),
        # and every downstream comparison between these two runs is void.
        try:
            base_a = pick_entry(log_a, path_a, 0)
            base_b = pick_entry(log_b, path_b, 0)
            same = base_a["per_query"] == base_b["per_query"]
            lines.append(
                "- **Baseline parity:** epoch-0 vectors are identical ✅"
                if same
                else "- **Baseline parity: FAILED ❌** — the two runs' untrained epoch-0 "
                "evaluations differ, so their eval paths are not the same. Do not "
                "compare these runs until that is explained."
            )
            lines.append("")
        except ValueError as exc:
            lines += [f"- Baseline parity: not checkable ({exc})", ""]

    for key in SCALAR_KEYS:
        if not args.swap and key in entry_a and key in entry_b and key != "epoch":
            note = ""
            if key == "val_loss" and not single:
                data_a = Path(str(log_a.get("config", {}).get("data", {}).get("train_data", ""))).name
                data_b = Path(str(log_b.get("config", {}).get("data", {}).get("train_data", ""))).name
                if data_a != data_b:
                    note = " — **not comparable across arms** (the runs score different val sets)"
            lines.append(f"- `{key}`: {entry_a[key]:.4f} → {entry_b[key]:.4f}{note}")
    lines.append("")

    lines += [
        "## Overall",
        "",
        render_table(compare(entry_a, entry_b, None, args.n_boot, args.seed), "A", "B"),
    ]

    if not args.no_personas:
        for persona in sorted(set(entry_a["per_query"]["persona_ids"])):
            rows = compare(entry_a, entry_b, persona, args.n_boot, args.seed)
            n = rows[0][1]["n"] if rows else 0
            lines += [f"## Persona: {persona} (n = {n})", "", render_table(rows, "A", "B")]

    report = "\n".join(lines)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        # These are the alignment and provenance guards, not bugs. A traceback would
        # bury the one sentence that says which runs cannot be compared and why.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
