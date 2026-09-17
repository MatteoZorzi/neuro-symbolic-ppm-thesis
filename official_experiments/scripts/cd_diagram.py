# Draw the critical difference diagram of one metric, one panel per protocol
#
# Every run of the grid is one row: a log, a noise level and a seed. The eight
# models are the columns. The models are ranked inside each row, the ranks are
# averaged over the rows, and the Friedman test and the Nemenyi critical
# difference (Demsar, 2006) say which average ranks can be told apart. Models
# joined by a bar are not significantly different at alpha = 0.05.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import friedmanchisquare, rankdata  # noqa: E402

from common import (FIGURES, GRID_CSV, INK, INK_SECONDARY, PROTOCOLS,  # noqa: E402
                    SURFACE, VARIANTS, copy_to_thesis, save_figure, shown)

#: Nemenyi q_alpha at alpha = 0.05 (studentized range over sqrt(2)), by the
#: number of models compared.
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
       9: 3.102, 10: 3.164}

#: Metric -> (heading, whether a lower value is better).
METRICS = {
    "forbidden_net": ("Forbidden probability mass", True),
    "suffix_dfa_violation_net": ("Suffix violation rate", True),
    "accuracy": ("Next-activity accuracy", False),
    "dl_similarity": ("Suffix similarity", False),
    "secs_train": ("Training time of a run", True),
}

#: A run is one log at one noise level under one seed.
RUN_KEYS = ["dataset", "noise", "seed"]


# Runs down the rows, models across the columns, only complete runs
def run_matrix(grid: pd.DataFrame, metric: str) -> pd.DataFrame:
    table = (grid[grid["variant"].isin(VARIANTS)]
             .pivot_table(index=RUN_KEYS, columns="variant", values=metric,
                          aggfunc="first"))
    return table[list(VARIANTS)].dropna()


# Average ranks, Friedman p-value and Nemenyi critical difference
def rank_statistics(matrix: pd.DataFrame, lower_is_better: bool
                    ) -> tuple[pd.Series, float, float]:
    data = matrix.to_numpy(float)
    rows, k = data.shape
    signed = data if lower_is_better else -data
    ranks = np.vstack([rankdata(row, method="average") for row in signed])
    average = pd.Series(ranks.mean(axis=0), index=matrix.columns)
    p_value = friedmanchisquare(*[data[:, i] for i in range(k)]).pvalue
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6.0 * rows))
    return average, p_value, cd


# One diagram on ``ax``: the rank axis, the CD bar, the models and the cliques
def draw_panel(ax: plt.Axes, average: pd.Series, cd: float, title: str) -> None:
    k = len(average)
    order = average.sort_values()
    lo, hi = 1, k
    ax.set_xlim(hi + 0.5, lo - 0.5)  # rank 1 on the right
    ax.set_ylim(k - (k + 1) // 2 - 0.4, k + 2.6)
    ax.axis("off")

    y0 = k + 1
    ax.plot([lo, hi], [y0, y0], color=INK, lw=1)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [y0, y0 + 0.12], color=INK, lw=1)
        ax.text(r, y0 + 0.25, str(r), ha="center", va="bottom", fontsize=8,
                color=INK_SECONDARY)

    ax.plot([lo, lo + cd], [y0 + 0.9, y0 + 0.9], color=INK, lw=2)
    ax.plot([lo, lo], [y0 + 0.82, y0 + 0.98], color=INK, lw=1)
    ax.plot([lo + cd, lo + cd], [y0 + 0.82, y0 + 0.98], color=INK, lw=1)
    ax.text(lo + cd / 2, y0 + 1.05, f"CD = {cd:.2f}", ha="center", va="bottom",
            fontsize=8, color=INK_SECONDARY)

    # The better half is labelled on the right, the worse half on the left.
    half = (k + 1) // 2
    for position, (variant, rank) in enumerate(order.items()):
        right = position < half
        level = position if right else k - 1 - position
        y = y0 - 1 - level
        x = lo - 0.3 if right else hi + 0.3
        ax.plot([rank, rank], [y0, y], color=INK, lw=0.8)
        ax.plot([rank, x], [y, y], color=INK, lw=0.8)
        ax.text(x - 0.15 if right else x + 0.15, y,
                f"{VARIANTS[variant]} ({rank:.2f})",
                ha="left" if right else "right", va="center", fontsize=8,
                color=INK)

    # Maximal cliques: runs of consecutive models within the CD of each other.
    ranks = order.to_numpy()
    intervals = []
    for i in range(k):
        j = i
        while j + 1 < k and ranks[j + 1] - ranks[i] <= cd:
            j += 1
        if j > i:
            intervals.append((i, j))
    intervals = [a for a in intervals
                 if not any(b != a and b[0] <= a[0] and b[1] >= a[1]
                            for b in intervals)]
    for level, (i, j) in enumerate(intervals):
        y = y0 - 0.55 - 0.22 * level
        ax.plot([ranks[i] - 0.05, ranks[j] + 0.05], [y, y], color=INK, lw=3.5,
                solid_capstyle="butt")

    ax.set_title(title, fontsize=9, color=INK, pad=4)


# A p-value as printed: below double precision it is only a bound
def format_p(p_value: float) -> str:
    return "p < 1e-16" if p_value < 1e-16 else f"p = {p_value:.1e}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw the critical difference diagram of one metric, "
                    "one panel per protocol.")
    parser.add_argument("--metric", default="forbidden_net",
                        choices=sorted(METRICS))
    parser.add_argument("--out", type=Path, default=FIGURES)
    parser.add_argument("--no-thesis-copy", action="store_true",
                        help="do not mirror the figure into the thesis")
    args = parser.parse_args()

    heading, lower_is_better = METRICS[args.metric]
    grid = pd.read_csv(GRID_CSV)

    fig, axes = plt.subplots(len(PROTOCOLS), 1, figsize=(6.3, 4.4))
    fig.patch.set_facecolor(SURFACE)
    for ax, (letter, protocol) in zip(axes, PROTOCOLS.items()):
        matrix = run_matrix(grid[grid["protocol"] == letter], args.metric)
        average, p_value, cd = rank_statistics(matrix, lower_is_better)
        print(f"{heading}, {protocol} protocol: {len(matrix)} runs, "
              f"{matrix.shape[1]} models, Friedman {format_p(p_value)}, "
              f"CD = {cd:.3f}")
        for variant, rank in average.sort_values().items():
            print(f"  {VARIANTS[variant]:<18} {rank:.3f}")
        ax.set_facecolor(SURFACE)
        draw_panel(ax, average, cd,
                   f"{protocol.capitalize()} protocol "
                   f"({len(matrix)} runs, Friedman {format_p(p_value)})")

    fig.tight_layout()
    path = args.out / f"{args.metric.replace('_', '-')}-cd.png"
    save_figure(fig, path)
    print(f"wrote {shown(path)}")
    if not args.no_thesis_copy:
        print(f"  mirrored to {shown(copy_to_thesis(path))}")


if __name__ == "__main__":
    main()
