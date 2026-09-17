# Draw the training-cost figure of \cref{subsec:res-cost}

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common import (AXIS, COLORS, FIGURES, GRID, GRID_CSV, INK,  # noqa: E402
                    INK_MUTED, INK_SECONDARY, SURFACE, VARIANTS,
                    copy_to_thesis, shown)

#: Where the name of each model goes, in data units, and how it is anchored.
#: Four of the eight points stand on x = 1 and no automatic placement keeps
#: their labels apart, so the offsets are set by hand and checked in the render.
LABELS = {
    "lll": (0.045, 0, "left"),
    "gll": (-0.05, -4, "right"),
    "checker_net": (0.035, 4, "left"),
    "checker_net_state": (-0.02, 0, "right"),
    "marking": (0.022, 6, "left"),
    "gnn": (0.035, -9, "left"),
    "seq": (0.0, 11, "center"),
    "baseline": (-0.035, -8, "right"),
}

#: The thesis figure pools the two protocols; one protocol alone can be asked
#: for by name. The grids keep the letters.
LETTER = {"test": "B", "train": "C"}

#: Height is a page budget, not a taste. The subsection holds a table, this
#: figure and its two captions inside one page, and anything taller pushes the
#: closing paragraph onto the next one.
FIGSIZE = (5.7, 2.75)

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


# Time of one epoch against the baseline, median over log and protocol
def cost(grid: pd.DataFrame) -> pd.Series:
    epoch = (grid.groupby(["protocol", "dataset", "variant"], observed=True)
             ["secs_per_epoch"].median().unstack("variant"))[list(VARIANTS)]
    return epoch.div(epoch["baseline"], axis=0).median()


# Percent of the baseline's ``metric`` that a model removes
def removed(grid: pd.DataFrame, metric: str, protocol: str) -> pd.Series:
    cells = (grid.groupby(["protocol", "dataset", "noise", "variant"],
                          observed=True)[metric].median().unstack("variant"))
    panel = cells if protocol == "both" else cells.xs(LETTER[protocol], level="protocol")
    base = panel["baseline"]
    return pd.Series({v: -100 * (panel[v] - base).mean() / abs(base.mean())
                      for v in VARIANTS})


def draw(grid: pd.DataFrame, path: Path, protocol: str) -> None:
    plt.rcParams.update(STYLE)
    fig = plt.figure(figsize=FIGSIZE)
    axis = fig.add_axes([0.105, 0.135, 0.875, 0.700])

    time = cost(grid)
    mass = removed(grid, "forbidden_net", protocol)
    violations = removed(grid, "suffix_dfa_violation_net", protocol)

    axis.axhline(0, color=AXIS, linewidth=0.8, linestyle=(0, (4, 2)), zorder=1)
    for key in VARIANTS:
        x = time[key]
        # The segment is the one thing two separate panels could not say: the
        # same model, the same price, two different answers.
        axis.plot([x, x], [mass[key], violations[key]], color=COLORS[key],
                  linewidth=0.9, alpha=0.5, zorder=2)
        axis.scatter([x], [violations[key]], s=34, facecolor=SURFACE,
                     edgecolor=COLORS[key], linewidth=1.2, zorder=3)
        axis.scatter([x], [mass[key]], s=34, color=COLORS[key], zorder=4,
                     edgecolor=SURFACE, linewidth=0.7)
        dx, dy, align = LABELS[key]
        axis.text(x + dx, mass[key] + dy, VARIANTS[key], fontsize=6.8,
                  color=INK_SECONDARY, va="center", ha=align, zorder=5)

    axis.set_xlim(0.82, 2.62)
    axis.set_ylim(-38, 108)
    axis.set_xlabel("time of one epoch, baseline $= 1$", fontsize=7.5,
                    color=INK_SECONDARY, labelpad=3)
    axis.set_ylabel("percent of the baseline removed", fontsize=7.5,
                    color=INK_SECONDARY, labelpad=4)
    axis.grid(color=GRID, linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)

    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=5,
               markerfacecolor=INK_MUTED, markeredgecolor=SURFACE,
               label="forbidden probability mass"),
        Line2D([], [], marker="o", linestyle="none", markersize=5,
               markerfacecolor=SURFACE, markeredgecolor=INK_MUTED,
               markeredgewidth=1.2, label="suffix violation rate"),
    ]
    fig.text(0.5, 0.985, "What the training time buys", ha="center", va="top",
             fontsize=10, color=INK)
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               fontsize=7.5, labelcolor=INK_SECONDARY, handlelength=1.2,
               columnspacing=1.8, bbox_to_anchor=(0.5, 0.935))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw the training-cost figure of the results chapter.")
    parser.add_argument("--protocol", choices=["both", *sorted(LETTER)], default="both",
                        help="the protocol the reductions are taken from; "
                             "'both' pools the two, which is what the thesis shows")
    parser.add_argument("--out", type=Path, default=FIGURES)
    parser.add_argument("--no-thesis-copy", action="store_true",
                        help="do not mirror the figure into the thesis")
    args = parser.parse_args()

    grid = pd.read_csv(GRID_CSV)
    grid = grid[grid["variant"].isin(VARIANTS)]

    print(f"the coordinates of the eight points, {args.protocol} protocol:")
    print(pd.DataFrame({
        "time of one epoch": cost(grid),
        "forbidden mass removed": removed(grid, "forbidden_net", args.protocol),
        "suffix violations removed": removed(grid, "suffix_dfa_violation_net", args.protocol),
    }).rename(index=VARIANTS).round(2).to_string())

    name = "training-cost" if args.protocol == "both" else f"training-cost_{args.protocol}"
    path = args.out / f"{name}.png"
    draw(grid, path, args.protocol)
    print(f"\nwrote {shown(path)}")

    if not args.no_thesis_copy:
        print(f"  mirrored to {shown(copy_to_thesis(path))}")


if __name__ == "__main__":
    main()
