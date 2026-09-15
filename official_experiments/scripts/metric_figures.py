"""Draw the per-metric noise curves of \\cref{sec:exp-results}, one per protocol.

The chapter presents each metric once, for all eight models, and the table that
goes with it averages the logs away. The figure is where the logs come back: a
panel each, the noise on the x axis, one line per model. What the reader is
meant to compare between panels is the *shape* of the curves, not their level,
so the y axis is free in every panel.

Test and train are two figures and never two halves of one. Between the two
protocols the mined net changes, so the mask the metric is measured against
changes with it, and two curves drawn side by side would invite a comparison of
levels that the numbers do not support.

The heading puts the title on the first line, the legend on the second and the
panels below. The palette and the markers come from :mod:`common`, so that a
model keeps its colour across every figure in the thesis.

The value of a point is the median over the five seeds. Seeds are aggregated
inside the cell, before anything else touches the number.

    python official_experiments/scripts/metric_figures.py
    python official_experiments/scripts/metric_figures.py --metric accuracy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import (AXIS, COLORS, DATASET_ORDER, FIGURES, GRID, GRID_CSV,  # noqa: E402
                    INK, INK_MUTED, INK_SECONDARY, MARKERS, PROTOCOLS, ROOT,
                    SHORT, SURFACE, VARIANTS, copy_to_thesis)

#: Metric -> the label of the y axis. Only the four the chapter reports.
AXIS_LABEL = {
    "forbidden_net": "forbidden probability mass",
    "suffix_dfa_violation_net": "violations per generated step",
    "accuracy": "next-activity accuracy",
    "dl_similarity": "Damerau-Levenshtein similarity",
}

#: Metric -> the heading of the figure. It names the metric in the words the
#: chapter uses for it, and the protocol is appended when the figure is drawn.
TITLE = {
    "forbidden_net": "Forbidden probability mass",
    "suffix_dfa_violation_net": "Suffix violation rate",
    "accuracy": "Next-activity accuracy",
    "dl_similarity": "Suffix similarity",
}

#: Designed to sit at ``\textwidth`` without being scaled: the tick labels are
#: 7pt in the file and 7pt on the page. A figure drawn wide and shrunk by
#: ``includegraphics`` renders its text at whatever the shrink leaves.
FIGSIZE = (5.7, 4.4)

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


def curves(grid: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Median over the seeds, per log, model and noise level."""
    return (grid[grid["variant"].isin(VARIANTS)]
            .groupby(["dataset", "variant", "noise"], observed=True)[metric]
            .median().reset_index())


def draw(table: pd.DataFrame, metric: str, protocol: str, path: Path) -> None:
    logs = [d for d in DATASET_ORDER if d in set(table["dataset"])]
    noises = sorted(table["noise"].unique())

    plt.rcParams.update(STYLE)
    fig, grid_axes = plt.subplots(2, 2, figsize=FIGSIZE)
    axes = list(grid_axes.flat)

    for axis, dataset in zip(axes, logs):
        panel = table[table["dataset"] == dataset]
        for key in VARIANTS:
            series = panel[panel["variant"] == key].sort_values("noise")
            if series.empty:
                continue
            # The baseline is dashed and drawn last. It is the reference every
            # other line is read against, and on this metric three of the other
            # lines land on top of it: solid and underneath, it disappears
            # exactly where the reader most needs to see it.
            reference = key == "baseline"
            axis.plot(series["noise"], series[metric], color=COLORS[key],
                      linewidth=1.8 if reference else 1.4,
                      linestyle=(0, (4, 2)) if reference else "-",
                      marker=MARKERS[key], markersize=3.4,
                      markeredgecolor=SURFACE, markeredgewidth=0.7,
                      label=VARIANTS[key], zorder=9 if reference else 3)

        axis.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        axis.set_axisbelow(True)
        axis.set_xticks(noises[::2])
        axis.set_xticklabels([f"{n:.0%}" for n in noises[::2]])
        axis.set_xlim(min(noises) - 0.02, max(noises) + 0.02)
        # The metric is a proportion, so zero is a real floor and the automatic
        # padding has no reason to go below it.
        if panel[metric].min() >= 0 and axis.get_ylim()[0] < 0:
            axis.set_ylim(0, axis.get_ylim()[1])
        axis.set_title(SHORT.get(dataset, dataset), fontsize=8, color=INK,
                       pad=4)

    # Heading of the figure: the title on the first line, the legend on the
    # second. Both belong to the whole figure rather than to any one panel, so
    # they sit above all four and the panels start below them.
    handles, labels = axes[0].get_legend_handles_labels()
    for axis in axes[len(logs):]:
        axis.set_axis_off()

    fig.text(0.5, 0.985, f"{TITLE[metric]}, {protocol} protocol", ha="center",
             va="top", fontsize=10, color=INK)
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=7.5, labelcolor=INK_SECONDARY, handlelength=1.8,
               columnspacing=1.4, bbox_to_anchor=(0.5, 0.945))

    fig.supxlabel("fraction of training events with a corrupted label",
                  fontsize=7.5, color=INK_SECONDARY, y=0.018)
    fig.supylabel(AXIS_LABEL[metric], fontsize=7.5, color=INK_SECONDARY, x=0.01)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.795, bottom=0.105,
                        hspace=0.40, wspace=0.26)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


#: Height of the paired layout. Four rows of panels instead of two, but one
#: heading and one caption instead of two of each.
FIGSIZE_PAIRED = (5.7, 6.6)


def draw_paired(tables: dict[str, pd.DataFrame], metric: str, path: Path
                ) -> None:
    """One figure for both protocols: logs down the rows, protocols across.

    The alternative to two stacked figures. Panels keep the width they have in
    the stacked layout, because two columns of a full-width figure are as wide
    as two columns of a half-height one, and the page saves a heading, a caption
    and the white space between two floats.

    The y axis stays free in every panel, here as there. Two panels in a row
    share a log but not a mined net, so their levels are no more comparable side
    by side than they were on facing pages.
    """
    protocols = list(tables)
    logs = [d for d in DATASET_ORDER
            if any(d in set(t["dataset"]) for t in tables.values())]
    noises = sorted(next(iter(tables.values()))["noise"].unique())

    plt.rcParams.update(STYLE)
    fig, grid_axes = plt.subplots(len(logs), len(protocols),
                                  figsize=FIGSIZE_PAIRED)

    for row, dataset in enumerate(logs):
        for column, protocol in enumerate(protocols):
            axis = grid_axes[row][column]
            panel = tables[protocol]
            panel = panel[panel["dataset"] == dataset]
            for key in VARIANTS:
                series = panel[panel["variant"] == key].sort_values("noise")
                if series.empty:
                    continue
                reference = key == "baseline"
                axis.plot(series["noise"], series[metric], color=COLORS[key],
                          linewidth=1.8 if reference else 1.4,
                          linestyle=(0, (4, 2)) if reference else "-",
                          marker=MARKERS[key], markersize=3.4,
                          markeredgecolor=SURFACE, markeredgewidth=0.7,
                          label=VARIANTS[key], zorder=9 if reference else 3)

            axis.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
            axis.set_axisbelow(True)
            axis.set_xticks(noises[::2])
            axis.set_xticklabels([f"{n:.0%}" for n in noises[::2]])
            axis.set_xlim(min(noises) - 0.02, max(noises) + 0.02)
            if panel[metric].min() >= 0 and axis.get_ylim()[0] < 0:
                axis.set_ylim(0, axis.get_ylim()[1])

            # The protocol names the column once, at the top. The log names the
            # row once, on the left, so that neither is repeated eight times.
            if row == 0:
                axis.set_title(f"{protocol} protocol", fontsize=8.5, color=INK,
                               pad=6)
            if column == 0:
                axis.set_ylabel(SHORT.get(dataset, dataset), fontsize=8,
                                color=INK, labelpad=6)

    handles, labels = grid_axes[0][0].get_legend_handles_labels()
    fig.text(0.5, 0.99, TITLE[metric], ha="center", va="top", fontsize=10,
             color=INK)
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=7.5, labelcolor=INK_SECONDARY, handlelength=1.8,
               columnspacing=1.4, bbox_to_anchor=(0.5, 0.962))

    fig.supxlabel("fraction of training events with a corrupted label",
                  fontsize=7.5, color=INK_SECONDARY, y=0.012)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.862, bottom=0.068,
                        hspace=0.45, wspace=0.20)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="forbidden_net",
                        choices=sorted(AXIS_LABEL))
    parser.add_argument("--out", type=Path, default=FIGURES)
    parser.add_argument("--no-thesis-copy", action="store_true",
                        help="do not mirror the figures into the thesis")
    parser.add_argument("--paired", action="store_true",
                        help="one figure with the protocols side by side, "
                             "instead of one figure each")
    args = parser.parse_args()

    grid = pd.read_csv(GRID_CSV)
    print(f"logs present: {', '.join(sorted(grid['dataset'].unique()))}")
    print(f"seeds present: {sorted(grid['seed'].unique())}")

    name = args.metric.replace("_", "-")

    if args.paired:
        tables = {protocol: curves(grid[grid["protocol"] == letter],
                                   args.metric)
                  for letter, protocol in PROTOCOLS.items()}
        path = args.out / f"{name}-paired.png"
        draw_paired(tables, args.metric, path)
        print(f"wrote {path.relative_to(ROOT)}")
        if not args.no_thesis_copy:
            print(f"  mirrored to {copy_to_thesis(path).relative_to(ROOT)}")
        return

    for letter, protocol in PROTOCOLS.items():
        table = curves(grid[grid["protocol"] == letter], args.metric)
        path = args.out / f"{name}-{protocol}.png"
        draw(table, args.metric, protocol, path)
        print(f"wrote {path.relative_to(ROOT)}")

        if not args.no_thesis_copy:
            print(f"  mirrored to {copy_to_thesis(path).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
