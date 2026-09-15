"""Recompute the summary table of \\cref{sec:exp-results} from the grids.

Every model against the baseline, on the four metrics, under both protocols.
The chapter shows the levels in the figures, which keep the logs in their own
panels; what a table can carry is the difference against the baseline, because
a difference is comparable across logs where a level is not.

A cell is one log at one noise level. Inside the cell the five seeds are
reduced to their median before anything else touches the number, and the model
is paired against the baseline of that same cell. Four logs and nine levels
make thirty-six cells per protocol.

Two numbers per model and metric. The percentage is the mean difference over
the cells, relative to the mean of the baseline over the same cells; it keeps
the sign of the metric, so on the two conformance metrics a negative percentage
is an improvement. The fraction counts the cells in which the model is at least
as good as the baseline. A tie counts for the model: on a conformance metric a
pair is often equal, either at zero or at the same non-zero rate, and charging
those cells to the model would understate one that never loses.

    python official_experiments/scripts/summary_table.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "official_experiments" / "all_grids.csv"

#: The protocol letters the runs were launched with, and the names the thesis
#: gives them. On disk they are still ``runs/noise_curve_b`` and ``_c``.
PROTOCOLS = {"B": "test", "C": "train"}

#: CSV key -> the name of the model in the thesis, in the order of the chapter:
#: the four knowledge channels that act on the loss, then the three that act on
#: the input. The baseline is the reference and has no row of its own.
VARIANTS = {
    "lll": "Local loss",
    "gll": "Global loss",
    "checker_net": "Projected net",
    "checker_net_state": "State-indexed net",
    "marking": "Marking",
    "gnn": "Graph encoder",
    "seq": "Marking sequence",
}

#: Metric -> the name of the column and the direction of improvement, ``+1``
#: where a higher value is better and ``-1`` where a lower one is.
METRICS = {
    "accuracy": ("Accuracy", +1),
    "forbidden_net": ("Forbidden mass", -1),
    "dl_similarity": ("DL similarity", +1),
    "suffix_dfa_violation_net": ("Suffix violations", -1),
}


def cells(grid: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Median over the seeds, one row per protocol, log, model and level."""
    return (grid.groupby(["protocol", "dataset", "noise", "variant"],
                         observed=True)[metric].median()
            .unstack("variant"))


def compare(grid: pd.DataFrame) -> pd.DataFrame:
    """The percentage and the count, per protocol, model and metric."""
    rows = {}
    for metric, (column, direction) in METRICS.items():
        paired = cells(grid, metric)
        for letter, protocol in PROTOCOLS.items():
            panel = paired.xs(letter, level="protocol")
            base = panel["baseline"]
            for key, name in VARIANTS.items():
                delta = panel[key] - base
                relative = 100 * delta.mean() / abs(base.mean())
                ok = int((direction * delta >= 0).sum())
                rows.setdefault((protocol, name), {})[column] = (
                    f"{relative:+.1f}%  {ok}/{len(delta)}")

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index = pd.MultiIndex.from_tuples(table.index,
                                            names=["knowledge", "variant"])
    return table.reindex(pd.MultiIndex.from_product(
        [list(PROTOCOLS.values()), list(VARIANTS.values())],
        names=["knowledge", "variant"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    grid = pd.read_csv(GRID)
    missing = sorted((set(VARIANTS) | {"baseline"}) - set(grid["variant"]))
    if missing:
        raise SystemExit(f"the grid has no rows for {missing}")

    print("mean difference against the baseline, and the cells in which the "
          "model is at least as good")
    print(f"logs present: {', '.join(sorted(grid['dataset'].unique()))}")
    print(f"seeds present: {sorted(grid['seed'].unique())}")
    print()
    print(compare(grid).to_string())


if __name__ == "__main__":
    main()
