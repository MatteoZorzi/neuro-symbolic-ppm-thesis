"""Print the training-cost table of \\cref{subsec:res-cost}.

One row per model. The four dataset columns are the seconds one epoch takes,
which is the only place in the thesis where the clock is read in seconds rather
than in multiples of the baseline: an epoch is 1.4 seconds on Sepsis and 20.7 on
BPIC 2012, and a ratio of 2.22 hides which of the two it is talking about.

The last two columns are ratios against the baseline of the same log, and they
do not say the same thing. ``per epoch`` is what the method costs. ``whole run``
is what it is actually paid, because early stopping fires at different epochs:
the global loss costs 2.23 an epoch and 2.62 a run, since it also trains longer,
while the local loss costs 1.01 an epoch and 1.00 a run only because on BPIC
2012 under the test protocol it stops at the first one and totals 0.43 of the
baseline -- which is not a saving but a model that settled.

Every entry is the median over the ten seeds and the nine noise levels; the
seconds pool the two protocols as well, which mine the same net from different
partitions and cost the same to train. The ratios are taken inside the log and
then averaged over the eight combinations of log and protocol, never across
logs: an epoch of BPIC 2012 is fifteen of Sepsis, and only the proportion
carries over.

    python official_experiments/scripts/cost_table.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GRID_CSV = ROOT / "official_experiments" / "all_grids.csv"

VARIANTS = {
    "baseline": "Baseline",
    "lll": "Local loss",
    "gll": "Global loss",
    "checker_net": "Projected net",
    "checker_net_state": "State-indexed net",
    "marking": "Marking",
    "gnn": "Graph encoder",
    "seq": "Marking sequence",
}

DATASETS = {
    "Sepsis_Case": "Sepsis",
    "BPIC_2013_incidents": "BPIC 2013",
    "BPIC_2020_DomesticDeclarations": "BPIC 2020",
    "BPI_Challenge_2012": "BPIC 2012",
}


def seconds(grid: pd.DataFrame) -> pd.DataFrame:
    """Seconds of one epoch, per log and model, both protocols pooled."""
    return (grid.groupby(["dataset", "variant"], observed=True)
            ["secs_per_epoch"].median().unstack("variant")[list(VARIANTS)]
            .reindex(list(DATASETS)).T)


def ratio(grid: pd.DataFrame, column: str) -> pd.Series:
    """Median of ``column`` against the baseline of the same log and protocol."""
    table = (grid.groupby(["protocol", "dataset", "variant"], observed=True)
             [column].median().unstack("variant"))[list(VARIANTS)]
    return table.div(table["baseline"], axis=0).median()


def epochs(grid: pd.DataFrame) -> pd.Series:
    """Number of epochs a run actually goes through, median over the cells.

    It is a count and not a ratio on purpose. The three columns would then read
    as a product, and medians do not multiply: the global loss is 2.23 an epoch
    and 1.12 in epochs, whose product is 2.49 and not the 2.62 of a whole run.

    ``best_epoch`` is the checkpoint that is kept, five epochs before the run
    ends, so it is the wrong count for a cost: what is paid is every epoch that
    was executed, patience included.
    """
    run = grid["secs_train"] / grid["secs_per_epoch"]
    table = (run.groupby([grid["protocol"], grid["dataset"], grid["variant"]],
                         observed=True).median().unstack("variant"))
    return table[list(VARIANTS)].median()


def main() -> None:
    grid = pd.read_csv(GRID_CSV)
    grid = grid[grid["variant"].isin(VARIANTS)]

    table = seconds(grid)
    table.columns = [DATASETS[d] for d in table.columns]
    table["per epoch"] = ratio(grid, "secs_per_epoch")
    table["whole run"] = ratio(grid, "secs_train")
    table["epochs"] = epochs(grid)
    table.index = [VARIANTS[v] for v in table.index]

    pd.set_option("display.width", 200)
    print("seconds of one epoch, the two ratios against the baseline, and the "
          "number of epochs a run goes through:")
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
