# Print the training-cost table of \cref{subsec:res-cost}

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


# Seconds of one epoch, per log and model, both protocols pooled
def seconds(grid: pd.DataFrame) -> pd.DataFrame:
    return (grid.groupby(["dataset", "variant"], observed=True)
            ["secs_per_epoch"].median().unstack("variant")[list(VARIANTS)]
            .reindex(list(DATASETS)).T)


# Median of ``column`` against the baseline of the same log and protocol
def ratio(grid: pd.DataFrame, column: str) -> pd.Series:
    table = (grid.groupby(["protocol", "dataset", "variant"], observed=True)
             [column].median().unstack("variant"))[list(VARIANTS)]
    return table.div(table["baseline"], axis=0).median()


# Number of epochs a run actually goes through, median over the cells
def epochs(grid: pd.DataFrame) -> pd.Series:
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
