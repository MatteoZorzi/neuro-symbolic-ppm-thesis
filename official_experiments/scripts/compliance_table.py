# Recompute ``tab:exp-compliance`` of Chapter 5 from the result grids

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

GRID = ROOT / "official_experiments" / "all_grids.csv"

#: The grid keeps the old protocol letters; the chapter names them by the
#: partition the knowledge is mined from.
PROTOCOL_OF_LETTER = {"B": "test", "C": "train"}
PROTOCOLS = ("test", "train")

#: Logs in the order the chapter lists them, with the labels the table prints.
DATASETS = {
    "Sepsis_Case": "Sepsis Cases",
    "BPIC_2013_incidents": "BPIC 2013 incidents",
    "BPIC_2020_DomesticDeclarations": "BPIC 2020 dom. dec.",
    "BPI_Challenge_2012": "BPIC 2012",
}

#: Three of the nine levels, the ones the table shows.
NOISES = (0.0, 0.4, 0.8)


# Collapse the variants of a cell, refusing to do so if they disagree
def one_value_per_cell(frame: pd.DataFrame, keys: list[str]) -> pd.Series:
    spread = frame.groupby(keys)["train_compliance"].nunique()
    disagreeing = spread[spread > 1]
    if not disagreeing.empty:
        raise SystemExit(
            "train_compliance is not constant across variants; it cannot be a "
            f"property of the data alone:\n{disagreeing}")
    return frame.groupby(keys)["train_compliance"].first()


# One value per (log, protocol, noise, seed), with the variants collapsed
def per_seed() -> pd.Series:
    grid = pd.read_csv(GRID)
    grid["protocol"] = grid["protocol"].map(PROTOCOL_OF_LETTER)
    return one_value_per_cell(grid, ["dataset", "protocol", "noise", "seed"])


# The chapter's layout: four logs down, protocol by noise across
def table(cells: pd.Series, how: str) -> pd.DataFrame:
    by_cell = cells.groupby(["dataset", "protocol", "noise"])
    agg = by_cell.std(ddof=0) if how == "std" else by_cell.mean()
    columns = {
        (protocol, f"{noise:.0%}"): {
            label: agg.get((dataset, protocol, noise))
            for dataset, label in DATASETS.items()
        }
        for protocol in PROTOCOLS
        for noise in NOISES
    }
    out = pd.DataFrame(columns)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out.loc[list(DATASETS.values())].round(3)


def main() -> None:
    cells = per_seed()
    seeds = sorted(cells.index.get_level_values("seed").unique())
    print(f"seeds present: {seeds}")
    print()
    print("As printed in the thesis, the mean over the seeds:")
    print(table(cells, "mean").to_string())
    print()
    print("Population spread across the seeds, not carried into the table:")
    print(table(cells, "std").to_string())
    print()
    print(f"largest spread in any cell: "
          f"{cells.groupby(['dataset', 'protocol', 'noise']).std(ddof=0).max():.4f}")


if __name__ == "__main__":
    main()
