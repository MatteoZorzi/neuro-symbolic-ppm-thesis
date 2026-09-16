# Recompute the summary table of \cref{sec:exp-results} from the grids

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "official_experiments" / "all_grids.csv"

#: The protocol letters the published grids were launched with, and the names
#: the thesis gives them; ``noise_curve.py`` now takes the names.
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


# Median over the seeds, one row per protocol, log, model and level
def cells(grid: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (grid.groupby(["protocol", "dataset", "noise", "variant"],
                         observed=True)[metric].median()
            .unstack("variant"))


# The percentage and the count, per protocol, model and metric
def compare(grid: pd.DataFrame) -> pd.DataFrame:
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
    parser = argparse.ArgumentParser(
        description="Recompute the summary table of the results chapter from the grids.")
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
