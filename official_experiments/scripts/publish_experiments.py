"""Rebuild ``all_grids.csv`` from the eight grids, and check them while doing it.

``official_experiments/`` holds the results the thesis reports, and it holds
them twice. The eight files under ``protocol-test/`` and ``protocol-train/`` are
the source of truth: one grid per file, named after the log and the source of
the knowledge, so that a reader can open one and know what is in it.
``all_grids.csv`` is the same rows concatenated, and exists because every script
in this directory wants one table rather than eight.

This script derives the second from the first. It is not how the directory was
originally filled -- that was done once, from the job output on the cluster,
through a staging copy that is no longer kept: ``runs/`` is 800 MB, is not under
version control and will never be published, so a rebuild path that started
there could not be run by anyone reading the thesis. It starts here instead,
which means the eight grids are data and not a derivation.

A grid is a pair (log, knowledge source): 9 variants x 9 noise levels x 10
seeds, 810 cells. Three checks run before anything is written, and if one fails
nothing is:

* the cell count, so that a grid that lost rows to a cancelled job is caught;
* no duplicate key, so that a grid merged twice is caught;
* one accelerator per grid. This is not pedantry. Between a V100 and an A100
  there is a factor of 1.74 on the training seconds, and a grid straddling two
  cards would carry ``secs_`` columns that mean nothing.

    python official_experiments/scripts/publish_experiments.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import GRID_CSV, PROTOCOL_DIR, ROOT

#: The order the rows are concatenated in, which is the order ``all_grids.csv``
#: has always had. It is **not** :data:`common.DATASET_ORDER`: that one puts
#: BPIC 2013 before BPIC 2020 because that is how the panels read across a
#: figure, and this one is the order the grids were merged in. Nothing depends
#: on either, since every script groups by ``dataset`` before it reads a number,
#: but keeping this one makes a rebuilt file comparable to the published one
#: byte for byte, which is the only way to tell a rebuild from a change.
GRID_ORDER = ("Sepsis_Case", "BPIC_2020_DomesticDeclarations",
              "BPIC_2013_incidents", "BPI_Challenge_2012")

#: The protocol letters the runs were launched with. The knowledge source is
#: what they differ in, and it is what the directory names say; the letters
#: survive in the ``protocol`` column of every row.
PROTOCOLS = ("B", "C")

KEY = ["dataset", "variant", "noise", "seed"]


def read_grid(root: Path, dataset: str, protocol: str) -> pd.DataFrame:
    path = root / PROTOCOL_DIR[protocol] / dataset / "results.csv"
    if not path.exists():
        raise SystemExit(f"missing grid: {path.relative_to(ROOT)}")
    return pd.read_csv(path)


def check(grid: pd.DataFrame, dataset: str, protocol: str, expected: int) -> None:
    cards = sorted(str(c) for c in grid.gpu.unique())
    duplicates = int(grid.duplicated(KEY).sum())

    problems = []
    if len(grid) != expected:
        problems.append(f"{len(grid)} cells instead of {expected}")
    if duplicates:
        problems.append(f"{duplicates} duplicate keys")
    if len(cards) != 1:
        problems.append(f"more than one accelerator: {cards}")

    print(f"{PROTOCOL_DIR[protocol]:15s} {dataset:32s} {len(grid):4d} cells  "
          f"seeds {sorted(grid.seed.unique())}  {cards[0]}"
          f"  {'ok' if not problems else '<-- ' + '; '.join(problems)}")
    if problems:
        raise SystemExit("grid does not conform, nothing written")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "official_experiments",
                        help="directory holding protocol-test and protocol-train")
    parser.add_argument("--out", type=Path, default=GRID_CSV)
    parser.add_argument("--seeds", type=int, default=10, help="seeds per grid")
    args = parser.parse_args()

    expected = 9 * 9 * args.seeds
    grids = []
    # Log by log, and inside a log the test protocol before the train one.
    for dataset in GRID_ORDER:
        for protocol in PROTOCOLS:
            grid = read_grid(args.root, dataset, protocol)
            check(grid, dataset, protocol, expected)
            grids.append(grid)

    everything = pd.concat(grids, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    everything.to_csv(args.out, index=False)
    print(f"\nwrote {args.out.relative_to(ROOT)}: {len(everything)} rows, "
          f"{everything.groupby(['dataset', 'protocol']).ngroups} grids")


if __name__ == "__main__":
    main()
