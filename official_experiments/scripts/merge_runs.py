# Merge the runs of one log into the grid published under protocol-test/ or protocol-train/

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import PROTOCOL_DIR, ROOT, shown

#: The grids keep the letters the runs were launched with.
LETTER = {"test": "B", "train": "C"}

KEY = ["dataset", "variant", "noise", "seed"]

#: One seed of a grid: nine methods at nine noise levels.
CELLS_PER_SEED = 9 * 9


# The rows of one log in one run, tagged with the protocol and the run they came from
def read_source(runs_root: Path, source: str, dataset: str, letter: str) -> pd.DataFrame:
    path = runs_root / source / "results.csv"
    if not path.exists():
        raise SystemExit(f"missing run: {shown(path)}")
    rows = pd.read_csv(path)
    rows = rows[rows["dataset"] == dataset].copy()
    rows["protocol"] = letter
    rows["source_dir"] = source
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge the runs of one log into its published grid, checking it on the way.")
    parser.add_argument("--protocol", choices=sorted(LETTER), required=True)
    parser.add_argument("--dataset", required=True, help="folder name under datasets/")
    parser.add_argument("--sources", nargs="+",
                        help="subdirectories of the runs root, in the order they are concatenated "
                             "(default: noise_curve_<protocol>)")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--out", type=Path,
                        help="default: official_experiments/protocol-<protocol>/<dataset>/results.csv")
    args = parser.parse_args()

    letter = LETTER[args.protocol]
    sources = args.sources or [f"noise_curve_{args.protocol}"]
    out = args.out or ROOT / "official_experiments" / PROTOCOL_DIR[letter] / args.dataset / "results.csv"

    parts = [read_source(args.runs_root, source, args.dataset, letter) for source in sources]
    for source, part in zip(sources, parts):
        print(f"{source:48s} {len(part):4d} rows  seeds {sorted(part['seed'].unique())}")
    grid = pd.concat(parts, ignore_index=True)

    seeds = grid["seed"].nunique()
    duplicates = int(grid.duplicated(KEY).sum())
    if duplicates or len(grid) != CELLS_PER_SEED * seeds:
        raise SystemExit(f"grid does not conform, nothing written: {len(grid)} rows for "
                         f"{seeds} seeds, {duplicates} duplicate keys")

    out.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out, index=False)
    print(f"\nwrote {shown(out)}: {len(grid)} rows, {seeds} seeds")


if __name__ == "__main__":
    main()
