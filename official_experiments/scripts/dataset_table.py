# Recompute ``tab:exp-datasets`` of Chapter 5 from the event logs themselves

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from matrix import find_log  # noqa: E402
from nspm.data.loader import ACTIVITY, CASE_ID, read_csv, read_xes  # noqa: E402

#: The four logs of the thesis, in the order the table lists them: by increasing
#: number of cases. The key is the folder name under ``datasets/``, the same one
#: the ``dataset`` column carries in the result CSVs. BPIC15 is left out because
#: it is left out of the table too.
DATASETS = (
    ("Sepsis_Case", "Sepsis Cases"),
    ("BPIC_2013_incidents", "BPIC 2013 incidents"),
    ("BPIC_2020_DomesticDeclarations", "BPIC 2020 dom. declarations"),
    ("BPI_Challenge_2012", "BPIC 2012"),
)


def read_log(dataset: str) -> pd.DataFrame:
    path = find_log(dataset)
    reader = read_csv if path.suffix.lower().lstrip(".").startswith("csv") else read_xes
    return reader(path)


def measure(dataset: str, label: str) -> dict:
    events = read_log(dataset)
    traces = events.groupby(CASE_ID, sort=False)[ACTIVITY]
    lengths = traces.size()
    return {
        "log": label,
        "cases": int(events[CASE_ID].nunique()),
        "events": int(len(events)),
        "activities": int(events[ACTIVITY].nunique()),
        "min": int(lengths.min()),
        "median": int(lengths.median()),
        "max": int(lengths.max()),
        "variants": int(traces.apply(tuple).nunique()),
    }


def main() -> None:
    table = pd.DataFrame([measure(dataset, label) for dataset, label in DATASETS])
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
