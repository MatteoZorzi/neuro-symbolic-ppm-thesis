# XES - CSV / event-log loading
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pm4py

# Native pm4py / XES column names
CASE_ID = "case:concept:name"
ACTIVITY = "concept:name"
TIMESTAMP = "time:timestamp"
ORG_GROUP = "org:group"
LIFECYCLE = "lifecycle:transition"
INDEX = "@@index"


def read_log(path: str | Path, max_cases: int | None = None) -> pd.DataFrame:
    # Reads csv or xes log file

    # Look at the real extension, not the last one: public logs often arrive
    # gzipped (``BPI_Challenge_2012.xes.gz``), and to ``Path.suffix`` that is a
    # ``.gz``. Both pm4py and pandas read the compressed file on their own, so
    # all this has to do is dispatch to the right call.
    path = Path(path)
    suffixes = [s.lower() for s in path.suffixes]
    if ".xes" in suffixes:
        return read_xes(path, max_cases)
    elif ".csv" in suffixes:
        return read_csv(path, max_cases)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def read_csv(path: str | Path, max_cases: int | None = None) -> pd.DataFrame:
    # Load a CSV log via pandas into one row per event

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV dataset not found: {path}")
    if max_cases is not None and max_cases <= 0:
        raise ValueError("max_cases must be positive.")

    events = pd.read_csv(str(path))

    if events.empty:
        raise ValueError(f"No events found in {path}")

    return _read_file(events, max_cases)

def read_xes(path: str | Path, max_cases: int | None = None) -> pd.DataFrame:
    # Load an XES log via pm4py into one row per event

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"XES dataset not found: {path}")
    if max_cases is not None and max_cases <= 0:
        raise ValueError("max_cases must be positive.")

    events = pm4py.read_xes(str(path))
    if not isinstance(events, pd.DataFrame):
        events = pm4py.convert_to_dataframe(events)

    if events.empty:
        raise ValueError(f"No events found in {path}")

    return _read_file(events, max_cases)

def _read_file(events: pd.DataFrame, max_cases: int | None = None) -> pd.DataFrame:
    # Ordering function for the dataset

    for column in (CASE_ID, ACTIVITY, TIMESTAMP):
        if column not in events.columns:
            raise ValueError(f"The log has no '{column}' attribute.")
    if events[ACTIVITY].isna().any():
        raise ValueError(f"The log contains events without '{ACTIVITY}'.")

    events = events.reset_index(drop=True)
    events[INDEX] = range(len(events))

    if max_cases is not None:
        kept = events[CASE_ID].drop_duplicates().head(max_cases)
        events = events[events[CASE_ID].isin(kept)]

    for column in (ORG_GROUP, LIFECYCLE):
        if column not in events.columns:
            events[column] = pd.NA

    events[TIMESTAMP] = pd.to_datetime(events[TIMESTAMP], utc=True)
    events[CASE_ID] = events[CASE_ID].astype(str)
    return events.sort_values([CASE_ID, INDEX]).reset_index(drop=True)