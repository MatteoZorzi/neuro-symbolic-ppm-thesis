"""XES/event-log loading, standardised on pm4py.

The whole pipeline uses pm4py's native column names so there is a single,
tool-standard schema end to end. The names are exposed as constants and should
be imported from here rather than written as string literals:

    CASE_ID   = "case:concept:name"
    ACTIVITY  = "concept:name"
    TIMESTAMP = "time:timestamp"
    ORG_GROUP = "org:group"
    LIFECYCLE = "lifecycle:transition"
    INDEX     = "@@index"   # explicit event order

pm4py does not expose a per-event order column, so ``INDEX`` is added to capture
the original in-file event sequence. This matters because the Sepsis log holds
many events sharing a timestamp; ordering by timestamp alone would reorder them
and corrupt the directly-follows / next-activity pairs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pm4py


# Native pm4py / XES column names — the canonical schema for the whole pipeline.
CASE_ID = "case:concept:name"
ACTIVITY = "concept:name"
TIMESTAMP = "time:timestamp"
ORG_GROUP = "org:group"
LIFECYCLE = "lifecycle:transition"
INDEX = "@@index"


def read_xes(path: str | Path, max_cases: int | None = None) -> pd.DataFrame:
    """Load an XES log via pm4py into one row per event.

    Returns a DataFrame using pm4py's native column names plus an explicit
    ``@@index`` column capturing the original event order. ``ORG_GROUP`` and
    ``LIFECYCLE`` are guaranteed to exist (filled with NA if the log omits them)
    so downstream code can rely on the schema.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"XES dataset not found: {path}")
    if max_cases is not None and max_cases <= 0:
        raise ValueError("max_cases must be positive.")

    try:
        events = pm4py.read_xes(str(path), show_progress_bar=False)
    except TypeError:  # Older pm4py without the keyword.
        events = pm4py.read_xes(str(path))
    if not isinstance(events, pd.DataFrame):
        events = pm4py.convert_to_dataframe(events)

    if events.empty:
        raise ValueError(f"No events found in {path}")
    if ACTIVITY not in events.columns:
        raise ValueError(f"The XES log has no '{ACTIVITY}' attribute.")
    if events[ACTIVITY].isna().any():
        raise ValueError(f"The XES log contains events without '{ACTIVITY}'.")

    # pm4py yields events in file order; freeze it before any sort so that tied
    # timestamps keep their original sequence.
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
