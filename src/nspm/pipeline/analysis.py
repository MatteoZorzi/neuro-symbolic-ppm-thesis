"""Exploratory tables and reports for an XES event log."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from ..data.xes import (
    ACTIVITY,
    CASE_ID,
    INDEX,
    LIFECYCLE,
    ORG_GROUP,
    TIMESTAMP,
    read_xes,
)
from ..visualization.plots import (
    plot_activity_frequency,
    plot_case_durations,
    plot_variants,
)


CORE_COLUMNS = {CASE_ID, INDEX, ACTIVITY, TIMESTAMP, ORG_GROUP, LIFECYCLE}

#: An outcome marker maps a boolean case-column name to a predicate over the set
#: of activities seen in that case. Pass a custom mapping (or ``{}`` to disable)
#: for non-Sepsis logs; these defaults encode Sepsis Cases domain knowledge.
OutcomeMarkers = Mapping[str, Callable[[set], bool]]

SEPSIS_OUTCOME_MARKERS: OutcomeMarkers = {
    "has_return_er": lambda activities: "Return ER" in activities,
    "has_admission_ic": lambda activities: "Admission IC" in activities,
    "has_release": lambda activities: any(
        str(activity).startswith("Release ") for activity in activities
    ),
}


@dataclass
class AnalysisTables:
    """All tabular products generated during exploratory analysis."""

    events: pd.DataFrame
    cases: pd.DataFrame
    activities: pd.DataFrame
    variants: pd.DataFrame
    transitions: pd.DataFrame
    groups: pd.DataFrame
    attributes: pd.DataFrame


def build_case_table(
    events: pd.DataFrame,
    outcome_markers: OutcomeMarkers | None = SEPSIS_OUTCOME_MARKERS,
) -> pd.DataFrame:
    """Aggregate event rows into one descriptive row per case.

    ``outcome_markers`` adds one boolean column per entry, flagging cases whose
    activity set satisfies the predicate. Defaults to the Sepsis outcome flags;
    pass a custom mapping or ``{}`` for other logs.
    """

    ordered = events.sort_values([CASE_ID, INDEX])
    grouped = ordered.groupby(CASE_ID, sort=False)
    cases = grouped.agg(
        event_count=(ACTIVITY, "size"),
        unique_activities=(ACTIVITY, "nunique"),
        start_time=(TIMESTAMP, "min"),
        end_time=(TIMESTAMP, "max"),
        start_activity=(ACTIVITY, "first"),
        end_activity=(ACTIVITY, "last"),
    ).reset_index()
    cases["duration_hours"] = (
        cases["end_time"] - cases["start_time"]
    ).dt.total_seconds() / 3600

    variants = grouped[ACTIVITY].agg(lambda values: " > ".join(values.astype(str)))
    cases = cases.merge(variants.rename("variant"), on=CASE_ID)
    activity_sets = grouped[ACTIVITY].agg(set)
    for column, predicate in (outcome_markers or {}).items():
        cases[column] = cases[CASE_ID].map(
            lambda case_id, predicate=predicate: predicate(activity_sets.loc[case_id])
        )
    return cases


def build_transition_table(events: pd.DataFrame) -> pd.DataFrame:
    """Count directly-follows relations between consecutive activities."""

    ordered = events.sort_values([CASE_ID, INDEX]).copy()
    ordered["next_activity"] = ordered.groupby(CASE_ID)[ACTIVITY].shift(-1)
    return (
        ordered.dropna(subset=[ACTIVITY, "next_activity"])
        .groupby([ACTIVITY, "next_activity"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def build_attribute_table(events: pd.DataFrame) -> pd.DataFrame:
    """Describe non-core event attributes without assuming a fixed schema."""

    rows: list[dict[str, object]] = []
    case_count = events[CASE_ID].nunique()
    for column in sorted(set(events.columns) - CORE_COLUMNS):
        series = events[column].dropna()
        if series.empty:
            continue
        covered_cases = events.loc[series.index, CASE_ID].nunique()
        common: dict[str, object] = {
            "attribute": column,
            "observations": int(series.size),
            "missing_rate": float(1 - series.size / len(events)),
            "cases_with_value": int(covered_cases),
            "case_coverage": float(covered_cases / case_count),
            "unique_values": int(series.nunique()),
        }
        is_boolean = pd.api.types.is_bool_dtype(series) or series.map(
            lambda value: isinstance(value, bool)
        ).all()
        if is_boolean:
            rows.append(
                {
                    **common,
                    "type": "boolean",
                    "mean_or_true_rate": float(series.mean()),
                    "min": None,
                    "max": None,
                }
            )
        elif pd.api.types.is_numeric_dtype(series):
            rows.append(
                {
                    **common,
                    "type": "numeric",
                    "mean_or_true_rate": float(series.mean()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                }
            )
        else:
            rows.append(
                {
                    **common,
                    "type": "categorical",
                    "mean_or_true_rate": None,
                    "min": None,
                    "max": None,
                }
            )
    return pd.DataFrame(rows)


def build_analysis_tables(
    events: pd.DataFrame,
    outcome_markers: OutcomeMarkers | None = SEPSIS_OUTCOME_MARKERS,
) -> AnalysisTables:
    """Build every reusable EDA table from an event DataFrame."""

    cases = build_case_table(events, outcome_markers)
    activities = (
        events[ACTIVITY].value_counts(dropna=False).rename_axis(ACTIVITY).rename("count").reset_index()
    )
    variants = (
        cases["variant"].value_counts().rename_axis("variant").rename("count").reset_index()
    )
    groups = (
        events[ORG_GROUP].value_counts(dropna=False).rename_axis(ORG_GROUP).rename("count").reset_index()
    )
    return AnalysisTables(
        events=events,
        cases=cases,
        activities=activities,
        variants=variants,
        transitions=build_transition_table(events),
        groups=groups,
        attributes=build_attribute_table(events),
    )


def summarise(tables: AnalysisTables, input_path: str | Path) -> dict[str, object]:
    """Create a compact JSON-compatible dataset summary."""

    cases = tables.cases
    # Report counts for any outcome-marker boolean columns present.
    outcomes = {
        column: int(cases[column].sum())
        for column in cases.columns
        if column.startswith("has_") and pd.api.types.is_bool_dtype(cases[column])
    }
    return {
        "input": str(input_path),
        "cases": int(len(cases)),
        "events": int(len(tables.events)),
        "activities": int(tables.events[ACTIVITY].nunique()),
        "variants": int(cases["variant"].nunique()),
        "groups": int(tables.events[ORG_GROUP].nunique()),
        "events_per_case": {
            "mean": float(cases["event_count"].mean()),
            "median": float(cases["event_count"].median()),
            "min": int(cases["event_count"].min()),
            "max": int(cases["event_count"].max()),
        },
        "duration_hours": {
            "mean": float(cases["duration_hours"].mean()),
            "median": float(cases["duration_hours"].median()),
            "min": float(cases["duration_hours"].min()),
            "max": float(cases["duration_hours"].max()),
        },
        "outcomes": outcomes,
        "most_common_activity": str(tables.activities.iloc[0][ACTIVITY]),
        "most_common_variant_cases": int(tables.variants.iloc[0]["count"]),
    }


def analyse(
    input_path: str | Path,
    output_dir: str | Path,
    max_cases: int | None = None,
    top_n: int = 20,
    save_events: bool = True,
    outcome_markers: OutcomeMarkers | None = SEPSIS_OUTCOME_MARKERS,
) -> dict[str, object]:
    """Run and persist the complete exploratory analysis."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = build_analysis_tables(
        read_xes(input_path, max_cases=max_cases), outcome_markers
    )
    summary = summarise(tables, input_path)

    outputs = {
        "cases.csv": tables.cases,
        "activities.csv": tables.activities,
        "variants.csv": tables.variants,
        "directly_follows.csv": tables.transitions,
        "organizational_groups.csv": tables.groups,
        "attribute_summary.csv": tables.attributes,
    }
    if save_events:
        outputs["events.csv"] = tables.events
    for filename, table in outputs.items():
        table.to_csv(output_dir / filename, index=False)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_activity_frequency(tables.events, output_dir / "activity_frequency.png", top_n)
    plot_case_durations(tables.cases, output_dir / "case_duration_distribution.png")
    plot_variants(tables.variants, output_dir / "top_variants.png", top_n)
    return summary
