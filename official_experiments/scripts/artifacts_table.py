"""Recompute ``tab:exp-artifacts`` of Chapter 5 by re-mining the artifacts.

This table is different from every other one in the chapter. The others
summarise a result grid, so checking them means reading a CSV. This one
describes the symbolic objects the runs were given -- the discovered Petri net,
its reachability automaton, the precedence constraints -- and none of those
counts is written to the grid. Checking it therefore means mining them again.

That is the point. The nets are rebuilt through ``build_dataset_artifacts``,
the same function ``matrix.py`` calls before it trains anything, with
the same protocol configuration. What comes out is what the runs saw, or the
run was not reproducible.

The strongest check here is not a count, it is ``net_fingerprint``. Every row of
the grid carries the fingerprint of the net that produced it, and the fingerprint
is built from portable things only -- counts and sorted labels, never the names
pm4py invents at each call. So the net mined now can be compared against the net
the runs used, exactly, and the script reports whether they agree. Two nets with
the same counts but different arcs would still pass; that limit is documented in
``nspm.process.artifact_store``.

Mining is not free. The artifact cache under ``runs/_artifacts`` is used, so the
first run is slow and later ones are not.

    python official_experiments/scripts/artifacts_table.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matrix  # noqa: E402
from common import GRID_CSV  # noqa: E402
from nspm.config import temporal_protocol  # noqa: E402

#: Logs in the order the chapter lists them, with the labels the table prints.
DATASETS = {
    "Sepsis_Case": "Sepsis Cases",
    "BPIC_2013_incidents": "BPIC 2013 incidents",
    "BPIC_2020_DomesticDeclarations": "BPIC 2020 dom. dec.",
    "BPI_Challenge_2012": "BPIC 2012",
}

#: The thesis name of each protocol, and the partition its knowledge comes from.
#: On disk the same two protocols were the folders ``noise_curve_b`` and
#: ``noise_curve_c``, and ``matrix.py`` still calls them B and C.
PROTOCOLS = {"test": "test", "train": "train"}

#: The recorded fingerprints are read from the published grid rather than from
#: the job output under ``runs/``, which is not versioned and is not published:
#: reading it here made this the one script in the directory that a clone could
#: not run. ``all_grids.csv`` carries the same ``net_fingerprint`` column, one
#: value per (log, protocol), so the check is the same one and it now runs
#: anywhere.
PROTOCOL_LETTER = {"test": "B", "train": "C"}

#: Only the plain prefix log is needed: nothing here reads a marking sequence,
#: and building one would cost minutes per log for a number the table does not
#: print. The masks are the two of RQ3 plus the evaluation mask.
FAMILIES = {"plain"}
MASKS = {"dfa", "net", "net_state"}
CACHE = ROOT / "runs" / "_artifacts"


def protocol_config(knowledge_source: str):
    """The configuration ``matrix.py`` builds for protocol B and for protocol C.

    They differ in one field, which is the whole point of the pair: the
    partition the knowledge is mined from. Everything else -- temporal split,
    vocabulary over all partitions, event-level noise -- comes from
    ``temporal_protocol`` untouched.
    """
    config = temporal_protocol()
    config = replace(config, data=replace(config.data,
                                          knowledge_source=knowledge_source))
    return replace(
        config,
        training=replace(config.training, logic_weight=matrix.LOGIC_WEIGHT,
                         epochs=matrix.MAX_EPOCHS),
        model=replace(config.model, hidden_dim=matrix.RECURRENT_HIDDEN,
                      recurrent_layers=matrix.RECURRENT_LAYERS))


def measure(dataset: str, knowledge_source: str) -> dict:
    """Mine one log under one protocol and count what the table reports."""
    art = matrix.build_dataset_artifacts(
        dataset, protocol_config(knowledge_source), FAMILIES, MASKS,
        cache_dir=CACHE)

    net, reachability = art["petrinet"], art["reachability"]
    silent = sum(1 for t in net.transitions if t.label is None)
    return {
        "places": len(net.places),
        "transitions": len(net.transitions),
        "silent": silent,
        # The arcs live on the wrapped pm4py net; the wrapper exposes only the
        # places and the transitions, because those are what it indexes.
        "arcs": len(net.network.arcs),
        "states": reachability.state_count,
        # "Pairs" in the thesis: the activity pairs the projection admits.
        "pairs": art["net_dfa"].transition_count,
        "constraints": len(art["constraints"]),
        "fingerprint": art["net_fingerprint"],
    }


def fingerprints_of_grid(protocol: str) -> dict[str, str]:
    """The net fingerprint each log carries in the grid, one per dataset.

    A log whose rows disagree is left out rather than reported: a grid with two
    nets under one protocol is a grid that should not have been merged, and
    ``publish_experiments.py`` is where that is caught.
    """
    if not GRID_CSV.exists():
        return {}
    grid = pd.read_csv(GRID_CSV)
    grid = grid[grid["protocol"] == PROTOCOL_LETTER[protocol]]
    unique = grid.groupby("dataset")["net_fingerprint"].unique()
    return {dataset: values[0] for dataset, values in unique.items()
            if len(values) == 1}


def main() -> None:
    rows, checks = [], []
    for protocol, knowledge_source in PROTOCOLS.items():
        recorded = fingerprints_of_grid(protocol)
        for dataset, label in DATASETS.items():
            measured = measure(dataset, knowledge_source)
            rows.append({"log": label, "knowledge": protocol,
                         **{k: v for k, v in measured.items()
                            if k != "fingerprint"}})
            checks.append({
                "log": label, "knowledge": protocol,
                "mined now": measured["fingerprint"],
                "in the grid": recorded.get(dataset, "--"),
                "agree": measured["fingerprint"] == recorded.get(dataset),
            })

    table = pd.DataFrame(rows).sort_values(["log", "knowledge"], kind="stable")
    print()
    print("The artifacts, re-mined:")
    print(table.to_string(index=False))
    print()
    print("Net fingerprint, against the one the runs recorded:")
    print(pd.DataFrame(checks).to_string(index=False))


if __name__ == "__main__":
    main()
