# Recompute ``tab:exp-artifacts`` of Chapter 5 by re-mining the artifacts

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


# The configuration ``matrix.py`` builds for protocol B and for protocol C
def protocol_config(knowledge_source: str):
    config = temporal_protocol()
    config = replace(config, data=replace(config.data,
                                          knowledge_source=knowledge_source))
    return replace(
        config,
        training=replace(config.training, logic_weight=matrix.LOGIC_WEIGHT,
                         epochs=matrix.MAX_EPOCHS),
        model=replace(config.model, hidden_dim=matrix.RECURRENT_HIDDEN,
                      recurrent_layers=matrix.RECURRENT_LAYERS))


# Mine one log under one protocol and count what the table reports
def measure(dataset: str, knowledge_source: str) -> dict:
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


# The net fingerprint each log carries in the grid, one per dataset
def fingerprints_of_grid(protocol: str) -> dict[str, str]:
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
