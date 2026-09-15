# Saves the mined objects to disk and recalls them identical, keyed by the
# fingerprint of the knowledge traces.

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

#: Changing this number invalidates every existing cache. Raise it when the
#: SHAPE of what is saved changes, or when the way the objects are built
#: changes -- otherwise a net mined with different parameters gets reused in the
#: belief that it is the right one.
FORMAT_VERSION = 1


# Fingerprint of the traces mined from, plus the miner's parameters
def knowledge_fingerprint(traces: Mapping[str, Sequence[str]],
                          **parameters: Any) -> str:

    digest = hashlib.sha256()
    digest.update(f"v{FORMAT_VERSION}\n".encode())
    for name in sorted(parameters):
        digest.update(f"{name}={parameters[name]!r}\n".encode())
    for case_id in sorted(traces):
        digest.update(case_id.encode("utf-8", "replace"))
        digest.update(b"\x00")
        digest.update("\x01".join(traces[case_id]).encode("utf-8", "replace"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


# Portable fingerprint of the net, to be written into the CSV
def net_fingerprint(net) -> str:

    silent = sum(1 for t in net.transitions if t.label is None)
    labels = sorted(t.label for t in net.transitions if t.label is not None)
    digest = hashlib.sha256()
    digest.update(f"{len(net.places)}/{len(net.transitions)}/{silent}\n".encode())
    for label in labels:
        digest.update(label.encode("utf-8", "replace"))
        digest.update(b"\n")
    return digest.hexdigest()[:12]


# The file for one ``(dataset, knowledge traces)`` combination
def cache_path(directory: Path, dataset: str, key: str) -> Path:

    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in dataset)
    return directory / f"{safe}.{key}.pkl"


# The saved contents, or an empty dictionary
def load(path: Path) -> dict:

    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as error:  # noqa: BLE001 - vedi docstring
        print(f"cache degli artefatti illeggibile ({path.name}): {error} "
              f"-- si rimina", flush=True)
        return {}
    if payload.get("format_version") != FORMAT_VERSION:
        return {}
    return payload


# Atomic write: a temporary file in the same directory, then ``replace``
def save(path: Path, payload: dict) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "format_version": FORMAT_VERSION}
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    except Exception as error:  # noqa: BLE001
        print(f"cache degli artefatti non salvata ({path.name}): {error}",
              flush=True)
        temporary.unlink(missing_ok=True)
