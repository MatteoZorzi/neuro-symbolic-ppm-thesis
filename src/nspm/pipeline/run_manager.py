"""Creation of incrementally numbered experiment directories."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    """All writable locations belonging to a single experiment run."""

    root: Path
    models: Path


def _safe_name(dataset_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", dataset_name).strip("_")
    if not cleaned:
        raise ValueError("dataset_name must contain an alphanumeric character.")
    return cleaned


def create_next_run(runs_dir: str | Path, dataset_name: str) -> RunPaths:
    """Atomically create ``<dataset>_run_N`` with the next available index."""

    runs_dir = Path(runs_dir).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{_safe_name(dataset_name)}_run_"
    indices = []
    for candidate in runs_dir.glob(f"{prefix}*"):
        suffix = candidate.name.removeprefix(prefix)
        if candidate.is_dir() and suffix.isdigit():
            indices.append(int(suffix))

    index = max(indices, default=0) + 1
    while True:
        root = runs_dir / f"{prefix}{index}"
        try:
            root.mkdir()
            break
        except FileExistsError:
            index += 1

    models = root / "models"
    models.mkdir()
    return RunPaths(root=root, models=models)

