"""Execute the Sepsis_Case TLEAF pipeline notebook in-place with full grid.

Runs every cell (incl. the corruption x fraction x model grid with the embedder
branch) on the GPU, writes outputs back into the .ipynb, and prints a summary.
"""
from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "Sepsis_Case_TLEAF_Pipeline.ipynb"

print(f"[run] executing {NB.name} ...", flush=True)
started = time.perf_counter()

nb = nbformat.read(NB, as_version=4)
client = NotebookClient(
    nb, timeout=7200, kernel_name="python3",
    resources={"metadata": {"path": str(NB.parent)}},
)
client.execute()
nbformat.write(nb, NB)

elapsed = time.perf_counter() - started
errs = [
    (i, o.get("ename"), o.get("evalue"))
    for i, c in enumerate(nb.cells)
    for o in c.get("outputs", [])
    if o.get("output_type") == "error"
]
print(f"[run] finished in {elapsed/60:.1f} min, error cells: {len(errs)}", flush=True)
for i, name, val in errs:
    print(f"  ERROR cell {i}: {name}: {val}", flush=True)

csv = ROOT / "runs" / "benchmark_corruption.csv"
if csv.exists():
    import pandas as pd
    d = pd.read_csv(csv)
    print(f"[run] CSV rows={len(d)}, datasets={sorted(d['dataset'].unique())}, "
          f"variants={sorted(d['variant'].unique())}, "
          f"fractions={sorted(d['train_fraction'].unique()) if 'train_fraction' in d else 'n/a'}",
          flush=True)
print("[run] DONE", flush=True)
