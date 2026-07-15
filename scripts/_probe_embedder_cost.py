"""Time a single worst-case grid cell (full data, all 4 variants) to gauge cost."""
from __future__ import annotations

import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nspm.config import ProjectPaths
from nspm.pipeline.benchmark import run_corruption_grid

paths = ProjectPaths.from_root(ROOT, "Sepsis_Case")

for variants in (("baseline", "baseline_mask", "checker"), ("embedder",)):
    t0 = time.perf_counter()
    df = run_corruption_grid(
        paths.dataset, dataset_name="Sepsis_Case",
        model_kinds=("gru",), noise_levels=(0.0,), train_fractions=(1.0,),
        variants=variants, epochs=30, embedder_epochs=5, logic_weight=0.5,
        seed=42, drop_unseen=True, verbose=True,
    )
    dt = time.perf_counter() - t0
    print(f">>> variants={variants} took {dt:.1f}s  ({dt/60:.1f} min)", flush=True)
