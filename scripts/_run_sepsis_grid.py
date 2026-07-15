"""Run the full Sepsis corruption x fraction x model grid directly (no notebook).

Writes results into runs/benchmark_corruption.csv using the same merge policy as
the notebook's grid cell, but without the Jupyter cell-timeout. Per-cell progress
is printed by run_corruption_grid(verbose=True). Variants can be trimmed via the
VARIANTS env-style constant below if the embedder branch proves intractable.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nspm.config import ProjectPaths
from nspm.pipeline.benchmark import run_corruption_grid

DATASET_NAME = "Sepsis_Case"
MODELS = ("gru", "lstm")
VARIANTS = ("baseline", "baseline_mask", "checker", "embedder")
NOISE_LEVELS = (0.0, 0.1, 0.25, 0.4)
TRAIN_FRACTIONS = (1.0, 0.5, 0.25, 0.125)
EPOCHS = 30
EMBEDDER_EPOCHS = 5
LOGIC_WEIGHT = 0.5
SEED = 42

paths = ProjectPaths.from_root(ROOT, DATASET_NAME)
RESULTS_CSV = ROOT / "runs" / "benchmark_corruption.csv"

print(f"[grid] {DATASET_NAME}: models={MODELS} variants={VARIANTS} "
      f"noise={NOISE_LEVELS} fractions={TRAIN_FRACTIONS} epochs={EPOCHS}", flush=True)
t0 = time.perf_counter()

grid = run_corruption_grid(
    paths.dataset, dataset_name=DATASET_NAME,
    model_kinds=MODELS, noise_levels=NOISE_LEVELS,
    train_fractions=TRAIN_FRACTIONS, variants=VARIANTS,
    epochs=EPOCHS, embedder_epochs=EMBEDDER_EPOCHS, logic_weight=LOGIC_WEIGHT,
    seed=SEED, drop_unseen=True, verbose=True,
)

cached = pd.read_csv(RESULTS_CSV) if RESULTS_CSV.exists() else None
compatible = cached is not None and "train_fraction" in cached.columns
others = cached[cached["dataset"] != DATASET_NAME] if compatible else None
combined = grid if others is None else pd.concat([others, grid], ignore_index=True)
combined.to_csv(RESULTS_CSV, index=False)

dt = time.perf_counter() - t0
print(f"[grid] DONE in {dt/60:.1f} min — wrote {len(grid)} rows for {DATASET_NAME} "
      f"({len(combined)} total) to {RESULTS_CSV.name}", flush=True)
