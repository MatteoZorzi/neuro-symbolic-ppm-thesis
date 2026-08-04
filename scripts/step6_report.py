"""Gradino 6 contro gradino 5: la GRNN rende quanto costa?

Legge ``runs/step6_grnn/results.csv`` (la GRNN, fuori dalla matrice
pre-registrata) e lo appaia per seed con le righe ``seq`` di
``runs/final_matrix/results.csv``. Il confronto e' appaiato per costruzione:
le due varianti girano sullo stesso split e sulle stesse etichette corrotte a
parita' di seed, quindi la differenza cancella la varianza di inizializzazione.

Il costo fa parte del verdetto quanto l'accuratezza, quindi il tempo per epoca
si riporta accanto: la GRNN srotola il tempo a mano e non puo' vettorizzare la
ricorrenza sull'asse temporale.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CELL_KEYS = ["dataset", "arch", "noise"]
METRICS = {"accuracy": True, "top3": True, "macro_f1": True, "forbidden": False}


def load() -> pd.DataFrame:
    step6 = ROOT / "runs" / "step6_grnn" / "results.csv"
    if not step6.exists():
        sys.exit(f"manca {step6}: lancia prima final_matrix.py --variants grnn --out-dir step6_grnn")
    matrix = pd.read_csv(ROOT / "runs" / "final_matrix" / "results.csv")
    return pd.concat([matrix[matrix.variant == "seq"], pd.read_csv(step6)], ignore_index=True)


def main() -> None:
    df = load()
    paired = df.pivot_table(index=CELL_KEYS + ["seed"], columns="variant", values=list(METRICS))
    paired = paired.dropna()
    if paired.empty:
        sys.exit("nessuna cella con entrambe le varianti: il gradino 6 non copre le celle di seq")

    cells = paired.index.droplevel("seed").nunique()
    print(f"celle appaiate: {cells}   run: {len(paired)}\n")

    print("=== delta grnn - seq, appaiato per seed ===")
    for metric, higher_is_better in METRICS.items():
        delta = paired[(metric, "grnn")] - paired[(metric, "seq")]
        favourable = (delta > 0).sum() if higher_is_better else (delta < 0).sum()
        verso = "piu' e' meglio" if higher_is_better else "meno e' meglio"
        print(f"  {metric:<9} {delta.mean():+.5f}  sd {delta.std():.5f}  "
              f"a favore {favourable}/{len(delta)}   ({verso})")

    print("\n=== per cella, solo accuracy ===")
    per_cell = (paired[("accuracy", "grnn")] - paired[("accuracy", "seq")]).groupby(level=CELL_KEYS)
    table = pd.DataFrame({
        "delta medio": per_cell.mean().round(5),
        "seed a favore": per_cell.apply(lambda s: f"{int((s > 0).sum())}/{len(s)}"),
    })
    print(table.to_string())

    if "secs_per_epoch" in df.columns:
        cost = df.dropna(subset=["secs_per_epoch"]).groupby("variant")["secs_per_epoch"].mean()
        if len(cost) > 1:
            print(f"\n=== costo ===\n{cost.round(2).to_string()}")
            print(f"la GRNN costa {cost.get('grnn', float('nan')) / cost.get('seq', float('nan')):.1f}x per epoca")
        else:
            print("\ncosto: le righe di seq sono anteriori alla colonna secs_per_epoch, niente confronto")


if __name__ == "__main__":
    main()
