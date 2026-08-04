"""Analisi della croce dei due canali: loss del checker x encoder di marking.

La scala di ablazione T12 muove una cosa alla volta e non incrocia mai i due
canali: le varianti feature (marking/gnn/seq) girano con cross-entropy pura, il
checker gira senza feature. Questo script legge le tre celle mancanti
(``runs/channel_cross``) e le monta insieme a T12 in una griglia 2x4:

                 encoder:  nessuno   marking     gnn        seq
    loss = CE              baseline  marking     gnn        seq
    loss = CE + checker    checker   checker_*   checker_*  checker_*

Le domande, tutte appaiate per seed (twin-run: stesso seed = stesse etichette):

1. **effetto principale della loss** senza feature:  checker - baseline
2. **effetto della loss sopra ogni feature**:        checker_X - X
3. **interazione**: (checker_X - X) - (checker - baseline). E' questa la
   domanda vera. Se e' zero i due canali sono additivi e la feature resta
   inerte anche sotto la pressione della loss; se e' negativa sulla forbidden
   mass, la feature aiuta il modello a soddisfare il vincolo.

Previsione pre-registrata (dal ragionamento del 29 luglio): interazione **nulla**,
perche' ``allowed_mask`` e' indicizzata dall'ultimo token, che il modello gia'
possiede -- il marking non gli da' nessuna informazione che gli serva per
soddisfare quel termine.
"""

import sys
from pathlib import Path
from statistics import mean, stdev

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

T12_CSV = ROOT / "runs" / "final_matrix" / "results.csv"
CROSS_CSV = ROOT / "runs" / "channel_cross" / "results.csv"

DATASET, ARCH, NOISE = "Sepsis_Case", "gru", 0.0
ENCODERS = ("marking", "gnn", "seq")
METRICS = (("accuracy", True), ("forbidden", False))  # (nome, higher_is_better)


def load() -> pd.DataFrame:
    """T12 + croce, filtrate sulla cella (dataset, arch, noise) dell'esperimento."""
    frames = []
    for path in (T12_CSV, CROSS_CSV):
        if not path.exists():
            raise SystemExit(f"manca {path} -- gira prima final_matrix.py")
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    return df[(df["dataset"] == DATASET) & (df["arch"] == ARCH)
              & (df["noise"] == NOISE)].copy()


def by_seed(df: pd.DataFrame, variant: str, metric: str) -> dict[int, float]:
    rows = df[df["variant"] == variant]
    return dict(zip(rows["seed"], rows[metric]))


def paired(df: pd.DataFrame, left: str, right: str, metric: str) -> list[float]:
    """left - right, seed per seed. Solleva se i seed non coincidono."""
    a, b = by_seed(df, left, metric), by_seed(df, right, metric)
    seeds = sorted(set(a) & set(b))
    if len(seeds) != len(a) or len(seeds) != len(b):
        raise SystemExit(f"seed non appaiati fra {left} e {right}: "
                         f"{sorted(a)} vs {sorted(b)}")
    return [a[s] - b[s] for s in seeds]


def summarise(deltas: list[float], higher_is_better: bool) -> str:
    """media +/- deviazione e quanti seed vanno nella direzione buona."""
    good = sum((d > 0) if higher_is_better else (d < 0) for d in deltas)
    spread = stdev(deltas) if len(deltas) > 1 else 0.0
    return f"{mean(deltas):+.4f} +/- {spread:.4f}  ({good}/{len(deltas)} seed a favore)"


def main() -> None:
    df = load()
    present = sorted(df["variant"].unique())
    print(f"cella: {DATASET} | {ARCH} | noise {NOISE}")
    print(f"varianti presenti ({len(present)}): {present}")
    counts = df.groupby("variant")["seed"].count().to_dict()
    if set(counts.values()) != {10}:
        print(f"ATTENZIONE: run per variante non uniformi: {counts}")

    for metric, higher_is_better in METRICS:
        direction = "piu' alto meglio" if higher_is_better else "piu' basso meglio"
        print(f"\n================ {metric.upper()}  ({direction}) ================")

        # ---- la griglia 2x4, medie sui 10 seed
        print(f"{'':<22}{'loss = CE':>14}{'loss = CE + checker':>24}")
        for encoder in ("nessuno",) + ENCODERS:
            plain = "baseline" if encoder == "nessuno" else encoder
            crossed = "checker" if encoder == "nessuno" else f"checker_{encoder}"
            values = [by_seed(df, v, metric) for v in (plain, crossed)]
            cells = ["n/d" if not v else f"{mean(v.values()):.4f}" for v in values]
            print(f"encoder {encoder:<14}{cells[0]:>14}{cells[1]:>24}")

        # ---- 1. effetto principale della loss, senza feature
        base_effect = paired(df, "checker", "baseline", metric)
        print(f"\n[1] loss sola          checker - baseline      "
              f"{summarise(base_effect, higher_is_better)}")

        # ---- 2. effetto della loss sopra ogni feature
        print("\n[2] la loss aggiunta a un modello che ha gia' la feature:")
        effects = {}
        for encoder in ENCODERS:
            crossed = f"checker_{encoder}"
            if crossed not in present:
                print(f"    {crossed:<18} assente, salto")
                continue
            effects[encoder] = paired(df, crossed, encoder, metric)
            print(f"    checker_{encoder:<10} - {encoder:<8} "
                  f"{summarise(effects[encoder], higher_is_better)}")

        # ---- 3. l'interazione: la feature cambia quanto rende la loss?
        print("\n[3] INTERAZIONE  (effetto della loss con la feature) - (senza):")
        for encoder, effect in effects.items():
            interaction = [e - b for e, b in zip(effect, base_effect)]
            print(f"    {encoder:<8} {summarise(interaction, higher_is_better)}")

        # ---- e la lettura duale: la feature aggiunta al checker
        print("\n[4] lettura duale, la feature aggiunta al checker:")
        for encoder in effects:
            deltas = paired(df, f"checker_{encoder}", "checker", metric)
            print(f"    checker_{encoder:<10} - checker  "
                  f"{summarise(deltas, higher_is_better)}")


if __name__ == "__main__":
    main()
