"""Oracolo: esercita tabelle e figure di ``nspm.visualization.matrix_plots``.

Verifica che il notebook dei risultati abbia sotto di se' funzioni che girano
davvero: 900 righe caricate, delta appaiati coerenti con un calcolo a mano,
vincitori con la reliability, e le tre figure salvate su disco.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nspm.visualization.architecture_plots import (  # noqa: E402
    plot_ablation_ladder, plot_variant_architecture,
)
from nspm.visualization.matrix_plots import (  # noqa: E402
    VARIANT_ORDER, cell_means, load_matrix, metric_table, model_catalogue,
    paired_deltas, plot_metric_by_noise, plot_paired_deltas,
    plot_winner_reliability, relative_change, winners,
)

out = ROOT / "runs" / "_check_matrix_plots"
out.mkdir(parents=True, exist_ok=True)

df = load_matrix(ROOT / "runs" / "final_matrix" / "results.csv")
print(f"[load ] {len(df)} righe, varianti in ordine: {list(df['variant'].cat.categories)}")
assert len(df) == 900, "la matrice non e' completa"

catalogue = model_catalogue(df)
print(f"[cat  ] {len(catalogue)} configurazioni, run per configurazione: "
      f"{sorted(catalogue['run'].unique())}, seed: {sorted(catalogue['seed'].unique())}")
assert len(catalogue) == 10 and set(catalogue["run"]) == {90} and set(catalogue["seed"]) == {10}

cells = cell_means(df)
print(f"[medie] {len(cells)} celle (attese 90), colonne: {len(cells.columns)}")
assert len(cells) == 90

# --- delta appaiati: confronto con un calcolo indipendente su una cella nota
deltas = paired_deltas(df, "accuracy")
print(f"[delta] {len(deltas)} coppie (attese 720)")
assert len(deltas) == 720

probe = dict(dataset="Sepsis_Case", arch="gru", noise=0.0, seed=0)
mask = (df["dataset"] == probe["dataset"]) & (df["arch"] == probe["arch"]) \
    & (df["noise"] == probe["noise"]) & (df["seed"] == probe["seed"])
cell = df[mask].set_index("variant")["accuracy"]
atteso = cell["seq"] - cell["baseline"]
ottenuto = deltas[(deltas["dataset"] == probe["dataset"]) & (deltas["arch"] == probe["arch"])
                  & (deltas["noise"] == probe["noise"]) & (deltas["seed"] == probe["seed"])
                  & (deltas["variant"] == "seq")]["delta"].item()
print(f"[delta] seq-baseline su {probe}: atteso {atteso:+.6f}, ottenuto {ottenuto:+.6f}")
assert abs(atteso - ottenuto) < 1e-12, "i delta non sono appaiati per seed"

for metric in ("accuracy", "forbidden", "violation_rate"):
    table = winners(df, metric)
    print(f"[vinci] {metric:15s} {len(table)} celle | vincitori: "
          f"{table['migliore'].value_counts().to_dict()} | "
          f"win rate medio {table['win_rate'].mean():.2f}")
    assert len(table) == 18

print("\n[tab  ] accuracy, GRU:")
print(metric_table(df, "accuracy", arch="gru"))

# la riduzione relativa deve coincidere con quella citata in JOURNAL.md (F3)
change = relative_change(df, "forbidden")
print("\n[rel  ] forbidden, variazione % sulla baseline:")
print(change.round(1))
assert change["checker"].max() < -25, "il calo del checker non e' piu' quello documentato"

figures = {
    "accuracy_deltas.png": plot_paired_deltas(df, "accuracy"),
    "forbidden_levels.png": plot_metric_by_noise(df, "forbidden"),
    "forbidden_winners.png": plot_winner_reliability(df, "forbidden"),
    "accuracy_winners.png": plot_winner_reliability(df, "accuracy"),
    "ladder.png": plot_ablation_ladder(),
}
figures.update({
    f"architettura_{variant}.png": plot_variant_architecture(variant)
    for variant in VARIANT_ORDER
})
for name, figure in figures.items():
    figure.savefig(out / name, dpi=120, bbox_inches="tight")
    print(f"[fig  ] {name} salvata")

print("\nTUTTO VERDE")
