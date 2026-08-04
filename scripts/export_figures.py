"""Renderizza in ``docs/figures/`` le figure della presentazione.

Solo le figure che **portano dati**: i delta appaiati sui 900 run, la massa
vietata contro il noise, e i cinque blocchi-diagramma delle architetture. Le
altre (mappa dei canali, griglia 2x4, lanci di moneta, caso 'AG') sono tabelle o
diagrammi e stanno in ``docs/PRESENTATION_DATA.md``: le disegna il tool di
design, che le fara' native al deck invece che importate.

Il motivo per cui questo script esiste invece di esportare a mano dal notebook:
dopo un re-run della matrice le immagini del deck si rigenerano con un comando,
invece di desincronizzarsi in silenzio.

Due preset:

* ``slide`` -- font ingranditi e una sola architettura (GRU) dove il pannello
  completo verrebbe verticale. Su 16:9 proiettato il testo delle figure tarate
  per il notebook (fontsize 7-8) finisce sotto i 6 pt: illeggibile dalla terza
  fila.
* ``paper`` -- dimensioni originali, tutte le architetture, per la tesi scritta.

Uso:
    python scripts/export_figures.py            # entrambi i preset
    python scripts/export_figures.py --preset slide
"""

import argparse
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _register_stub(name: str, path: Path) -> None:
    """Registra un pacchetto vuoto in ``sys.modules`` per saltarne l'``__init__``.

    Le figure hanno bisogno solo di numpy/pandas/matplotlib, ma ``nspm/__init__``
    importa l'intera pipeline (pm4py, torch) e ``nspm.visualization/__init__``
    importa ``plots`` (grafi degli automi). Con il pacchetto gia' in
    ``sys.modules``, ``import nspm.visualization.matrix_plots`` carica il solo
    sottomodulo. Serve perche' questo script deve poter girare in un env senza
    lo stack ML -- utile di suo, e necessario qui: matplotlib in ``tleaf`` ha il
    backend Agg rotto (crash nativo 0xC0000409 su qualunque ``draw()``), quindi
    le figure vanno renderizzate altrove.
    """
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


_register_stub("nspm", ROOT / "src" / "nspm")
_register_stub("nspm.visualization", ROOT / "src" / "nspm" / "visualization")

from nspm.visualization.architecture_plots import (  # noqa: E402
    plot_ablation_ladder, plot_variant_architecture,
)
from nspm.visualization.matrix_plots import (  # noqa: E402
    DATASET_LABELS, VARIANT_COLORS, VARIANT_LABELS, VARIANT_MARKERS,
    VARIANT_ORDER, cell_means, load_matrix, plot_metric_by_noise,
    plot_paired_deltas, plot_winner_reliability, winners,
)

NEUTRAL = "#8a8981"  # le varianti senza effetto, indistinguibili fra loro

MATRIX_CSV = ROOT / "runs" / "final_matrix" / "results.csv"
OUT = ROOT / "docs" / "figures"

#: Font piu' grandi per la proiezione. Le size scritte a mano dentro le funzioni
#: di plot (titoli dei pannelli a 10) non le tocca: quello che conta e' il
#: rapporto fra testo e figura, e la figura viene messa a piena larghezza.
SLIDE_RC = {
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
    "savefig.dpi": 200,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}

PAPER_RC = {"savefig.dpi": 200, "figure.facecolor": "white", "savefig.facecolor": "white"}


def plot_forbidden_slide(df, path: Path):
    """La massa vietata contro il noise, in geometria da slide.

    Due differenze rispetto a :func:`plot_metric_by_noise`, entrambe editoriali:

    * **un dataset per colonna** invece che per riga -- filtrata a una sola
      architettura, la griglia originale diventa una colonna verticale che su
      16:9 spreca due terzi della larghezza e sovrappone le etichette y;
    * **le quattro varianti senza effetto in grigio, una sola voce di legenda.**
      Baseline, marking, gnn e seq si sovrappongono entro lo spessore del tratto:
      disegnarle in quattro colori diverse le fa sembrare un errore di rendering,
      mentre il messaggio della figura e' esattamente che **sono la stessa
      linea**. Il colore resta a chi si stacca. La legenda lo dice esplicitamente,
      quindi non nasconde nulla; la versione a cinque colori resta nel preset
      ``paper``.
    """
    cells = cell_means(df, ["forbidden"])
    datasets = [d for d in DATASET_LABELS if d in set(cells["dataset"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 4.0))
    flat = [v for v in VARIANT_ORDER if v != "checker"]

    for ax, dataset in zip(axes, datasets):
        panel = cells[cells["dataset"] == dataset]
        for variant in flat:
            line = panel[panel["variant"] == variant].sort_values("noise")
            ax.plot(line["noise"], line["forbidden_mean"], color=NEUTRAL,
                    linewidth=1.6, marker="o", markersize=5, alpha=0.75, zorder=2)
        line = panel[panel["variant"] == "checker"].sort_values("noise")
        ax.errorbar(line["noise"], line["forbidden_mean"], yerr=line["forbidden_std"],
                    color=VARIANT_COLORS["checker"], marker=VARIANT_MARKERS["checker"],
                    markersize=8, linewidth=2.6, capsize=3, elinewidth=1, zorder=3)
        ax.set_title(DATASET_LABELS[dataset], fontsize=14)
        ax.set_xticks(sorted(panel["noise"].unique()))
        ax.set_xlabel("frazione di etichette corrotte")
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("massa di probabilità vietata")

    handles = [
        plt.Line2D([], [], color=NEUTRAL, marker="o", linewidth=1.6,
                   label="baseline · marking · gnn · seq  (sovrapposte)"),
        plt.Line2D([], [], color=VARIANT_COLORS["checker"],
                   marker=VARIANT_MARKERS["checker"], linewidth=2.6,
                   label=VARIANT_LABELS["checker"]),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Il vincolo nella loss è l'unica cosa che si stacca", y=1.12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _winner_grids(df, metric: str):
    """Le tre matrici (win rate, vincitore, seed vinti) di ``winners``, in griglia.

    Estratta da :func:`plot_winner_reliability` perche' serve due volte, e con le
    righe nello stesso ordine sulle due metriche: e' il confronto fianco a fianco
    a fare l'argomento, e basta una riga fuori posto per rovinarlo.
    """
    table = winners(df, metric)
    order = [f"{d} · {a.upper()}" for d in DATASET_LABELS.values()
             for a in sorted(table["arch"].unique())]
    table["row"] = pd.Categorical(
        table["dataset"].astype(str) + " · " + table["arch"].str.upper(),
        categories=order, ordered=True,
    )
    kwargs = dict(index="row", columns="noise", observed=True)
    return (
        table.pivot_table(values="win_rate", **kwargs),
        table.pivot_table(values="migliore", aggfunc="first", **kwargs),
        table.pivot_table(values="seed vinti", aggfunc="first", **kwargs),
    )


def plot_winner_pair_slide(df, path: Path):
    """Le due heatmap di affidabilita' affiancate, stessa scala di colore.

    Da sole le due figure dicono poco: una tabella di vincitori sembra sempre una
    classifica. Affiancate con lo **stesso** ramp (0.5 = testa o croce, 1.0 =
    dieci seed su dieci) diventano l'argomento della tesi in forma visiva --
    a sinistra celle pallide e vincitori che cambiano, a destra una colonna sola
    di celle scure con sempre lo stesso nome dentro. Il colorbar condiviso e' la
    parte che non si puo' togliere: e' cio' che rende confrontabili i due lati.
    """
    grids = {m: _winner_grids(df, m) for m in ("accuracy", "forbidden")}
    titles = {
        "accuracy": "Accuratezza — vince un modello diverso quasi ogni volta",
        "forbidden": "Massa vietata — vince sempre il checker",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4))
    for ax, metric in zip(axes, ("accuracy", "forbidden")):
        grid, labels, counts = grids[metric]
        image = ax.imshow(grid.to_numpy(), cmap="Blues", vmin=0.5, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(grid.columns)))
        ax.set_xticklabels([f"noise {n:.2f}" for n in grid.columns])
        ax.set_yticks(range(len(grid.index)))
        ax.set_yticklabels(grid.index if metric == "accuracy" else [])
        for i in range(len(grid.index)):
            for j in range(len(grid.columns)):
                rate = grid.to_numpy()[i, j]
                ax.text(j, i, f"{labels.to_numpy()[i, j]}\n{counts.to_numpy()[i, j]} seed",
                        ha="center", va="center", fontsize=10,
                        color="#ffffff" if rate > 0.82 else "#0b0b0b")
        ax.set_title(titles[metric], fontsize=13, pad=10)

    bar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.02)
    bar.set_label("seed vinti contro il secondo classificato")
    fig.suptitle("Quanto tiene la vittoria, cella per cella", y=0.99, fontsize=16)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def export(preset: str) -> None:
    """Scrive tutte le figure del preset in ``docs/figures/<preset>/``."""
    out = OUT / preset
    out.mkdir(parents=True, exist_ok=True)
    rc = SLIDE_RC if preset == "slide" else PAPER_RC
    df = load_matrix(MATRIX_CSV)
    print(f"[{preset}] {len(df)} righe caricate da {MATRIX_CSV.name}")

    # Su slide una sola architettura: il pannello 3 dataset x 2 arch e' verticale
    # e su 16:9 sprecherebbe meta' larghezza. LSTM resta nelle figure paper.
    frame = df[df["arch"] == "gru"] if preset == "slide" else df
    arch_note = " (solo GRU)" if preset == "slide" else ""

    with plt.rc_context(rc):
        # --- LA figura dei risultati: i delta appaiati, centrati sullo zero.
        # Non un grafico a barre delle accuratezze: sono identiche entro 0.06 pt
        # e barre uguali fanno sembrare i dati mal misurati, non piatti.
        path = out / "paired-deltas-accuracy.png"
        plot_paired_deltas(frame, "accuracy", path=path)
        print(f"  paired-deltas-accuracy.png{arch_note}")

        # --- la conformita': l'unica regolarita' della tesi, e l'unica figura
        # del deck a colore pieno. Su slide la versione a un dataset per colonna
        # col resto in grigio; su carta i cinque colori e tutte le architetture.
        if preset == "slide":
            plot_forbidden_slide(frame, out / "forbidden-by-noise.png")
        else:
            plot_metric_by_noise(frame, "forbidden", path=out / "forbidden-by-noise.png")
        print("  forbidden-by-noise.png")

        # --- gli stessi delta sulla forbidden mass: qui il checker si stacca
        path = out / "paired-deltas-forbidden.png"
        plot_paired_deltas(frame, "forbidden", path=path)
        print("  paired-deltas-forbidden.png")

        # --- chi vince ogni cella e su quanti seed la vittoria regge. Sempre su
        # ``df`` intero e mai su ``frame``: le righe sono dataset x architettura,
        # filtrare a GRU dimezzerebbe la figura invece di raddrizzarla.
        # Su slide le due metriche affiancate (il confronto E' il messaggio);
        # su carta separate, che stanno in due pagine diverse.
        if preset == "slide":
            plot_winner_pair_slide(df, out / "winner-reliability-pair.png")
            print("  winner-reliability-pair.png")
        for metric in ("accuracy", "forbidden"):
            path = out / f"winner-reliability-{metric}.png"
            plot_winner_reliability(df, metric, path=path)
            print(f"  winner-reliability-{metric}.png")

        # --- le architetture: una per gradino (build progressiva sulla slide)
        for index, variant in enumerate(VARIANT_ORDER, start=1):
            path = out / f"ladder-{index}-{variant}.png"
            plot_variant_architecture(variant, path=path)
            print(f"  ladder-{index}-{variant}.png")

        # --- i cinque affiancati: 20 pollici di larghezza, per la tesi scritta
        # in landscape, MAI su una slide (il testo dei blocchi va sotto i 6 pt)
        if preset == "paper":
            plot_ablation_ladder(path=out / "ladder-full.png")
            print("  ladder-full.png")

    plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("slide", "paper", "both"), default="both")
    args = parser.parse_args()
    presets = ("slide", "paper") if args.preset == "both" else (args.preset,)
    for preset in presets:
        export(preset)
    print(f"\nfatto -> {OUT}")


if __name__ == "__main__":
    main()
