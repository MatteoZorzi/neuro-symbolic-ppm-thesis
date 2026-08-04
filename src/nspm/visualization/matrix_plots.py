"""Tables and figures for the final experiment matrix (``runs/final_matrix/results.csv``).

The matrix holds one row per evaluated cell of

    3 datasets x 2 architectures x 5 variants x 3 noise levels x 10 seeds = 900 runs

and is the evidence base of the thesis' quantitative claims. These helpers turn
it into the three answers a reader asks for:

* **which model wins on accuracy** -- and, crucially, *whether that win is real*
  or a coin flip between seeds (:func:`winners` reports both);
* **which model wins on conformance** -- ``violation_rate`` (argmax violations)
  and ``forbidden`` (probability mass on process-forbidden activities);
* **how each answer moves with label noise**.

Design note: every comparison here is **paired by seed**. Two variants trained
on seed 3 share initialisation and batch order, so their difference cancels most
of the run-to-run variance; averaging first and subtracting after would throw
that away and inflate the apparent spread.

Like :mod:`nspm.visualization.benchmark_plots`, plotting functions return the
:class:`~matplotlib.figure.Figure` so a notebook can show it inline; pass
``path=`` to also save it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .benchmark_plots import _maybe_save

# --------------------------------------------------------------------- schema

#: Ablation ladder order: each variant adds exactly one ingredient to the
#: previous one, which is what licenses reading a delta as that ingredient's
#: effect. Kept in one place so every table and figure orders variants the same.
VARIANT_ORDER = ("baseline", "checker", "marking", "gnn", "seq")

#: Il gradino 6 (GRNN) vive in ``runs/step6_grnn/results.csv`` e **non** entra
#: in :data:`VARIANT_ORDER`: quella tupla e' la scala pre-registrata su cui il
#: notebook asserisce 900 righe, e allargarla aggiungerebbe una categoria vuota
#: a ogni pivot della matrice. Etichette e colore stanno qui perche' una figura
#: del gradino 6 possa usarli senza reinventarli.
STEP6_VARIANT = "grnn"

#: Categorical slots 1-5 of the validated palette, assigned in ladder order and
#: never cycled: the reader learns the mapping once and it holds everywhere.
VARIANT_COLORS = {
    "baseline": "#2a78d6",
    "checker": "#eb6834",
    "marking": "#1baf7a",
    "gnn": "#eda100",
    "seq": "#e87ba4",
    "grnn": "#8a63d2",
}

#: Secondary encoding, so identity never rests on colour alone.
VARIANT_MARKERS = {
    "baseline": "o",
    "checker": "s",
    "marking": "^",
    "gnn": "D",
    "seq": "v",
    "grnn": "P",
}

VARIANT_LABELS = {
    "baseline": "1 - baseline",
    "checker": "2 - checker",
    "marking": "3 - marking",
    "gnn": "4 - gnn",
    "seq": "5 - seq",
    "grnn": "6 - grnn",
}

#: What each rung injects, and where the knowledge enters the model.
VARIANT_KNOWLEDGE = {
    "baseline": ("nessuna", "-", "il modello puramente neurale di riferimento"),
    "checker": ("DFG del processo", "loss", "penalizza la massa di probabilita' su attivita' vietate"),
    "marking": ("marking della Petri net", "feature", "lo stato del replay, vettore piatto concatenato all'RNN"),
    "gnn": ("marking + struttura", "feature", "GNN eterogenea a 2 hop sul grafo posti/transizioni"),
    "seq": ("marking + struttura + tempo", "feature", "un marking per passo, GNN per passo + GRU interna"),
    "grnn": ("marking + struttura + tempo", "feature", "GRNN di TACO: le convoluzioni stanno dentro le gate"),
}

DATASET_LABELS = {
    "Sepsis_Case": "Sepsis",
    "BPIC_2013_incidents": "BPIC 2013",
    "BPIC_2020_DomesticDeclarations": "BPIC 2020",
}

#: metric -> is a higher value better?
METRIC_HIGHER_IS_BETTER = {
    "accuracy": True,
    "macro_f1": True,
    "top3": True,
    "violation_rate": False,
    "forbidden": False,
}

METRIC_LABELS = {
    "accuracy": "accuracy",
    "macro_f1": "macro-F1",
    "top3": "top-3 accuracy",
    "violation_rate": "violation rate (argmax)",
    "forbidden": "forbidden mass (distribuzione)",
}

#: The columns that identify a cell once the seed is averaged away.
CELL_KEYS = ["dataset", "arch", "variant", "noise"]


# ----------------------------------------------------------------- caricamento


def load_matrix(path: str | Path) -> pd.DataFrame:
    """Read the matrix CSV, ordering variants along the ablation ladder."""
    df = pd.read_csv(path)
    df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER, ordered=True)
    df["dataset"] = pd.Categorical(df["dataset"], categories=list(DATASET_LABELS), ordered=True)
    return df.sort_values(CELL_KEYS + ["seed"]).reset_index(drop=True)


def model_catalogue(df: pd.DataFrame) -> pd.DataFrame:
    """One row per trained model configuration: what it is and what it adds.

    A "configuration" is an (architecture, variant) pair -- 10 in total. Each
    one is trained once per dataset x noise level x seed, so the coverage
    column says how many runs stand behind every row of the result tables.
    """
    rows = []
    for arch in sorted(df["arch"].unique()):
        for variant in VARIANT_ORDER:
            cells = df[(df["arch"] == arch) & (df["variant"] == variant)]
            if cells.empty:
                continue
            knowledge, entry, description = VARIANT_KNOWLEDGE[variant]
            rows.append({
                "modello": f"{arch}_{variant}" if variant != "baseline" else arch,
                "architettura": arch.upper(),
                "gradino": VARIANT_LABELS[variant],
                "conoscenza": knowledge,
                "entra come": entry,
                "descrizione": description,
                "run": len(cells),
                "seed": cells["seed"].nunique(),
            })
    return pd.DataFrame(rows)


def cell_means(df: pd.DataFrame, metrics: Sequence[str] | None = None) -> pd.DataFrame:
    """Average the seeds away: mean and standard deviation per cell.

    The standard deviation is the *seed spread* -- the yardstick every delta in
    this module has to be compared against before it can be called an effect.
    """
    metrics = list(metrics or METRIC_HIGHER_IS_BETTER)
    grouped = df.groupby(CELL_KEYS, observed=True)[metrics]
    means = grouped.mean().add_suffix("_mean")
    stds = grouped.std().add_suffix("_std")
    counts = grouped.size().rename("n_seed")
    return pd.concat([means, stds, counts], axis=1).reset_index()


def metric_table(df: pd.DataFrame, metric: str, arch: str | None = None) -> pd.DataFrame:
    """Mean +- seed spread of ``metric``, datasets x noise on the rows."""
    subset = df if arch is None else df[df["arch"] == arch]
    cells = cell_means(subset, [metric])
    cells["valore"] = (
        cells[f"{metric}_mean"].map("{:.4f}".format)
        + " ± "
        + cells[f"{metric}_std"].map("{:.4f}".format)
    )
    index = ["dataset", "noise"] if arch is not None else ["dataset", "arch", "noise"]
    return cells.pivot_table(
        index=index, columns="variant", values="valore", aggfunc="first", observed=True
    )


# ------------------------------------------------------------------- confronti


def paired_deltas(
    df: pd.DataFrame, metric: str, reference: str = "baseline"
) -> pd.DataFrame:
    """Per-seed difference of every variant against ``reference``.

    Pairing by seed is what makes these differences readable: the two runs being
    subtracted share the same initialisation and batch order.
    """
    keys = ["dataset", "arch", "noise", "seed"]
    ref = df[df["variant"] == reference].set_index(keys)[metric].rename("_ref")
    others = df[df["variant"] != reference].merge(ref, left_on=keys, right_index=True)
    others["delta"] = others[metric] - others["_ref"]
    return others[keys + ["variant", "delta"]].reset_index(drop=True)


def relative_change(
    df: pd.DataFrame, metric: str, reference: str = "baseline", arch: str | None = None
) -> pd.DataFrame:
    """Percentage change of ``metric`` against ``reference``, per dataset x noise.

    Conformance metrics live on very different scales across datasets and noise
    levels (0.0001 to 0.40 for ``forbidden``), so an absolute delta is not
    comparable across the grid while a relative one is -- this is the table the
    "-26/-32% of forbidden mass" claim is read from.

    With ``arch=None`` the two architectures are pooled **before** taking the
    ratio, not after: a ratio of averages, not an average of ratios. The two
    disagree by a point or two on small denominators, and the pooled form is
    the one the thesis quotes.
    """
    subset = df if arch is None else df[df["arch"] == arch]
    means = subset.groupby(["dataset", "noise", "variant"], observed=True)[metric].mean()
    table = means.unstack("variant")
    changes = table.drop(columns=reference).sub(table[reference], axis=0)
    return changes.div(table[reference], axis=0) * 100


def winners(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Best variant per (dataset, arch, noise) -- with the reliability of the win.

    Reporting only the winner would be misleading when variants sit inside each
    other's seed spread, so every row also carries:

    ``margine``
        how far ahead of the runner-up the winner's mean is;
    ``spread``
        the typical seed-to-seed standard deviation in that cell;
    ``seed vinti``
        how many of the seeds the winner actually beats the runner-up on, head
        to head. **This is the column that decides whether the win is real**:
        around half means the ranking is a coin flip, regardless of the margin.
    """
    higher_is_better = METRIC_HIGHER_IS_BETTER[metric]
    rows = []
    for (dataset, arch, noise), cell in df.groupby(["dataset", "arch", "noise"], observed=True):
        per_variant = cell.pivot_table(
            index="seed", columns="variant", values=metric, observed=True
        )
        means = per_variant.mean()
        ranked = means.sort_values(ascending=not higher_is_better)
        best, second = ranked.index[0], ranked.index[1]
        head_to_head = per_variant[best] - per_variant[second]
        wins = (head_to_head > 0).sum() if higher_is_better else (head_to_head < 0).sum()
        rows.append({
            "dataset": DATASET_LABELS.get(dataset, dataset),
            "arch": arch,
            "noise": noise,
            "migliore": VARIANT_LABELS[best],
            "valore": means[best],
            "secondo": VARIANT_LABELS[second],
            "margine": abs(means[best] - means[second]),
            "spread": per_variant.std().mean(),
            "seed vinti": f"{int(wins)}/{len(per_variant)}",
            "win_rate": wins / len(per_variant),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- figure


def _panel_grid(df: pd.DataFrame, rows: str, cols: str, size=(3.4, 2.6)):
    row_values = list(dict.fromkeys(df[rows].dropna()))
    col_values = sorted(df[cols].unique())
    fig, axes = plt.subplots(
        len(row_values), len(col_values),
        figsize=(size[0] * len(col_values), size[1] * len(row_values)),
        squeeze=False, sharex=True,
    )
    return fig, axes, row_values, col_values


def plot_metric_by_noise(
    df: pd.DataFrame, metric: str, path: str | Path | None = None
) -> plt.Figure:
    """Level of ``metric`` against noise, one panel per dataset x architecture.

    Error bars are +-1 seed standard deviation: when they overlap across
    variants, the ordering of the means carries no information.
    """
    cells = cell_means(df, [metric])
    fig, axes, datasets, archs = _panel_grid(cells, "dataset", "arch")
    for r, dataset in enumerate(datasets):
        for c, arch in enumerate(archs):
            ax = axes[r][c]
            panel = cells[(cells["dataset"] == dataset) & (cells["arch"] == arch)]
            for variant in VARIANT_ORDER:
                line = panel[panel["variant"] == variant].sort_values("noise")
                if line.empty:
                    continue
                ax.errorbar(
                    line["noise"], line[f"{metric}_mean"], yerr=line[f"{metric}_std"],
                    color=VARIANT_COLORS[variant], marker=VARIANT_MARKERS[variant],
                    markersize=6, linewidth=2, capsize=3, elinewidth=1,
                    label=VARIANT_LABELS[variant],
                )
            ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)} · {arch.upper()}", fontsize=10)
            ax.set_xticks(sorted(panel["noise"].unique()))
            ax.grid(alpha=0.25, linewidth=0.6)
            ax.set_axisbelow(True)
            if c == 0:
                ax.set_ylabel(METRIC_LABELS.get(metric, metric))
            if r == len(datasets) - 1:
                ax.set_xlabel("frazione di etichette corrotte")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{METRIC_LABELS.get(metric, metric)} — media su 10 seed", y=1.06)
    fig.tight_layout()
    return _maybe_save(fig, path)


def plot_paired_deltas(
    df: pd.DataFrame, metric: str, reference: str = "baseline",
    path: str | Path | None = None,
) -> plt.Figure:
    """Per-seed deltas against ``reference``, in percentage points.

    One dot per seed per architecture, the diamond is the mean. The vertical
    line at zero is the only reference that matters: a variant helps only if
    its cloud sits clearly on one side of it.
    """
    deltas = paired_deltas(df, metric, reference)
    deltas["delta_pt"] = deltas["delta"] * 100
    fig, axes, datasets, noises = _panel_grid(deltas, "dataset", "noise", size=(3.6, 2.4))
    variants = [v for v in VARIANT_ORDER if v != reference]
    rng = np.random.default_rng(0)  # jitter riproducibile
    for r, dataset in enumerate(datasets):
        for c, noise in enumerate(noises):
            ax = axes[r][c]
            panel = deltas[(deltas["dataset"] == dataset) & (deltas["noise"] == noise)]
            for y, variant in enumerate(variants):
                points = panel[panel["variant"] == variant]["delta_pt"]
                ax.scatter(points, y + rng.uniform(-0.16, 0.16, len(points)),
                           s=16, alpha=0.45, color=VARIANT_COLORS[variant],
                           edgecolors="none")
                ax.scatter(points.mean(), y, s=110, marker="D",
                           color=VARIANT_COLORS[variant], edgecolors="white",
                           linewidths=1.4, zorder=3)
            ax.axvline(0, color="#52514e", linewidth=1.2, zorder=1)
            ax.set_yticks(range(len(variants)))
            ax.set_yticklabels([VARIANT_LABELS[v] for v in variants] if c == 0 else [])
            ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)} · noise {noise:.2f}", fontsize=10)
            ax.grid(axis="x", alpha=0.25, linewidth=0.6)
            ax.set_axisbelow(True)
            if r == len(datasets) - 1:
                ax.set_xlabel(f"Δ {METRIC_LABELS.get(metric, metric)} (punti)")
    fig.suptitle(
        f"{METRIC_LABELS.get(metric, metric)}: ogni variante meno «{reference}», appaiata per seed",
        y=1.02,
    )
    fig.tight_layout()
    return _maybe_save(fig, path)


def plot_winner_reliability(
    df: pd.DataFrame, metric: str, path: str | Path | None = None
) -> plt.Figure:
    """Who wins each cell, and on how many seeds the win actually holds.

    The colour is the head-to-head win rate against the runner-up, on the
    sequential blue ramp: pale cells are ties dressed up as rankings, dark
    cells are wins that survive the seed spread.
    """
    table = winners(df, metric)
    # righe nell'ordine dei dataset del progetto, non alfabetico
    order = [f"{d} · {a.upper()}" for d in DATASET_LABELS.values()
             for a in sorted(table["arch"].unique())]
    table["row"] = pd.Categorical(
        table["dataset"].astype(str) + " · " + table["arch"].str.upper(),
        categories=order, ordered=True,
    )
    grid = table.pivot_table(index="row", columns="noise", values="win_rate", observed=True)
    labels = table.pivot_table(index="row", columns="noise", values="migliore",
                               aggfunc="first", observed=True)
    counts = table.pivot_table(index="row", columns="noise", values="seed vinti",
                               aggfunc="first", observed=True)

    fig, ax = plt.subplots(figsize=(2.6 * len(grid.columns) + 2.2, 0.62 * len(grid) + 2.0))
    image = ax.imshow(grid.to_numpy(), cmap="Blues", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels([f"noise {n:.2f}" for n in grid.columns])
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index)
    for i in range(len(grid.index)):
        for j in range(len(grid.columns)):
            rate = grid.to_numpy()[i, j]
            ax.text(j, i, f"{labels.to_numpy()[i, j]}\n{counts.to_numpy()[i, j]} seed",
                    ha="center", va="center", fontsize=9,
                    color="#ffffff" if rate > 0.82 else "#0b0b0b")
    bar = fig.colorbar(image, ax=ax, shrink=0.85)
    bar.set_label("seed vinti contro il secondo classificato")
    ax.set_title(
        f"Chi vince su {METRIC_LABELS.get(metric, metric)} — e quanto tiene la vittoria",
        pad=12,
    )
    fig.tight_layout()
    return _maybe_save(fig, path)
