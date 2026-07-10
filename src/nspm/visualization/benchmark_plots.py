"""Plots for the corruption-grid benchmark (``runs/benchmark_corruption.csv``).

These functions read the tidy benchmark table produced by
:func:`nspm.pipeline.benchmark.run_corruption_grid` -- one row per evaluated
``(dataset, architecture, variant, noise_level)`` cell -- and turn it into the
figures the thesis argues from:

* how the three event logs differ in difficulty (dataset comparison);
* whether the neural model degrades as training labels get noisier;
* whether the **clean symbolic layer** keeps predictions process-conformant
  under that noise (the T-LEAF claim: ``baseline`` drifts, ``checker`` softens
  it, ``baseline_mask`` hard-zeros forbidden mass);
* the accuracy vs. conformance trade-off and the metric correlations.

Unlike the structural plots in :mod:`nspm.visualization.plots`, every function
here **returns the** :class:`~matplotlib.figure.Figure` so a notebook can show
it inline; pass ``path=`` to also save it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Stable colour / order per logic variant, reused across every figure so the
# reader learns "red = unconstrained baseline, green = hard mask, blue = soft
# checker loss" once and it holds throughout the notebook.
VARIANT_COLORS: dict[str, str] = {
    "baseline": "#C0504D",
    "baseline_mask": "#70AD47",
    "checker": "#4C78A8",
    "embedder": "#F58518",
}
VARIANT_ORDER = ("baseline", "baseline_mask", "checker", "embedder")
VARIANT_LABELS = {
    "baseline": "baseline (no logic)",
    "baseline_mask": "baseline + mask",
    "checker": "checker (logic loss)",
    "embedder": "embedder (T-LEAF)",
}


def _maybe_save(fig: plt.Figure, path: str | Path | None) -> plt.Figure:
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def _variants_in(df: pd.DataFrame) -> list[str]:
    present = set(df["variant"].unique())
    return [v for v in VARIANT_ORDER if v in present]


def _datasets_in(df: pd.DataFrame) -> list[str]:
    # Preserve first-seen order so facets line up with the CSV / story.
    return list(dict.fromkeys(df["dataset"].tolist()))


def _short_dataset(name: str) -> str:
    return name.replace("_", " ")


# ---------------------------------------------------------------------------
# 1. How hard is each dataset?  (clean, noise-free baseline)
# ---------------------------------------------------------------------------
def plot_dataset_overview(
    df: pd.DataFrame,
    path: str | Path | None = None,
    *,
    metrics: Sequence[str] = ("accuracy", "macro_f1", "top_k_accuracy"),
) -> plt.Figure:
    """Grouped bars of clean-baseline predictive quality per dataset.

    Uses the ``noise_level == 0`` ``baseline`` rows (averaged over
    architectures) so each bar is the *uncorrupted* difficulty of the log, and
    annotates the DFA size + training volume that contextualise it.
    """

    clean = df[(df["noise_level"] == 0.0) & (df["variant"] == "baseline")]
    datasets = _datasets_in(clean)
    agg = clean.groupby("dataset")[list(metrics)].mean()
    context = clean.groupby("dataset")[["dfa_states", "n_train_examples"]].max()

    x = np.arange(len(datasets))
    width = 0.8 / len(metrics)
    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]

    fig, ax = plt.subplots(figsize=(1.9 * len(datasets) + 4, 5))
    for i, metric in enumerate(metrics):
        offset = (i - (len(metrics) - 1) / 2) * width
        values = [agg.loc[d, metric] for d in datasets]
        bars = ax.bar(x + offset, values, width, label=metric.replace("_", " "),
                      color=palette[i % len(palette)])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{_short_dataset(d)}\n(DFA {int(context.loc[d, 'dfa_states'])} states · "
         f"{int(context.loc[d, 'n_train_examples']):,} prefixes)" for d in datasets],
        fontsize=9,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Dataset difficulty — clean baseline (noise = 0), mean over GRU/LSTM")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _maybe_save(fig, path)


# ---------------------------------------------------------------------------
# 2. Does the model degrade as labels get noisier?  (predictive metric)
# ---------------------------------------------------------------------------
def plot_noise_robustness(
    df: pd.DataFrame,
    path: str | Path | None = None,
    *,
    metric: str = "accuracy",
) -> plt.Figure:
    """One facet per dataset: ``metric`` vs. noise level, a line per variant.

    Lines are the mean over architectures. Shows how far predictive quality
    moves as a growing fraction of training targets is corrupted.
    """

    datasets = _datasets_in(df)
    variants = _variants_in(df)
    noises = sorted(df["noise_level"].unique())

    fig, axes = plt.subplots(
        1, len(datasets), figsize=(4.6 * len(datasets), 4.4), squeeze=False, sharey=True
    )
    for ax, dataset in zip(axes[0], datasets):
        sub = df[df["dataset"] == dataset]
        for variant in variants:
            series = (
                sub[sub["variant"] == variant]
                .groupby("noise_level")[metric].mean().reindex(noises)
            )
            ax.plot(noises, series.values, marker="o",
                    color=VARIANT_COLORS.get(variant, "#555"),
                    label=VARIANT_LABELS.get(variant, variant))
        ax.set_title(_short_dataset(dataset), fontsize=10)
        ax.set_xlabel("noise level (fraction of labels corrupted)")
        ax.set_xticks(noises)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel(metric.replace("_", " "))
    axes[0][-1].legend(fontsize=8, loc="best")
    fig.suptitle(f"Noise robustness — test {metric.replace('_', ' ')} vs. label corruption")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _maybe_save(fig, path)


# ---------------------------------------------------------------------------
# 3. Does the symbolic layer keep predictions conformant?  (the T-LEAF claim)
# ---------------------------------------------------------------------------
def plot_conformance_by_noise(
    df: pd.DataFrame,
    path: str | Path | None = None,
    *,
    metric: str = "forbidden_mass",
) -> plt.Figure:
    """One facet per dataset: process non-conformance vs. noise, line per variant.

    ``forbidden_mass`` is the probability the model puts on DFA-forbidden next
    activities (lower = more conformant). The expected story: ``baseline`` rises
    with noise, ``checker`` (soft logic loss) stays lower, ``baseline_mask``
    sits at exactly 0 because forbidden classes are removed at inference.
    """

    datasets = _datasets_in(df)
    variants = _variants_in(df)
    noises = sorted(df["noise_level"].unique())

    fig, axes = plt.subplots(
        1, len(datasets), figsize=(4.6 * len(datasets), 4.4), squeeze=False, sharey=True
    )
    for ax, dataset in zip(axes[0], datasets):
        sub = df[df["dataset"] == dataset]
        for variant in variants:
            series = (
                sub[sub["variant"] == variant]
                .groupby("noise_level")[metric].mean().reindex(noises)
            )
            ax.plot(noises, series.values, marker="o",
                    color=VARIANT_COLORS.get(variant, "#555"),
                    label=VARIANT_LABELS.get(variant, variant))
        ax.set_title(_short_dataset(dataset), fontsize=10)
        ax.set_xlabel("noise level")
        ax.set_xticks(noises)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel(f"{metric.replace('_', ' ')}  (lower = more conformant)")
    axes[0][-1].legend(fontsize=8, loc="best")
    fig.suptitle("Process conformance under noise — forbidden probability mass per variant")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _maybe_save(fig, path)


# ---------------------------------------------------------------------------
# 4. Does the logic loss earn its keep?  (conformance gain vs accuracy cost)
# ---------------------------------------------------------------------------
def plot_logic_effect(
    df: pd.DataFrame,
    path: str | Path | None = None,
    *,
    reference: str = "baseline",
    treatment: str = "checker",
) -> plt.Figure:
    """Per dataset: forbidden-mass reduction vs. accuracy change of the logic loss.

    Averaged over the *noisy* cells (noise > 0) and architectures, compares the
    ``treatment`` variant (default ``checker``) against ``reference`` (default
    ``baseline``): left axis bars = % reduction in forbidden mass (the win),
    right axis dots = accuracy change in percentage points (the cost). The
    thesis claim holds when the bars are large and positive and the dots sit
    near zero.
    """

    noisy = df[df["noise_level"] > 0]
    datasets = _datasets_in(noisy)
    rows = []
    for dataset in datasets:
        sub = noisy[noisy["dataset"] == dataset]
        ref = sub[sub["variant"] == reference]
        trt = sub[sub["variant"] == treatment]
        if ref.empty or trt.empty:
            continue
        ref_fm, trt_fm = ref["forbidden_mass"].mean(), trt["forbidden_mass"].mean()
        rows.append({
            "dataset": dataset,
            "fm_reduction": (1 - trt_fm / ref_fm) * 100 if ref_fm > 0 else 0.0,
            "acc_delta": (trt["accuracy"].mean() - ref["accuracy"].mean()) * 100,
        })
    summary = pd.DataFrame(rows)

    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(2.0 * len(summary) + 4, 5))
    bars = ax.bar(x, summary["fm_reduction"], width=0.5,
                  color=VARIANT_COLORS.get(treatment, "#4C78A8"), alpha=0.85,
                  label=f"forbidden-mass reduction ({treatment} vs {reference})")
    ax.bar_label(bars, fmt="%.0f%%", fontsize=9, padding=2)
    ax.set_ylabel("forbidden-mass reduction (%)  — higher is better")
    ax.set_ylim(0, max(100, summary["fm_reduction"].max() * 1.15))
    ax.axhline(0, color="black", lw=0.8)

    cost = ax.twinx()
    cost.plot(x, summary["acc_delta"], "D", color="#333", markersize=9,
              label="accuracy change (pp)")
    for xi, d in zip(x, summary["acc_delta"]):
        cost.annotate(f"{d:+.1f}pp", (xi, d), textcoords="offset points",
                      xytext=(8, 0), fontsize=8, va="center")
    cost.set_ylabel("accuracy change (percentage points)")
    cost.axhline(0, color="#888", lw=0.8, ls="--")
    span = max(1.0, abs(summary["acc_delta"]).max() * 2.5)
    cost.set_ylim(-span, span)

    ax.set_xticks(x)
    ax.set_xticklabels([_short_dataset(d) for d in summary["dataset"]], fontsize=9)
    ax.set_title(f"Does the logic loss pay off?  {treatment} vs {reference}, "
                 "mean over noisy cells")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = cost.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8)
    fig.tight_layout()
    return _maybe_save(fig, path)


# ---------------------------------------------------------------------------
# 5. The accuracy <-> conformance trade-off, every cell at once
# ---------------------------------------------------------------------------
def plot_accuracy_conformance_tradeoff(
    df: pd.DataFrame,
    path: str | Path | None = None,
) -> plt.Figure:
    """Scatter of every cell: accuracy (x) vs. forbidden mass (y).

    Colour = variant, marker = dataset, point size grows with noise level. The
    ideal corner is bottom-right (accurate **and** conformant); the logic
    variants should sit lower than the bare baseline at matched accuracy.
    """

    datasets = _datasets_in(df)
    variants = _variants_in(df)
    markers = ["o", "s", "^", "D", "P", "X"]
    noises = sorted(df["noise_level"].unique())

    def size_for(noise: float) -> float:
        rank = noises.index(noise)
        return 40 + 55 * rank

    fig, ax = plt.subplots(figsize=(9, 6))
    for variant in variants:
        for marker, dataset in zip(markers, datasets):
            sub = df[(df["variant"] == variant) & (df["dataset"] == dataset)]
            if sub.empty:
                continue
            ax.scatter(sub["accuracy"], sub["forbidden_mass"],
                       s=[size_for(n) for n in sub["noise_level"]],
                       color=VARIANT_COLORS.get(variant, "#555"),
                       marker=marker, alpha=0.75, edgecolors="white", linewidths=0.6)

    variant_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=VARIANT_COLORS.get(v, "#555"),
                   label=VARIANT_LABELS.get(v, v)) for v in variants
    ]
    dataset_handles = [
        plt.Line2D([], [], marker=m, linestyle="", color="#555", label=_short_dataset(d))
        for m, d in zip(markers, datasets)
    ]
    size_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="#999",
                   markersize=(size_for(n) / 40) ** 0.5 * 4, label=f"noise {n:g}")
        for n in noises
    ]
    legend1 = ax.legend(handles=variant_handles, title="variant",
                        loc="upper left", fontsize=8)
    legend2 = ax.legend(handles=dataset_handles, title="dataset",
                        loc="upper center", fontsize=8)
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    ax.legend(handles=size_handles, title="point size", loc="upper right", fontsize=8)

    ax.set_xlabel("accuracy  (higher is better →)")
    ax.set_ylabel("forbidden mass  (↓ lower is more conformant)")
    ax.set_title("Accuracy vs. conformance — every (dataset, variant, noise) cell")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _maybe_save(fig, path)


# ---------------------------------------------------------------------------
# 6. What moves together?  (metric correlation)
# ---------------------------------------------------------------------------
def plot_metric_correlation(
    df: pd.DataFrame,
    path: str | Path | None = None,
    *,
    columns: Sequence[str] | None = None,
    method: str = "spearman",
) -> plt.Figure:
    """Correlation heatmap over the benchmark's numeric design + outcome columns.

    Defaults to a curated set linking the *design* axes (noise, corrupted
    count, training volume, DFA size) to the *outcomes* (loss, accuracy,
    macro-F1, conformance). Spearman by default so monotone-but-nonlinear
    relations (e.g. noise → forbidden mass) show up.
    """

    default_cols = [
        "noise_level", "n_train_corrupted", "n_train_examples", "dfa_states",
        "loss", "accuracy", "macro_f1", "macro_precision", "macro_recall",
        "top_k_accuracy", "violation_rate", "forbidden_mass", "train_runtime_s",
    ]
    cols = [c for c in (columns or default_cols)
            if c in df.columns and df[c].nunique() > 1]
    corr = df[cols].corr(method=method)

    fig, ax = plt.subplots(figsize=(0.7 * len(cols) + 2, 0.7 * len(cols) + 1))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels([c.replace("_", " ") for c in cols], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([c.replace("_", " ") for c in cols], fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            value = corr.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    color="white" if abs(value) > 0.55 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"{method} correlation")
    ax.set_title(f"Metric correlation ({method}) across all benchmark cells")
    fig.tight_layout()
    return _maybe_save(fig, path)


# ---------------------------------------------------------------------------
# Optional: data-scarcity learning curve (only if the CSV carries train_fraction)
# ---------------------------------------------------------------------------
def plot_data_scarcity(
    df: pd.DataFrame,
    path: str | Path | None = None,
    *,
    metric: str = "accuracy",
) -> plt.Figure | None:
    """Learning curve over training-set size, if the grid varied ``train_fraction``.

    Returns ``None`` (and draws nothing) when the benchmark file has a single
    training fraction — the corruption-only sweep does not vary data volume, so
    data-scarcity is reported by a separate grid.
    """

    # Only a real training-size axis counts; ``n_train_examples`` differs *across
    # datasets* rather than across a scarcity sweep, so it must not trigger this.
    frac_col = next((c for c in ("train_fraction", "n_train_cases")
                     if c in df.columns and df.groupby("dataset")[c].nunique().max() > 1), None)
    if frac_col is None:
        return None

    datasets = _datasets_in(df)
    variants = _variants_in(df)
    clean = df[df["noise_level"] == 0.0]
    order = sorted(clean[frac_col].unique())

    fig, axes = plt.subplots(
        1, len(datasets), figsize=(4.6 * len(datasets), 4.4), squeeze=False, sharey=True
    )
    for ax, dataset in zip(axes[0], datasets):
        sub = clean[clean["dataset"] == dataset]
        for variant in variants:
            series = (sub[sub["variant"] == variant]
                      .groupby(frac_col)[metric].mean().reindex(order))
            ax.plot(order, series.values, marker="o",
                    color=VARIANT_COLORS.get(variant, "#555"),
                    label=VARIANT_LABELS.get(variant, variant))
        ax.set_title(_short_dataset(dataset), fontsize=10)
        ax.set_xlabel(frac_col.replace("_", " "))
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel(metric.replace("_", " "))
    axes[0][-1].legend(fontsize=8)
    fig.suptitle(f"Data scarcity (clean labels) — {metric.replace('_', ' ')} vs. training size")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _maybe_save(fig, path)


def summarize_benchmark(df: pd.DataFrame) -> Mapping[str, object]:
    """Small dict describing the grid, for a notebook header cell."""

    return {
        "datasets": _datasets_in(df),
        "architectures": sorted(df["architecture"].unique()),
        "variants": _variants_in(df),
        "noise_levels": sorted(df["noise_level"].unique()),
        "n_rows": len(df),
        "seeds": sorted(df["seed"].unique()) if "seed" in df else [],
    }