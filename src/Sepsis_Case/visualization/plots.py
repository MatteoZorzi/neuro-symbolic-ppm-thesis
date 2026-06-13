"""Non-interactive plotting functions used by scripts and notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from ..data.xes import ACTIVITY
from ..process.automaton import END, START, ProcessDFA


def _save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_activity_frequency(
    events: pd.DataFrame, path: str | Path, top_n: int = 20
) -> None:
    counts = events[ACTIVITY].value_counts().head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot.barh(ax=ax)
    ax.set(title="Most frequent activities", xlabel="Events", ylabel="")
    _save(fig, path)


def plot_case_durations(cases: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(cases["duration_hours"].dropna(), bins=40)
    ax.set(
        title="Case duration distribution",
        xlabel="Duration (hours)",
        ylabel="Cases",
    )
    _save(fig, path)


def plot_variants(
    variants: pd.DataFrame, path: str | Path, top_n: int = 20
) -> None:
    selected = variants.head(top_n).sort_values("count")
    labels = [
        value if len(value) <= 90 else value[:87] + "..."
        for value in selected["variant"]
    ]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(labels, selected["count"])
    ax.set(title="Most frequent trace variants", xlabel="Cases")
    _save(fig, path)


def plot_automaton(automaton: ProcessDFA, path: str | Path) -> None:
    """Render the empirical DFA with distinct start and terminal states."""

    graph = automaton.to_networkx()
    positions = nx.spring_layout(graph, seed=42, k=1.2)
    colors = [
        "#70AD47" if node == START else "#C0504D" if node == END else "#5B9BD5"
        for node in graph.nodes
    ]
    fig, ax = plt.subplots(figsize=(15, 11))
    nx.draw_networkx(
        graph,
        pos=positions,
        node_color=colors,
        node_size=1800,
        font_size=8,
        arrows=True,
        arrowsize=14,
        ax=ax,
    )
    ax.set_title("Training-set empirical DFA")
    ax.axis("off")
    _save(fig, path)


def plot_learning_curves(
    histories: Mapping[str, pd.DataFrame], path: str | Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, history in histories.items():
        axes[0].plot(history["epoch"], history["train_loss"], label=label)
        axes[1].plot(
            history["epoch"], history["validation_loss"], label=label
        )
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Validation loss", xlabel="Epoch", ylabel="Loss")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.25)
    _save(fig, path)


def plot_model_comparison(results: pd.DataFrame, path: str | Path) -> None:
    """Compare baseline and logic models without mixing incompatible scales."""

    required = {
        "run",
        "architecture",
        "logic_enabled",
        "accuracy",
        "macro_f1",
        "top_k_accuracy",
        "forbidden_mass",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Missing comparison columns: {sorted(missing)}")

    ordered = results.sort_values(["architecture", "logic_enabled"])
    labels = [
        f"{row.architecture.upper()}\n"
        f"{'Logic' if row.logic_enabled else 'Baseline'}"
        for row in ordered.itertuples()
    ]
    x = np.arange(len(ordered))
    width = 0.24
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for offset, metric, label, color in (
        (-width, "accuracy", "Accuracy", "#4C78A8"),
        (0.0, "macro_f1", "Macro F1", "#F58518"),
        (width, "top_k_accuracy", "Top-3 accuracy", "#54A24B"),
    ):
        axes[0].bar(x + offset, ordered[metric], width, label=label, color=color)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Predictive quality")
    axes[0].set_ylabel("Score")
    axes[0].legend()

    axes[1].bar(x, ordered["forbidden_mass"], color="#B279A2")
    axes[1].set_title("Process non-conformance")
    axes[1].set_ylabel("Forbidden probability mass (lower is better)")

    delta_rows = []
    for architecture, group in ordered.groupby("architecture"):
        baseline = group.loc[~group["logic_enabled"]]
        logic = group.loc[group["logic_enabled"]]
        if baseline.empty or logic.empty:
            continue
        baseline = baseline.iloc[0]
        logic = logic.iloc[0]
        delta_rows.append(
            {
                "architecture": architecture.upper(),
                "accuracy": logic["accuracy"] - baseline["accuracy"],
                "macro_f1": logic["macro_f1"] - baseline["macro_f1"],
                "top_k_accuracy": (
                    logic["top_k_accuracy"] - baseline["top_k_accuracy"]
                ),
                "forbidden_reduction": (
                    1 - logic["forbidden_mass"] / baseline["forbidden_mass"]
                    if baseline["forbidden_mass"] > 0
                    else 0.0
                ),
            }
        )
    deltas = pd.DataFrame(delta_rows)
    if not deltas.empty:
        delta_x = np.arange(len(deltas))
        axes[2].bar(
            delta_x - width / 2,
            deltas["accuracy"] * 100,
            width,
            label="Accuracy delta (pp)",
            color="#4C78A8",
        )
        axes[2].bar(
            delta_x + width / 2,
            deltas["macro_f1"] * 100,
            width,
            label="Macro F1 delta (pp)",
            color="#F58518",
        )
        conformance_axis = axes[2].twinx()
        conformance_axis.plot(
            delta_x,
            deltas["forbidden_reduction"] * 100,
            marker="o",
            linewidth=2,
            label="Forbidden mass reduction (%)",
            color="#B279A2",
        )
        conformance_axis.set_ylabel("Forbidden mass reduction (%)", color="#B279A2")
        conformance_axis.tick_params(axis="y", labelcolor="#B279A2")
        conformance_axis.set_ylim(bottom=0)
        axes[2].set_xticks(delta_x, deltas["architecture"])
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title("Logic model change vs baseline")
    axes[2].set_ylabel("Predictive metric delta (percentage points)")
    handles, legend_labels = axes[2].get_legend_handles_labels()
    if not deltas.empty:
        secondary_handles, secondary_labels = (
            conformance_axis.get_legend_handles_labels()
        )
        handles += secondary_handles
        legend_labels += secondary_labels
    axes[2].legend(handles, legend_labels, fontsize=8)

    for ax in axes[:2]:
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", rotation=0)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    _save(fig, path)
