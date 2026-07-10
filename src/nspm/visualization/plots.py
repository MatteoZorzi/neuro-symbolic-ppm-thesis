"""Non-interactive plotting functions used by scripts and notebooks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Rectangle

from ..data.loader import ACTIVITY
from ..process.automaton import END, START, ProcessDFA
from ..process.ltl_constraints import ConstraintDFA, Guard, PrecedenceConstraint


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


# ---------------------------------------------------------------------------
# Symbolic-knowledge and whole-trace visualizations.
#
# These render the *data structures* the pipeline reasons over -- the LTLf
# constraint automata, the mined precedence rule base, the embedder's training
# signal, and aligned true-vs-predicted traces -- so each pipeline step is
# legible rather than tabular.
# ---------------------------------------------------------------------------


def _guard_label(guard: Guard) -> str:
    """Short edge label for a propositional guard over the activity alphabet."""

    if guard.is_true:
        return "*"
    if guard.positives:
        return " | ".join(guard.positives)
    if guard.negatives:
        return "else"
    return "?"


def plot_constraint_dfa(
    constraint_dfa: ConstraintDFA, path: str | Path, title: str | None = None
) -> None:
    """Draw one LTLf precedence DFA: states, guarded edges, accepting states.

    Green = initial state, blue = accepting, red = rejecting sink. Edge labels
    show the activity guard (``*`` = any, ``else`` = neither named activity).
    """

    states = list(constraint_dfa.node_types)
    if set(states) == {0, 1, 2}:
        positions = {0: (0.0, 0.0), 1: (1.6, 0.75), 2: (1.6, -0.75)}
    else:
        positions = {
            state: (math.cos(2 * math.pi * i / len(states)),
                    math.sin(2 * math.pi * i / len(states)))
            for i, state in enumerate(states)
        }

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for state in states:
        x, y = positions[state]
        if state == constraint_dfa.init_state:
            color, role = "#70AD47", "init"
        elif state in constraint_dfa.accepting:
            color, role = "#5B9BD5", "accept"
        else:
            color, role = "#C0504D", "sink"
        ax.scatter([x], [y], s=2800, c=color, edgecolors="black", zorder=2)
        ax.text(x, y, f"{state}\n{role}", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=3)

    for src, dst, guard in constraint_dfa.edges:
        x0, y0 = positions[src]
        x1, y1 = positions[dst]
        label = _guard_label(guard)
        if src == dst:
            ax.add_patch(Circle((x0, y0 + 0.42), 0.2, fill=False, color="#999", zorder=1))
            ax.text(x0, y0 + 0.72, label, ha="center", fontsize=8, color="#444")
        else:
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.6,
                                shrinkA=28, shrinkB=28,
                                connectionstyle="arc3,rad=0.12"),
                zorder=1,
            )
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, label,
                    ha="center", fontsize=8, color="#333")

    ax.set_title(title or "LTLf precedence constraint DFA")
    ax.set_xlim(-0.9, 2.5)
    ax.set_ylim(-1.6, 1.7)
    ax.axis("off")
    _save(fig, path)


def plot_precedence_constraints(
    constraints: Sequence[PrecedenceConstraint], path: str | Path
) -> None:
    """Render the mined precedence rule base as an ``earlier -> later`` graph."""

    if not constraints:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No precedence constraints mined", ha="center", va="center")
        ax.axis("off")
        _save(fig, path)
        return

    graph = nx.DiGraph()
    for constraint in constraints:
        graph.add_edge(constraint.earlier, constraint.later, confidence=constraint.confidence)
    positions = nx.spring_layout(graph, seed=42, k=1.6)
    confidences = [graph[u][v]["confidence"] for u, v in graph.edges]
    vmin = min(confidences)

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_nodes(graph, positions, node_color="#5B9BD5", node_size=1600, ax=ax)
    nx.draw_networkx_labels(graph, positions, font_size=8, ax=ax)
    nx.draw_networkx_edges(
        graph, positions, edge_color=confidences, edge_cmap=plt.cm.viridis,
        edge_vmin=vmin, edge_vmax=1.0, width=2.5, arrows=True, arrowsize=16,
        connectionstyle="arc3,rad=0.1", ax=ax,
    )
    scalar = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=1.0))
    scalar.set_array([])
    fig.colorbar(scalar, ax=ax, label="precedence confidence", fraction=0.04)
    ax.set_title("Mined LTLf precedence rules  (earlier → later)")
    ax.axis("off")
    _save(fig, path)


def plot_embedder_training(history: pd.DataFrame, path: str | Path) -> None:
    """Plot the embedder's triplet hinge loss and ranking accuracy per epoch."""

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["triplet_loss"], marker="o",
            color="#C0504D", label="triplet loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("triplet loss", color="#C0504D")
    ax.tick_params(axis="y", labelcolor="#C0504D")
    ax.grid(alpha=0.25)

    accuracy_axis = ax.twinx()
    accuracy_axis.plot(history["epoch"], history["triplet_accuracy"], marker="s",
                       color="#4C78A8", label="triplet accuracy")
    accuracy_axis.set_ylabel("triplet accuracy", color="#4C78A8")
    accuracy_axis.set_ylim(0, 1.02)
    accuracy_axis.tick_params(axis="y", labelcolor="#4C78A8")
    ax.set_title("T-LEAF embedder training (triplet hinge loss)")
    _save(fig, path)


def plot_trace_comparison(
    examples: Sequence[Mapping[str, object]], path: str | Path, max_examples: int = 6
) -> None:
    """Align true vs predicted traces as a coloured grid (green hit / red miss).

    Each example shows the given prefix (grey) followed by the ground-truth
    continuation (top, blue) and the model's prediction (bottom, green/red).
    """

    examples = list(examples)[:max_examples]
    if not examples:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No examples", ha="center", va="center")
        ax.axis("off")
        _save(fig, path)
        return

    def abbreviate(text: str) -> str:
        return text if len(text) <= 10 else text[:9] + "…"

    max_len = max(len(ex["prefix"]) + len(ex["true"]) for ex in examples)
    fig, axes = plt.subplots(
        len(examples), 1,
        figsize=(min(1.05 * max_len + 2, 26), 1.7 * len(examples)),
        squeeze=False,
    )
    for ax, ex in zip(axes[:, 0], examples):
        prefix = list(ex["prefix"])
        true = list(ex["true"])
        predicted = list(ex["predicted"])
        column = 0
        for activity in prefix:
            for y in (0, 1):
                ax.add_patch(Rectangle((column, y), 0.96, 0.96,
                                       facecolor="#D9D9D9", edgecolor="white"))
            ax.text(column + 0.48, 1.48, abbreviate(activity), ha="center", va="center", fontsize=7)
            ax.text(column + 0.48, 0.48, abbreviate(activity), ha="center", va="center", fontsize=7)
            column += 1
        for i, activity in enumerate(true):
            guess = predicted[i] if i < len(predicted) else ""
            ax.add_patch(Rectangle((column, 1), 0.96, 0.96, facecolor="#A6CEE3", edgecolor="white"))
            ax.add_patch(Rectangle((column, 0), 0.96, 0.96,
                                   facecolor="#70AD47" if guess == activity else "#C0504D",
                                   edgecolor="white"))
            ax.text(column + 0.48, 1.48, abbreviate(activity), ha="center", va="center", fontsize=7)
            ax.text(column + 0.48, 0.48, abbreviate(guess), ha="center", va="center", fontsize=7, color="white")
            column += 1
        ax.set_xlim(0, max_len)
        ax.set_ylim(0, 2)
        ax.set_yticks([0.48, 1.48])
        ax.set_yticklabels(["pred", "true"], fontsize=8)
        ax.set_xticks([])
        ax.set_title(f"case {ex.get('case_id', '?')}  (grey = given prefix)", fontsize=8, loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Whole-trace prediction: true vs predicted  (green = match, red = miss)")
    _save(fig, path)


def plot_fraction_curves(
    grid: pd.DataFrame,
    metrics: Sequence[str],
    path: str | Path,
    *,
    mode_col: str = "logic_mode",
    x_col: str = "n_train_cases",
    x_label: str = "training cases",
    titles: Mapping[str, str] | None = None,
    suptitle: str | None = None,
) -> None:
    """Plot one learning curve per metric vs training-set size, line per mode.

    Generic helper shared by the next-activity and whole-trace grid searches:
    the mean of each metric is drawn against the number of training cases, with
    a separate line for each value of ``mode_col`` (logic mode or decoding
    variant).
    """

    titles = dict(titles or {})
    order = sorted(grid[x_col].unique())
    modes = list(dict.fromkeys(grid[mode_col].dropna().tolist()))

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.2), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for mode in modes:
            subset = grid[grid[mode_col] == mode]
            series = subset.groupby(x_col)[metric].mean().reindex(order)
            ax.plot(order, series.values, marker="o", label=str(mode))
        ax.set_xlabel(x_label)
        ax.set_title(titles.get(metric, metric))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    if suptitle:
        fig.suptitle(suptitle)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, path)
