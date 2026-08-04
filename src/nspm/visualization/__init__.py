"""Plots for exploratory analysis, automata and model comparisons."""

from .benchmark_plots import (
    plot_accuracy_conformance_tradeoff,
    plot_conformance_by_noise,
    plot_data_scarcity,
    plot_dataset_overview,
    plot_logic_effect,
    plot_metric_correlation,
    plot_noise_robustness,
    summarize_benchmark,
)
from .architecture_plots import plot_ablation_ladder, plot_variant_architecture
from .matrix_plots import (
    cell_means,
    load_matrix,
    metric_table,
    model_catalogue,
    paired_deltas,
    plot_metric_by_noise,
    plot_paired_deltas,
    plot_winner_reliability,
    winners,
)
from .plots import (
    plot_activity_frequency,
    plot_automaton,
    plot_case_durations,
    plot_learning_curves,
    plot_model_comparison,
    plot_variants,
)

__all__ = [
    "plot_activity_frequency",
    "plot_automaton",
    "plot_case_durations",
    "plot_learning_curves",
    "plot_model_comparison",
    "plot_variants",
    # benchmark-grid analysis
    "plot_dataset_overview",
    "plot_noise_robustness",
    "plot_conformance_by_noise",
    "plot_logic_effect",
    "plot_accuracy_conformance_tradeoff",
    "plot_metric_correlation",
    "plot_data_scarcity",
    "summarize_benchmark",
    # matrice finale (900 run)
    "load_matrix",
    "model_catalogue",
    "cell_means",
    "metric_table",
    "paired_deltas",
    "winners",
    "plot_metric_by_noise",
    "plot_paired_deltas",
    "plot_winner_reliability",
    # struttura dei modelli
    "plot_ablation_ladder",
    "plot_variant_architecture",
]

