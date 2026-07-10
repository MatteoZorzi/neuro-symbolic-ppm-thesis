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
]

