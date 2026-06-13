"""End-to-end orchestration of Sepsis next-activity experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

from ..config import ExperimentConfig
from ..data.prefixes import (
    ActivityVocabulary,
    extract_traces,
    make_data_loader,
    make_prefix_examples,
    split_traces,
)
from ..data.xes import read_xes
from ..learning.evaluation import EvaluationResult, evaluate_model
from ..learning.logic import EmbeddingLogicLoss, build_allowed_mask
from ..learning.models import ModelKind
from ..learning.training import TrainingResult, resolve_device, train_model
from ..process.automaton import ProcessDFA
from ..visualization.plots import (
    plot_activity_frequency,
    plot_automaton,
    plot_case_durations,
    plot_learning_curves,
    plot_model_comparison,
    plot_variants,
)
from .analysis import build_analysis_tables, summarise


@dataclass
class ExperimentRun:
    """In-memory products of one complete experiment."""

    results: pd.DataFrame
    vocabulary: ActivityVocabulary
    automaton: ProcessDFA
    training_runs: dict[str, TrainingResult]
    validation_search: pd.DataFrame | None = None


def _make_loaders(events: pd.DataFrame, config: ExperimentConfig):
    traces = extract_traces(events)
    splits = split_traces(
        traces,
        validation_fraction=config.data.validation_fraction,
        test_fraction=config.data.test_fraction,
        seed=config.seed,
    )
    vocabulary = ActivityVocabulary.from_traces(splits.train.values())

    # Every held-out activity must be representable by the training vocabulary.
    unseen = {
        activity
        for partition in (splits.validation, splits.test)
        for trace in partition.values()
        for activity in trace
        if activity not in vocabulary.class_to_id
    }
    if unseen:
        raise ValueError(f"Held-out cases contain unseen activities: {sorted(unseen)}")

    examples = {
        "train": make_prefix_examples(splits.train, vocabulary),
        "validation": make_prefix_examples(splits.validation, vocabulary),
        "test": make_prefix_examples(splits.test, vocabulary),
    }
    loaders = {
        name: make_data_loader(
            partition,
            vocabulary,
            batch_size=config.data.batch_size,
            shuffle=name == "train",
            num_workers=config.data.num_workers,
        )
        for name, partition in examples.items()
    }
    return splits, vocabulary, examples, loaders


def run_experiment(
    input_path: str | Path,
    output_dir: str | Path,
    model_dir: str | Path,
    config: ExperimentConfig | None = None,
    model_kinds: Iterable[ModelKind] = ("gru", "lstm"),
    logic_weights: Iterable[float] | None = None,
    max_validation_accuracy_drop: float = 0.01,
    max_cases: int | None = None,
    logic_mode: str = "checker",
    embedder=None,
    feature_space=None,
    constraints=None,
    verbose: bool = True,
) -> ExperimentRun:
    """Train baseline and logic-aware variants for each requested architecture.

    ``logic_mode`` chooses the logic branch: ``"checker"`` (default) penalises
    the empirical-DFA forbidden mass; ``"embedder"`` uses the learned T-LEAF
    embedding distance, in which case a trained ``embedder``, its
    ``feature_space`` and the mined ``constraints`` must be supplied. The
    baseline (logic-free) variant is always trained as a matched control, so a
    single call yields ``baseline`` + the chosen logic branch.

    When ``logic_weights`` is provided, every weight is evaluated on validation
    data. The selected model minimises forbidden probability mass while keeping
    validation accuracy within ``max_validation_accuracy_drop`` of its baseline.
    Test data is evaluated only after this selection.
    """

    config = config or ExperimentConfig()
    if logic_mode not in {"checker", "embedder"}:
        raise ValueError("logic_mode must be 'checker' or 'embedder'.")
    if logic_mode == "embedder" and (
        embedder is None or feature_space is None or constraints is None
    ):
        raise ValueError(
            "logic_mode='embedder' requires embedder, feature_space and constraints."
        )
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    model_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    events = read_xes(input_path, max_cases=max_cases)
    tables = build_analysis_tables(events)
    splits, vocabulary, examples, loaders = _make_loaders(events, config)
    automaton = ProcessDFA.from_traces(splits.train.values())
    allowed_mask = build_allowed_mask(automaton, vocabulary)

    # Persist preprocessing metadata before training so a failed run is diagnosable.
    summary = summarise(tables, input_path)
    summary["split_cases"] = {
        "train": len(splits.train),
        "validation": len(splits.validation),
        "test": len(splits.test),
    }
    summary["prefix_examples"] = {name: len(value) for name, value in examples.items()}
    summary["dfa"] = {
        "states": len(automaton.states),
        "transitions": automaton.transition_count,
        "validation_transition_coverage": automaton.transition_coverage(
            splits.validation.values()
        ),
        "test_transition_coverage": automaton.transition_coverage(splits.test.values()),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    (output_dir / "vocabulary.json").write_text(
        json.dumps(
            {"tokens": vocabulary.tokens, "activities": vocabulary.activities},
            indent=2,
        ),
        encoding="utf-8",
    )
    automaton.save(output_dir / "dfa.json")

    training_runs: dict[str, TrainingResult] = {}
    result_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    device = resolve_device(config.training.device)

    # Build the (frozen) learned-embedder logic loss once; its constraint-DFA
    # anchors are cached and reused across architectures and logic weights.
    embedding_logic = None
    if logic_mode == "embedder":
        embedder = embedder.to(device)
        feature_space = feature_space.to(device)
        embedding_logic = EmbeddingLogicLoss(
            embedder, feature_space, constraints, vocabulary
        )

    candidate_weights = None
    if logic_weights is not None:
        candidate_weights = tuple(sorted(set(float(weight) for weight in logic_weights)))
        if not candidate_weights or any(weight <= 0 for weight in candidate_weights):
            raise ValueError("logic_weights must contain one or more positive values.")
        if max_validation_accuracy_drop < 0:
            raise ValueError("max_validation_accuracy_drop cannot be negative.")

    for model_kind in model_kinds:
        weights_to_train = (
            (0.0, *candidate_weights)
            if candidate_weights is not None
            else (0.0, config.training.logic_weight)
        )
        architecture_runs: dict[float, TrainingResult] = {}

        for logic_weight in weights_to_train:
            use_logic = logic_weight > 0
            weight_label = str(logic_weight).replace(".", "p")
            run_name = (
                f"{model_kind}_logic_w{weight_label}"
                if use_logic
                else f"{model_kind}_baseline"
            )
            if verbose:
                print(f"\nTraining {run_name}")

            def report(row: dict[str, float], name: str = run_name) -> None:
                if verbose:
                    print(
                        f"{name}: epoch {int(row['epoch']):02d} "
                        f"train={row['train_loss']:.4f} "
                        f"val={row['validation_loss']:.4f}"
                    )

            run_config = replace(
                config,
                training=replace(config.training, logic_weight=logic_weight),
            )
            training = train_model(
                model_kind=model_kind,
                vocabulary=vocabulary,
                train_loader=loaders["train"],
                validation_loader=loaders["validation"],
                allowed_mask=allowed_mask,
                config=run_config,
                checkpoint_path=model_dir / f"{run_name}.pt",
                use_logic=use_logic,
                logic_mode=logic_mode,
                embedding_logic=embedding_logic,
                epoch_callback=report,
            )
            training.history.to_csv(
                output_dir / f"{run_name}_history.csv", index=False
            )
            training_runs[run_name] = training
            architecture_runs[logic_weight] = training
            validation_rows.append(
                {
                    "run": run_name,
                    "architecture": model_kind,
                    "logic_weight": logic_weight,
                    "logic_enabled": use_logic,
                    "best_epoch": training.best_epoch,
                    "stopped_early": training.stopped_early,
                    **training.validation.metrics(),
                }
            )

        baseline = architecture_runs[0.0]
        selected_weight = config.training.logic_weight
        if candidate_weights is not None:
            minimum_accuracy = (
                baseline.validation.accuracy - max_validation_accuracy_drop
            )
            eligible_weights = [
                weight
                for weight, training in architecture_runs.items()
                if training.validation.accuracy >= minimum_accuracy
            ]
            selected_weight = min(
                eligible_weights,
                key=lambda weight: (
                    architecture_runs[weight].validation.forbidden_mass,
                    -architecture_runs[weight].validation.macro_f1,
                    architecture_runs[weight].validation.loss,
                ),
            )
            if verbose:
                print(
                    f"\nSelected {model_kind} logic_weight={selected_weight:g} "
                    f"on validation (accuracy floor={minimum_accuracy:.4f})"
                )

        selected_runs = [(0.0, "baseline")]
        if selected_weight > 0:
            selected_runs.append((selected_weight, "logic_selected"))

        for logic_weight, variant in selected_runs:
            training = architecture_runs[logic_weight]
            test_result: EvaluationResult = evaluate_model(
                training.model,
                loaders["test"],
                allowed_mask.to(device),
                device,
                top_k=config.training.top_k,
            )
            result_rows.append(
                {
                    "run": f"{model_kind}_{variant}",
                    "architecture": model_kind,
                    "logic_enabled": logic_weight > 0,
                    "logic_weight": logic_weight,
                    "best_epoch": training.best_epoch,
                    "stopped_early": training.stopped_early,
                    **test_result.metrics(),
                }
            )

    validation_search = pd.DataFrame(validation_rows)
    validation_search["selected"] = False
    for architecture in validation_search["architecture"].unique():
        selected_test_rows = [
            row
            for row in result_rows
            if row["architecture"] == architecture and row["logic_enabled"]
        ]
        selected_weight = (
            selected_test_rows[0]["logic_weight"] if selected_test_rows else 0.0
        )
        validation_search.loc[
            (validation_search["architecture"] == architecture)
            & (validation_search["logic_weight"] == selected_weight),
            "selected",
        ] = True
    validation_search.to_csv(output_dir / "validation_logic_weight_search.csv", index=False)

    results = pd.DataFrame(result_rows)
    results.to_csv(output_dir / "test_results.csv", index=False)
    (output_dir / "test_results.json").write_text(
        json.dumps(results.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    plot_activity_frequency(events, output_dir / "activity_frequency.png")
    plot_case_durations(tables.cases, output_dir / "case_duration_distribution.png")
    plot_variants(tables.variants, output_dir / "top_variants.png")
    plot_automaton(automaton, output_dir / "dfa.png")
    plot_learning_curves(
        {name: run.history for name, run in training_runs.items()},
        output_dir / "learning_curves.png",
    )
    plot_model_comparison(results, output_dir / "model_comparison.png")
    return ExperimentRun(
        results, vocabulary, automaton, training_runs, validation_search
    )
