"""Command-line interface for analysis and model experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .config import DEFAULT_DATASET, ExperimentConfig, ProjectPaths
from .pipeline.analysis import analyse
from .pipeline.experiment import run_experiment
from .pipeline.run_manager import create_next_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Neuro-symbolic PPM analysis pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analysis_parser = subparsers.add_parser("analyze", help="Run exploratory analysis")
    analysis_parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    analysis_parser.add_argument(
        "--input", type=Path, help="Explicit XES log; defaults to the dataset's log."
    )
    analysis_parser.add_argument(
        "--output-dir", type=Path, help="Defaults to datasets/<dataset-name>/analysis."
    )
    analysis_parser.add_argument("--max-cases", type=int)
    analysis_parser.add_argument("--top-n", type=int, default=20)
    analysis_parser.add_argument("--no-events-csv", action="store_true")

    experiment_parser = subparsers.add_parser(
        "experiment", help="Train baseline and logic-aware recurrent models"
    )
    experiment_parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    experiment_parser.add_argument(
        "--input", type=Path, help="Explicit XES log; defaults to the dataset's log."
    )
    experiment_parser.add_argument(
        "--runs-dir", type=Path, help="Run output root; defaults to ./runs."
    )
    experiment_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Explicit output directory; disables automatic run numbering.",
    )
    experiment_parser.add_argument(
        "--model-dir",
        type=Path,
        help="Explicit checkpoint directory; defaults to <run>/models.",
    )
    experiment_parser.add_argument(
        "--model",
        choices=("gru", "lstm", "transformer", "both", "all"),
        default="both",
    )
    experiment_parser.add_argument("--epochs", type=int, default=12)
    experiment_parser.add_argument(
        "--early-stopping-patience", type=int, default=5
    )
    experiment_parser.add_argument(
        "--early-stopping-min-delta", type=float, default=1e-4
    )
    experiment_parser.add_argument(
        "--early-stopping-min-epochs", type=int, default=5
    )
    experiment_parser.add_argument("--batch-size", type=int, default=128)
    experiment_parser.add_argument("--logic-weight", type=float, default=0.5)
    experiment_parser.add_argument(
        "--task-loss-weight",
        type=float,
        default=1.0,
        help="Coefficient applied to cross-entropy during training.",
    )
    experiment_parser.add_argument(
        "--logic-weights",
        type=float,
        nargs="+",
        help="Validation-search grid; for example: 0.05 0.1 0.25 0.5 1.0",
    )
    experiment_parser.add_argument(
        "--max-validation-accuracy-drop",
        type=float,
        default=0.01,
        help="Maximum validation accuracy loss accepted during logic-weight selection.",
    )
    experiment_parser.add_argument("--device", default="auto")
    experiment_parser.add_argument("--seed", type=int, default=42)
    experiment_parser.add_argument("--max-cases", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    paths = ProjectPaths.from_root(Path.cwd(), args.dataset_name)
    if args.command == "analyze":
        summary = analyse(
            args.input or paths.dataset,
            args.output_dir or paths.analysis_dir,
            max_cases=args.max_cases,
            top_n=args.top_n,
            save_events=not args.no_events_csv,
        )
        print(json.dumps(summary, indent=2))
        return

    config = ExperimentConfig(seed=args.seed)
    config = replace(
        config,
        data=replace(config.data, batch_size=args.batch_size),
        training=replace(
            config.training,
            epochs=args.epochs,
            task_loss_weight=args.task_loss_weight,
            logic_weight=args.logic_weight,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            early_stopping_min_epochs=args.early_stopping_min_epochs,
            device=args.device,
        ),
    )
    if args.model == "both":
        model_kinds = ("gru", "lstm")
    elif args.model == "all":
        model_kinds = ("gru", "lstm", "transformer")
    else:
        model_kinds = (args.model,)
    if args.output_dir:
        output_dir = args.output_dir
        model_dir = args.model_dir or output_dir / "models"
    else:
        run_paths = create_next_run(args.runs_dir or paths.runs_dir, args.dataset_name)
        output_dir = run_paths.root
        model_dir = args.model_dir or run_paths.models
        print(f"Run directory: {output_dir}")

    run = run_experiment(
        args.input or paths.dataset,
        output_dir,
        model_dir,
        config=config,
        model_kinds=model_kinds,
        logic_weights=args.logic_weights,
        max_validation_accuracy_drop=args.max_validation_accuracy_drop,
        max_cases=args.max_cases,
    )
    if args.logic_weights:
        print("\nValidation logic-weight search")
        print(run.validation_search.to_string(index=False))
    print("\nTest results")
    print(run.results.to_string(index=False))
