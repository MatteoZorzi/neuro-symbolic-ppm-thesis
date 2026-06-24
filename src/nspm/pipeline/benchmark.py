"""Grid-search benchmark over models, logic variants and training-label noise.

The empirical DFA and the activity vocabulary are built **once** from the clean
training traces and shared across every noise level: only the training *labels*
are perturbed. Each cell of the grid trains the models it needs and evaluates
the requested variants on the (clean) test set, returning one tidy row per
evaluated model. Validation/test loaders are clean throughout.

Variants:

* ``baseline``       -- plain next-activity model (logic-free).
* ``baseline_mask``  -- the same baseline weights, with DFA-forbidden classes
  removed from the logits at inference (hard conformance).
* ``checker``        -- trained with the differentiable forbidden-mass logic
  loss derived from the clean DFA.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random
import tempfile
import time
from typing import Iterable, Sequence

import pandas as pd

from ..config import ExperimentConfig
from ..data.corruption import corrupt_targets
from ..data.prefixes import (
    ActivityVocabulary,
    extract_traces,
    make_data_loader,
    make_prefix_examples,
    split_traces,
)
from ..data.xes import read_xes
from ..learning.evaluation import evaluate_model
from ..learning.logic import build_allowed_mask
from ..learning.models import ModelKind
from ..learning.training import resolve_device, train_model
from ..process.automaton import ProcessDFA

ALL_VARIANTS = ("baseline", "baseline_mask", "checker")


def run_corruption_grid(
    input_path: str | Path,
    *,
    dataset_name: str | None = None,
    model_kinds: Sequence[ModelKind] = ("gru", "lstm"),
    noise_levels: Iterable[float] = (0.0, 0.1, 0.2, 0.3),
    variants: Sequence[str] = ALL_VARIANTS,
    config: ExperimentConfig | None = None,
    epochs: int | None = None,
    device: str | None = None,
    max_cases: int | None = None,
    seed: int = 42,
    drop_unseen: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the model x variant x noise grid; return one tidy row per evaluation."""

    unknown = set(variants) - set(ALL_VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    input_path = Path(input_path)
    dataset_name = dataset_name or input_path.parent.name
    noise_levels = sorted(set(float(level) for level in noise_levels))

    config = config or ExperimentConfig(seed=seed)
    training_overrides = {}
    if epochs is not None:
        training_overrides["epochs"] = epochs
    if device is not None:
        training_overrides["device"] = device
    if training_overrides:
        config = replace(config, training=replace(config.training, **training_overrides))
    resolved_device = resolve_device(config.training.device)

    # --- Clean preprocessing: split, vocabulary and DFA come from clean data ---
    events = read_xes(input_path, max_cases=max_cases)
    traces = extract_traces(events)
    splits = split_traces(
        traces,
        validation_fraction=config.data.validation_fraction,
        test_fraction=config.data.test_fraction,
        seed=config.seed,
    )
    vocabulary = ActivityVocabulary.from_traces(splits.train.values())
    known = set(vocabulary.class_to_id)

    def filter_partition(partition):
        kept = {
            case_id: trace
            for case_id, trace in partition.items()
            if all(activity in known for activity in trace)
        }
        return kept, len(partition) - len(kept)

    unseen = {
        activity
        for partition in (splits.validation, splits.test)
        for trace in partition.values()
        for activity in trace
        if activity not in known
    }
    if unseen:
        if not drop_unseen:
            raise ValueError(
                f"Held-out cases contain unseen activities: {sorted(unseen)}"
            )
        validation, dropped_val = filter_partition(splits.validation)
        test, dropped_test = filter_partition(splits.test)
        splits = replace(splits, validation=validation, test=test)
        if verbose:
            print(
                f"[{dataset_name}] dropped {dropped_val} validation / {dropped_test} "
                f"test cases with unseen activities ({len(unseen)} activities)"
            )

    automaton = ProcessDFA.from_traces(splits.train.values())
    allowed_mask = build_allowed_mask(automaton, vocabulary)
    eval_mask = allowed_mask.to(resolved_device)

    clean_train = make_prefix_examples(splits.train, vocabulary)
    validation_examples = make_prefix_examples(splits.validation, vocabulary)
    test_examples = make_prefix_examples(splits.test, vocabulary)
    validation_loader = make_data_loader(
        validation_examples, vocabulary, config.data.batch_size, shuffle=False
    )
    test_loader = make_data_loader(
        test_examples, vocabulary, config.data.batch_size, shuffle=False
    )

    need_baseline = bool({"baseline", "baseline_mask"} & set(variants))
    need_checker = "checker" in variants

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_dir = Path(tmp)
        for model_kind in model_kinds:
            for noise in noise_levels:
                rng = random.Random(config.seed + int(noise * 1000))
                train_examples, corrupted = corrupt_targets(
                    clean_train, noise, vocabulary, rng
                )
                train_loader = make_data_loader(
                    train_examples, vocabulary, config.data.batch_size, shuffle=True
                )
                if verbose:
                    print(
                        f"[{dataset_name}] {model_kind} noise={noise:.2f} "
                        f"(corrupted {corrupted}/{len(train_examples)} targets)"
                    )

                trained: dict[str, object] = {}
                if need_baseline:
                    started = time.perf_counter()
                    trained["baseline"] = train_model(
                        model_kind=model_kind,
                        vocabulary=vocabulary,
                        train_loader=train_loader,
                        validation_loader=validation_loader,
                        allowed_mask=allowed_mask,
                        config=config,
                        checkpoint_path=checkpoint_dir / "baseline.pt",
                        use_logic=False,
                    )
                    trained["baseline_runtime"] = time.perf_counter() - started
                if need_checker:
                    started = time.perf_counter()
                    trained["checker"] = train_model(
                        model_kind=model_kind,
                        vocabulary=vocabulary,
                        train_loader=train_loader,
                        validation_loader=validation_loader,
                        allowed_mask=allowed_mask,
                        config=config,
                        checkpoint_path=checkpoint_dir / "checker.pt",
                        use_logic=True,
                        logic_mode="checker",
                    )
                    trained["checker_runtime"] = time.perf_counter() - started

                for variant in variants:
                    if variant == "checker":
                        result = trained["checker"]
                        enforce_mask = False
                    else:
                        result = trained["baseline"]
                        enforce_mask = variant == "baseline_mask"
                    evaluation = evaluate_model(
                        result.model,
                        test_loader,
                        eval_mask,
                        resolved_device,
                        top_k=config.training.top_k,
                        enforce_mask=enforce_mask,
                    )
                    source = "checker" if variant == "checker" else "baseline"
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "architecture": model_kind,
                            "variant": variant,
                            "noise_level": noise,
                            "seed": config.seed,
                            "n_train_examples": len(train_examples),
                            "n_train_corrupted": corrupted,
                            "dfa_states": len(automaton.states),
                            "dfa_transitions": automaton.transition_count,
                            "best_epoch": result.best_epoch,
                            "stopped_early": result.stopped_early,
                            "train_runtime_s": round(trained[f"{source}_runtime"], 2),
                            **evaluation.metrics(),
                        }
                    )

    return pd.DataFrame(rows)
