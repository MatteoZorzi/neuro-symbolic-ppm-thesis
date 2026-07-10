"""Grid-search benchmark over models, logic variants, training-label noise and
training-set size.

The symbolic layer is built **once** from the full clean training set and held
fixed across the whole grid: the activity vocabulary, the empirical
directly-follows DFA, the mined LTLf precedence constraints and (optionally) the
trained T-LEAF embedder. Only the *neural* training data is varied:

* ``train_fractions`` keeps a nested subset of training **cases**
  (1 ⊃ 1/2 ⊃ 1/4 ⊃ ...), a learning curve over data scarcity;
* ``noise_levels`` then corrupts that fraction of the (subset's) training
  **targets** with a random DFA activity.

Validation and test loaders stay clean and fixed, so every cell is comparable.
Each cell trains the models its variants need and evaluates on the clean test
set, returning one tidy row per evaluated model.

Variants:

* ``baseline``       -- plain next-activity model (logic-free).
* ``baseline_mask``  -- the same baseline weights, with DFA-forbidden classes
  removed from the logits at inference (hard conformance).
* ``checker``        -- trained with the differentiable forbidden-mass logic
  loss derived from the clean DFA.
* ``embedder``       -- trained with the T-LEAF embedding logic loss
  ``‖q(A) − q(w_pred)‖²`` from the clean constraints (needs mined constraints).
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
from ..data.preparation import (
    ActivityVocabulary,
    PrefixLog,
    TraceSplits,
    TraceUtils,
)
from ..data.loader import read_log
from ..learning.embedder import HierarchicalEmbedder
from ..learning.embedder_training import EmbedderConfig, train_embedder
from ..learning.evaluation import evaluate_model
from ..learning.logic import (
    EmbeddingLogicLoss,
    build_allowed_mask,
    precedence_violation_rate,
)
from ..learning.models import ModelKind
from ..learning.training import resolve_device, train_model
from ..process.automaton import ProcessDFA
from ..process.graph_encoding import FeatureSpace
from ..process.ltl_constraints import mine_precedence_constraints

ALL_VARIANTS = ("baseline", "baseline_mask", "checker", "embedder")


def _nested_case_order(case_ids: Sequence[str], seed: int) -> list[str]:
    """A single fixed shuffle, so smaller fractions are nested in larger ones."""

    ordered = sorted(case_ids)
    random.Random(seed).shuffle(ordered)
    return ordered


def run_corruption_grid(
    input_path: str | Path,
    *,
    dataset_name: str | None = None,
    model_kinds: Sequence[ModelKind] = ("gru", "lstm"),
    noise_levels: Iterable[float] = (0.0, 0.1, 0.25, 0.4),
    train_fractions: Iterable[float] = (1.0,),
    variants: Sequence[str] = ALL_VARIANTS,
    config: ExperimentConfig | None = None,
    epochs: int | None = None,
    embedder_epochs: int = 5,
    logic_weight: float = 0.5,
    device: str | None = None,
    max_cases: int | None = None,
    seed: int = 42,
    drop_unseen: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the model x variant x train_fraction x noise grid; one tidy row per eval."""

    unknown = set(variants) - set(ALL_VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    input_path = Path(input_path)
    dataset_name = dataset_name or input_path.parent.name
    noise_levels = sorted(set(float(level) for level in noise_levels))
    train_fractions = sorted(set(float(f) for f in train_fractions), reverse=True)
    if any(not 0.0 < f <= 1.0 for f in train_fractions):
        raise ValueError("train_fractions must be in (0, 1].")

    config = config or ExperimentConfig(seed=seed)
    training_overrides = {}
    if epochs is not None:
        training_overrides["epochs"] = epochs
    if device is not None:
        training_overrides["device"] = device
    if training_overrides:
        config = replace(config, training=replace(config.training, **training_overrides))
    logic_config = replace(config, training=replace(config.training, logic_weight=logic_weight))
    resolved_device = resolve_device(config.training.device)

    # --- Clean preprocessing: split, vocabulary, DFA and constraints from the
    #     full clean training set; held fixed across the entire grid. ---
    events = read_log(input_path, max_cases=max_cases)
    traces = TraceUtils.extract_traces(events)
    splits = TraceSplits.from_traces(
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
    constraints = mine_precedence_constraints(
        splits.train.values(),
        min_support=max(3, len(splits.train) // 10),
        min_confidence=0.95,
        max_constraints=15,
    )

    validation_loader = PrefixLog.from_traces(splits.validation, vocabulary).data_loader(
        config.data.batch_size, shuffle=False,
    )
    test_loader = PrefixLog.from_traces(splits.test, vocabulary).data_loader(
        config.data.batch_size, shuffle=False,
    )

    need_baseline = bool({"baseline", "baseline_mask"} & set(variants))
    need_checker = "checker" in variants
    need_embedder = "embedder" in variants

    # --- Train the T-LEAF embedder once on the clean constraints (reused). ---
    embedding_logic = None
    if need_embedder:
        if not constraints:
            if verbose:
                print(f"[{dataset_name}] no constraints mined -> skipping embedder variant")
            variants = [v for v in variants if v != "embedder"]
            need_embedder = False
        else:
            space = FeatureSpace.create(
                vocabulary.activities, prop_dim=50, node_dim=100,
                seed=config.seed, device=resolved_device,
            )
            embedder = HierarchicalEmbedder(
                aggregation="random_walk", normalize=False, meta_layers=2
            ).to(resolved_device)
            train_embedder(
                embedder, space, constraints, vocabulary.activities,
                EmbedderConfig(
                    epochs=embedder_epochs, triplets_per_constraint=12,
                    margin=5.0, learning_rate=2e-3, seed=config.seed,
                ),
                device=resolved_device, verbose=verbose,
            )
            embedding_logic = EmbeddingLogicLoss(
                embedder, space, constraints, vocabulary,
                max_examples=10, max_constraints=3,
            )

    train_order = _nested_case_order(list(splits.train), config.seed)
    n_train_cases = len(train_order)

    def pvr(model) -> float:
        if not constraints:
            return float("nan")
        return precedence_violation_rate(
            model, test_loader, constraints, vocabulary, resolved_device
        )

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_dir = Path(tmp)
        for model_kind in model_kinds:
            for fraction in train_fractions:
                keep = max(1, round(fraction * n_train_cases))
                frac_ids = train_order[:keep]
                frac_train = {cid: splits.train[cid] for cid in frac_ids}
                clean_log = PrefixLog.from_traces(frac_train, vocabulary)

                for noise in noise_levels:
                    rng = random.Random(config.seed + int(noise * 1000))
                    train_log, corrupted = clean_log.corrupt_targets(noise, rng)
                    train_loader = train_log.data_loader(
                        config.data.batch_size, shuffle=True
                    )
                    if verbose:
                        print(
                            f"[{dataset_name}] {model_kind} frac={fraction:g} "
                            f"noise={noise:.2f} ({len(frac_train)} cases, "
                            f"{corrupted}/{len(train_log)} corrupted)"
                        )

                    trained: dict[str, object] = {}
                    if need_baseline:
                        started = time.perf_counter()
                        trained["baseline"] = train_model(
                            model_kind=model_kind, vocabulary=vocabulary,
                            train_loader=train_loader, validation_loader=validation_loader,
                            allowed_mask=allowed_mask, config=config,
                            checkpoint_path=checkpoint_dir / "baseline.pt",
                            use_logic=False,
                        )
                        trained["baseline_runtime"] = time.perf_counter() - started
                        trained["baseline_pvr"] = pvr(trained["baseline"].model)
                    if need_checker:
                        started = time.perf_counter()
                        trained["checker"] = train_model(
                            model_kind=model_kind, vocabulary=vocabulary,
                            train_loader=train_loader, validation_loader=validation_loader,
                            allowed_mask=allowed_mask, config=logic_config,
                            checkpoint_path=checkpoint_dir / "checker.pt",
                            use_logic=True, logic_mode="checker",
                        )
                        trained["checker_runtime"] = time.perf_counter() - started
                        trained["checker_pvr"] = pvr(trained["checker"].model)
                    if need_embedder:
                        started = time.perf_counter()
                        trained["embedder"] = train_model(
                            model_kind=model_kind, vocabulary=vocabulary,
                            train_loader=train_loader, validation_loader=validation_loader,
                            allowed_mask=allowed_mask, config=logic_config,
                            checkpoint_path=checkpoint_dir / "embedder.pt",
                            use_logic=True, logic_mode="embedder",
                            embedding_logic=embedding_logic,
                        )
                        trained["embedder_runtime"] = time.perf_counter() - started
                        trained["embedder_pvr"] = pvr(trained["embedder"].model)

                    for variant in variants:
                        if variant == "checker":
                            source, enforce_mask = "checker", False
                        elif variant == "embedder":
                            source, enforce_mask = "embedder", False
                        else:
                            source = "baseline"
                            enforce_mask = variant == "baseline_mask"
                        result = trained[source]
                        evaluation = evaluate_model(
                            result.model, test_loader, eval_mask, resolved_device,
                            top_k=config.training.top_k, enforce_mask=enforce_mask,
                        )
                        rows.append(
                            {
                                "dataset": dataset_name,
                                "architecture": model_kind,
                                "variant": variant,
                                "train_fraction": fraction,
                                "noise_level": noise,
                                "seed": config.seed,
                                "n_train_cases": len(frac_train),
                                "n_train_examples": len(train_log),
                                "n_train_corrupted": corrupted,
                                "n_constraints": len(constraints),
                                "dfa_states": len(automaton.states),
                                "dfa_transitions": automaton.transition_count,
                                "best_epoch": result.best_epoch,
                                "stopped_early": result.stopped_early,
                                "train_runtime_s": round(trained[f"{source}_runtime"], 2),
                                "precedence_violation_rate": trained[f"{source}_pvr"],
                                **evaluation.metrics(),
                            }
                        )

    return pd.DataFrame(rows)
