"""Deterministic training for baseline and logic-aware recurrent models."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
import random
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# Required by deterministic CUDA matrix multiplications on CUDA 10.2+.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..data.preparation import ActivityVocabulary, PrefixBatch
from ..process.petrinet import AdjacencyMatrix
from .evaluation import EvaluationResult, evaluate_model
from .logic import forbidden_probability_mass
from .models import ModelKind, build_model, save_checkpoint


def set_random_seed(seed: int) -> None:
    """Seed random generators and request deterministic PyTorch kernels."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


@dataclass
class TrainingResult:
    model: nn.Module
    history: pd.DataFrame
    validation: EvaluationResult
    checkpoint_path: Path
    best_epoch: int
    stopped_early: bool


def train_model(
    model_kind: ModelKind,
    vocabulary: ActivityVocabulary,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    allowed_mask: torch.Tensor,
    config: ExperimentConfig,
    checkpoint_path: Path,
    use_logic: bool,
    logic_mode: str = "checker",
    embedding_logic=None,
    epoch_callback: Callable[[dict[str, float]], None] | None = None,
    marking_dim: int = 0,
    adjacency: tuple[AdjacencyMatrix, AdjacencyMatrix] | None = None,
) -> TrainingResult:
    """Train one model and restore the epoch with best validation loss.

    ``logic_mode`` selects which differentiable penalty the ``logic_weight``
    multiplies when ``use_logic`` is set: ``"checker"`` uses the empirical-DFA
    forbidden-probability mass, ``"embedder"`` uses the learned T-LEAF embedding
    distance supplied via ``embedding_logic``. Forbidden mass is always recorded
    so process conformance stays comparable across modes.
    """

    if logic_mode not in {"checker", "embedder"}:
        raise ValueError("logic_mode must be 'checker' or 'embedder'.")
    if use_logic and logic_mode == "embedder" and embedding_logic is None:
        raise ValueError("logic_mode='embedder' requires an embedding_logic loss.")

    set_random_seed(config.seed)
    device = resolve_device(config.training.device)
    model = build_model(
        model_kind,
        vocabulary_size=len(vocabulary.tokens),
        number_of_classes=len(vocabulary.activities),
        pad_id=vocabulary.pad_id,
        config=config.model,
        marking_dim=marking_dim,
        adjacency=adjacency,
    ).to(device)
    allowed_mask = allowed_mask.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        running_loss = 0.0
        running_ce = 0.0
        running_logic = 0.0
        running_forbidden = 0.0
        example_count = 0

        for raw_batch in train_loader:
            batch: PrefixBatch = raw_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.tokens, batch.lengths, batch.markings)
            cross_entropy = criterion(logits, batch.targets)
            # Forbidden mass is tracked in every mode as a conformance measure.
            forbidden_mass = forbidden_probability_mass(
                logits, batch.tokens, batch.lengths, allowed_mask
            )
            if use_logic and logic_mode == "embedder":
                logic_penalty = embedding_logic(logits, batch.tokens, batch.lengths)
            else:
                logic_penalty = forbidden_mass
            loss = config.training.task_loss_weight * cross_entropy
            if use_logic:
                loss = loss + config.training.logic_weight * logic_penalty
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), config.training.gradient_clip
            )
            optimizer.step()

            batch_size = batch.targets.size(0)
            example_count += batch_size
            running_loss += loss.item() * batch_size
            running_ce += cross_entropy.item() * batch_size
            running_logic += float(logic_penalty) * batch_size
            running_forbidden += forbidden_mass.item() * batch_size

        validation = evaluate_model(
            model,
            validation_loader,
            allowed_mask,
            device,
            top_k=config.training.top_k,
        )
        row = {
            "epoch": float(epoch),
            "train_loss": running_loss / example_count,
            "train_cross_entropy": running_ce / example_count,
            "train_logic_penalty": running_logic / example_count,
            "train_forbidden_mass": running_forbidden / example_count,
            "logic_mode": logic_mode if use_logic else "none",
            "task_loss_weight": config.training.task_loss_weight,
            "logic_loss_weight": (
                config.training.logic_weight if use_logic else 0.0
            ),
            "validation_loss": validation.loss,
            "validation_accuracy": validation.accuracy,
            "validation_forbidden_mass": validation.forbidden_mass,
        }
        history.append(row)
        if epoch_callback:
            epoch_callback(row)
        improved = (
            validation.loss
            < best_validation_loss - config.training.early_stopping_min_delta
        )
        if improved:
            best_validation_loss = validation.loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epoch >= config.training.early_stopping_min_epochs
            and epochs_without_improvement
            >= config.training.early_stopping_patience
        ):
            stopped_early = True
            break

    if best_state is None:
        raise RuntimeError("Training completed without producing a model state.")
    model.load_state_dict(best_state)
    validation = evaluate_model(
        model,
        validation_loader,
        allowed_mask,
        device,
        top_k=config.training.top_k,
    )
    save_checkpoint(
        checkpoint_path,
        model,
        model_kind,
        vocabulary,
        config,
        logic_enabled=use_logic,
        best_epoch=best_epoch,
        stopped_early=stopped_early,
    )
    return TrainingResult(
        model=model,
        history=pd.DataFrame(history),
        validation=validation,
        checkpoint_path=checkpoint_path,
        best_epoch=best_epoch,
        stopped_early=stopped_early,
    )
