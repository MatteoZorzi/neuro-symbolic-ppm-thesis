"""Save and load GRU, LSTM and Transformer models with their metadata.

Both halves of the checkpoint format live here so that the serialised payload
and its reconstruction evolve together.
"""

from __future__ import annotations


from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from ..config import ExperimentConfig, ModelConfig
from ..data.prefixes import ActivityVocabulary
from .models import ModelKind, build_model


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_kind: ModelKind,
    vocabulary: ActivityVocabulary,
    config: ExperimentConfig,
    logic_enabled: bool,
    best_epoch: int,
    stopped_early: bool,
) -> None:
    """Save enough metadata to reconstruct the trained model."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_kind": model_kind,
            "model_config": asdict(config.model),
            "activities": list(vocabulary.activities),
            "logic_enabled": logic_enabled,
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
            "experiment_config": config.to_dict(),
        },
        path,
    )


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[nn.Module, ActivityVocabulary, dict[str, object]]:
    """Load a checkpoint created by :func:`save_checkpoint`."""

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    vocabulary = ActivityVocabulary(tuple(payload["activities"]))
    model_config = ModelConfig(**payload["model_config"])
    model = build_model(
        payload["model_kind"],
        len(vocabulary.tokens),
        len(vocabulary.activities),
        vocabulary.pad_id,
        model_config,
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, vocabulary, payload