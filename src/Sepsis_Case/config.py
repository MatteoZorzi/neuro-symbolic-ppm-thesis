"""Central configuration for the Sepsis next-activity experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    """Dataset split and batching parameters."""

    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 128
    num_workers: int = 0

    def __post_init__(self) -> None:
        held_out = self.validation_fraction + self.test_fraction
        if not 0.0 < held_out < 1.0:
            raise ValueError("Validation and test fractions must sum to a value in (0, 1).")


@dataclass(frozen=True)
class ModelConfig:
    """Shared and architecture-specific model dimensions."""

    embedding_dim: int = 32
    hidden_dim: int = 64
    dropout: float = 0.20
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_feedforward_dim: int = 128
    transformer_max_length: int = 512

    def __post_init__(self) -> None:
        if self.embedding_dim % self.transformer_heads != 0:
            raise ValueError("embedding_dim must be divisible by transformer_heads.")


@dataclass(frozen=True)
class TrainingConfig:
    """Optimisation and logic-regularisation parameters."""

    epochs: int = 12
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    task_loss_weight: float = 1.0
    logic_weight: float = 0.5
    gradient_clip: float = 1.0
    top_k: int = 3
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    early_stopping_min_epochs: int = 5
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.task_loss_weight < 0 or self.logic_weight < 0:
            raise ValueError("Loss weights cannot be negative.")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1.")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta cannot be negative.")
        if self.early_stopping_min_epochs < 1:
            raise ValueError("early_stopping_min_epochs must be at least 1.")


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete, serialisable experiment configuration."""

    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectPaths:
    """Default project paths, resolved from the repository root."""

    root: Path
    dataset: Path
    analysis_dir: Path
    runs_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        root = root.resolve()
        dataset_dir = root / "datasets" / "Sepsis_Case"
        return cls(
            root=root,
            dataset=dataset_dir / "Sepsis_Cases_Event_Log.xes",
            analysis_dir=dataset_dir / "analysis",
            runs_dir=root / "runs",
        )
