# Central configuration for the next-activity experiment

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SPLIT_STRATEGIES = ("random", "temporal")
VOCABULARY_SCOPES = ("train", "all")
KNOWLEDGE_SOURCES = ("train", "test")
NOISE_MODELS = ("target", "event")


@dataclass(frozen=True)
class DataConfig:
    # Dataset split and batching parameters

    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 128
    num_workers: int = 0

    #: ``"random"`` shuffles cases with ``seed`` (the original behaviour, kept as
    #: the default so existing runs and resumed CSVs stay reproducible).
    #: ``"temporal"`` orders cases by the timestamp of their first event and
    #: cuts by position -- the protocol of Mezini et al. (2026).
    split_strategy: str = "random"

    #: ``"train"`` builds the activity alphabet from the training partition only
    #: (original behaviour; held-out cases with unseen activities must then be
    #: dropped or raise). ``"all"`` builds it over every partition, which is what
    #: lets a temporal split keep every case without filtering.
    vocabulary_scope: str = "train"

    #: Partition the background knowledge (DFA, precedence constraints, Petri
    #: net) is discovered from. ``"test"`` follows Mezini et al. (2026) Sec. 4.1.
    knowledge_source: str = "train"

    #: What the noise corrupts. ``"target"`` replaces only the label to be
    #: predicted, leaving the prefixes and therefore the compliance of the log
    #: intact. ``"event"`` replaces the label of an event inside the trace, so
    #: the damage propagates to the later prefixes, to the replay on the Petri
    #: net and to compliance -- the model of Mezini et al. (2026) Sec. 4.1.
    #: Two different regimes, not two intensities of the same thing.
    noise_model: str = "target"

    def __post_init__(self) -> None:
        held_out = self.validation_fraction + self.test_fraction
        if not 0.0 < held_out < 1.0:
            raise ValueError("Validation and test fractions must sum to a value in (0, 1).")
        if self.split_strategy not in SPLIT_STRATEGIES:
            raise ValueError(f"split_strategy must be one of {SPLIT_STRATEGIES}.")
        if self.vocabulary_scope not in VOCABULARY_SCOPES:
            raise ValueError(f"vocabulary_scope must be one of {VOCABULARY_SCOPES}.")
        if self.knowledge_source not in KNOWLEDGE_SOURCES:
            raise ValueError(f"knowledge_source must be one of {KNOWLEDGE_SOURCES}.")
        if self.noise_model not in NOISE_MODELS:
            raise ValueError(f"noise_model must be one of {NOISE_MODELS}.")

    @property
    def train_fraction(self) -> float:
        # Implied by the held-out fractions; 0.70 with the 0.15/0.15 defaults.
        return 1.0 - self.validation_fraction - self.test_fraction


@dataclass(frozen=True)
class ModelConfig:
    # Shared and architecture-specific model dimensions

    embedding_dim: int = 32
    hidden_dim: int = 64
    #: Stacked layers of the recurrent trunk, GRU or LSTM alike. One is enough
    #: for the compact baseline; Mezini et al. (2026) use two of 100 units, and
    #: that is the size to pass when their trunk is wanted.
    recurrent_layers: int = 1
    dropout: float = 0.20

    def __post_init__(self) -> None:
        if self.recurrent_layers < 1:
            raise ValueError("recurrent_layers must be at least 1.")


@dataclass(frozen=True)
class TrainingConfig:
    # Optimisation and logic-regularization parameters

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
    # Complete, serializable experiment configuration

    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

# Config for the temporal / knowledge-from-test protocol
def temporal_protocol(
    seed: int = 42,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    **overrides: Any,
) -> ExperimentConfig:

    return ExperimentConfig(
        seed=seed,
        data=DataConfig(
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            split_strategy="temporal",
            vocabulary_scope="all",
            knowledge_source="test",
            noise_model="event",
            **{k: v for k, v in overrides.items() if k in {"batch_size", "num_workers"}},
        ),
        **{k: v for k, v in overrides.items() if k in {"model", "training"}},
    )


#: Default dataset used when none is specified on the command line
DEFAULT_DATASET = "Sepsis_Case"


@dataclass(frozen=True)
class ProjectPaths:
    # Default project paths, resolved from the repository root

    root: Path
    dataset: Path
    analysis_dir: Path
    runs_dir: Path

    @classmethod
    def from_root(cls, root: Path, dataset_name: str = DEFAULT_DATASET, *, dataset_file: str | None = None) -> "ProjectPaths":
        root = root.resolve()
        dataset_dir = root / "datasets" / dataset_name
        if dataset_file is not None:
            dataset = dataset_dir / dataset_file
        else:
            dataset = cls._discover_log(dataset_dir)
        return cls(
            root=root,
            dataset=dataset,
            analysis_dir=dataset_dir / "analysis",
            runs_dir=root / "runs",
        )

    @staticmethod
    def _discover_log(dataset_dir: Path) -> Path:
        # Return the lone ``*.xes``/``*.csv`` log in ``dataset_dir`` (conventional path otherwise)

        candidates = sorted(dataset_dir.glob("*.xes")) or sorted(dataset_dir.glob("*.csv"))
        if not candidates:
            # No log present (e.g. resolving defaults before data is in place):
            # fall back to a conventional name so error messages point at the dir.
            return dataset_dir / f"{dataset_dir.name}.xes"
        return candidates[0]
