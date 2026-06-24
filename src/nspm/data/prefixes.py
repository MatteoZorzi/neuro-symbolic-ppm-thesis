"""Trace preparation and prefix-based PyTorch datasets."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from ..process.automaton import START
from .xes import ACTIVITY, CASE_ID, INDEX


PAD = "<PAD>"


@dataclass(frozen=True)
class ActivityVocabulary:
    """Separate token IDs used as inputs from class IDs used as targets."""

    activities: tuple[str, ...]

    @classmethod
    def from_traces(cls, traces: Iterable[Sequence[str]]) -> "ActivityVocabulary":
        activities = sorted({activity for trace in traces for activity in trace})
        if not activities:
            raise ValueError("Cannot build a vocabulary without activities.")
        return cls(tuple(activities))

    @property
    def tokens(self) -> tuple[str, ...]:
        return (PAD, START, *self.activities)

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    @property
    def class_to_id(self) -> dict[str, int]:
        return {activity: index for index, activity in enumerate(self.activities)}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    def encode_prefix(self, prefix: Sequence[str]) -> tuple[int, ...]:
        lookup = self.token_to_id
        return tuple(lookup[action] for action in prefix)


@dataclass(frozen=True)
class TraceSplits:
    train: Mapping[str, tuple[str, ...]]
    validation: Mapping[str, tuple[str, ...]]
    test: Mapping[str, tuple[str, ...]]


def extract_traces(events: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """Convert ordered event rows into immutable case traces."""

    ordered = events.sort_values([CASE_ID, INDEX])
    return {
        str(case_id): tuple(group[ACTIVITY].astype(str))
        for case_id, group in ordered.groupby(CASE_ID, sort=False)
    }


def split_traces(
    traces: Mapping[str, Sequence[str]],
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> TraceSplits:
    """Split by case ID so prefixes from one patient never cross partitions."""

    case_ids = np.array(sorted(traces))
    if len(case_ids) < 3:
        raise ValueError("At least three cases are required for train/validation/test splits.")

    held_out_fraction = validation_fraction + test_fraction
    train_ids, held_out_ids = train_test_split(
        case_ids, test_size=held_out_fraction, random_state=seed, shuffle=True
    )
    relative_test_fraction = test_fraction / held_out_fraction
    validation_ids, test_ids = train_test_split(
        held_out_ids,
        test_size=relative_test_fraction,
        random_state=seed,
        shuffle=True,
    )

    def select(ids: np.ndarray) -> dict[str, tuple[str, ...]]:
        return {case_id: tuple(traces[case_id]) for case_id in ids}

    return TraceSplits(select(train_ids), select(validation_ids), select(test_ids))


@dataclass(frozen=True)
class PrefixExample:
    case_id: str
    token_ids: tuple[int, ...]
    target_id: int


def make_prefix_examples(
    traces: Mapping[str, Sequence[str]],
    vocabulary: ActivityVocabulary,
) -> list[PrefixExample]:
    """Create one supervised example for every observed next action."""

    examples: list[PrefixExample] = []
    class_to_id = vocabulary.class_to_id
    for case_id, trace in traces.items():
        history = [START]
        for target in trace:
            examples.append(
                PrefixExample(
                    case_id=case_id,
                    token_ids=vocabulary.encode_prefix(history),
                    target_id=class_to_id[target],
                )
            )
            history.append(target)
    return examples


class PrefixDataset(Dataset):
    """Thin Dataset wrapper around pre-encoded variable-length prefixes."""

    def __init__(self, examples: Sequence[PrefixExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PrefixExample:
        return self.examples[index]


@dataclass
class PrefixBatch:
    case_ids: list[str]
    tokens: torch.Tensor
    lengths: torch.Tensor
    targets: torch.Tensor

    def to(self, device: torch.device) -> "PrefixBatch":
        return PrefixBatch(
            case_ids=self.case_ids,
            tokens=self.tokens.to(device),
            lengths=self.lengths.to(device),
            targets=self.targets.to(device),
        )


def collate_prefixes(
    examples: Sequence[PrefixExample], pad_id: int
) -> PrefixBatch:
    sequences = [torch.tensor(example.token_ids, dtype=torch.long) for example in examples]
    return PrefixBatch(
        case_ids=[example.case_id for example in examples],
        tokens=pad_sequence(sequences, batch_first=True, padding_value=pad_id),
        lengths=torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long),
        targets=torch.tensor([example.target_id for example in examples], dtype=torch.long),
    )


def make_data_loader(
    examples: Sequence[PrefixExample],
    vocabulary: ActivityVocabulary,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """Build a loader with the vocabulary-specific padding value."""

    return DataLoader(
        PrefixDataset(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(collate_prefixes, pad_id=vocabulary.pad_id),
    )
