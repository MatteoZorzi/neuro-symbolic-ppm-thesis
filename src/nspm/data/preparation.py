# Data preparation: traces and splits, vocabulary, prefix log, label corruption

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from functools import partial
from typing import Iterable, Iterator, Mapping, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from ..process.automaton import START
from .loader import ACTIVITY, CASE_ID, INDEX
if TYPE_CHECKING:
    from ..process.petrinet import PetriNet

PAD = "<PAD>"

# ---------------------------------------------------------------- traces & splits
@dataclass(frozen=True)
class TraceSplits:
    # Case-level train/validation/test partitions.

    train: Mapping[str, tuple[str, ...]]
    validation: Mapping[str, tuple[str, ...]]
    test: Mapping[str, tuple[str, ...]]

    @classmethod
    def from_traces(cls, traces: Mapping[str, Sequence[str]], validation_fraction: float = 0.15, test_fraction: float = 0.15, seed: int = 42) -> "TraceSplits":
        # Split by case ID so prefixes from one case never cross partitions

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

        return cls(select(train_ids), select(validation_ids), select(test_ids))


@dataclass(frozen=True)
class TraceUtils:
    # Trace Utilities functions

    @classmethod
    def extract_traces(cls, events: pd.DataFrame) -> dict[str, tuple[str, ...]]:
        # Convert ordered event rows into immutable case traces

        ordered = events.sort_values([CASE_ID, INDEX])
        return {str(case_id): tuple(group[ACTIVITY].astype(str)) for case_id, group in ordered.groupby(CASE_ID, sort=False)}

    @classmethod
    def subsample_traces(cls, traces: dict[str, tuple[str, ...]], fraction: float, seed: int = 42) -> dict[str, tuple[str, ...]]:
        # Keep a random "fraction" of cases (for data-ablation study)

        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1].")
        if fraction == 1.0:
            return dict(traces)
        case_ids = sorted(traces)
        keep = max(1, round(len(case_ids) * fraction))
        chosen = random.Random(seed).sample(case_ids, keep)
        return {case_id: traces[case_id] for case_id in chosen}

# --------------------------------------------------------------------- vocabulary

@dataclass(frozen=True)
class ActivityVocabulary:
    # Separate token IDs used as inputs from class IDs used as targets.

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


# --------------------------------------------------------------------- prefix log

@dataclass(frozen=True)
class PrefixExample:

    case_id: str
    token_ids: tuple[int, ...]
    target_id: int
    marking: tuple[int, ...] | None = None
    marking_sequence: tuple[tuple[int, ...], ...] | None = None

class PrefixDataset(Dataset):
    # Thin Dataset wrapper around pre-encoded variable-length prefixes

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
    markings: torch.Tensor | None
    marking_sequences: torch.Tensor | None

    def to(self, device: torch.device) -> "PrefixBatch":
        return PrefixBatch(
            case_ids=self.case_ids,
            tokens=self.tokens.to(device),
            lengths=self.lengths.to(device),
            targets=self.targets.to(device),
            markings=self.markings.to(device) if self.markings is not None else None,
            marking_sequences=self.marking_sequences.to(device) if self.marking_sequences is not None else None,
        )

def collate_prefixes(examples: Sequence[PrefixExample], pad_id: int) -> PrefixBatch:

    sequences = [torch.tensor(example.token_ids, dtype=torch.long) for example in examples]
    return PrefixBatch(
        case_ids=[example.case_id for example in examples],
        tokens=pad_sequence(sequences, batch_first=True, padding_value=pad_id),
        lengths=torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long),
        targets=torch.tensor([example.target_id for example in examples], dtype=torch.long),
        markings=None if examples[0].marking is None else torch.stack([torch.tensor(example.marking, dtype=torch.float32) for example in examples]),
        marking_sequences=None if examples[0].marking_sequence is None else pad_sequence([torch.tensor(example.marking_sequence, dtype=torch.float32) for example in examples], batch_first=True, padding_value=0.0),
    )


@dataclass(frozen=True)
class PrefixLog:
    # Encoded supervised examples for one trace partition, plus their vocabulary.

    examples: tuple[PrefixExample, ...]
    vocabulary: ActivityVocabulary

    @classmethod
    def from_traces(cls, traces: Mapping[str, Sequence[str]], vocabulary: ActivityVocabulary) -> "PrefixLog":
        # Create one supervised example for every observed next action

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
        return cls(tuple(examples), vocabulary)

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[PrefixExample]:
        return iter(self.examples)

    def data_loader(self, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
        # Build a loader with the vocabulary-specific padding value

        return DataLoader(
            PrefixDataset(self.examples),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=partial(collate_prefixes,
            pad_id=self.vocabulary.pad_id),
        )

    def with_markings(self, petrinet: PetriNet) -> PrefixLog:

        vocab_tokes = self.vocabulary.tokens
        markings_log: list[PrefixExample] = []
        for example in self.examples:
            prefix = [vocab_tokes[token_id] for token_id in example.token_ids[1:]]
            markings_log.append(replace(example, marking=petrinet.prefix_marking(prefix)))

        return PrefixLog(tuple(markings_log), self.vocabulary)

    def with_marking_sequences(self, petrinet: PetriNet) -> PrefixLog:
        # One replay chain per case: the longest prefix's marking sequence
        # contains every shorter prefix's sequence as a slice.

        vocab_tokens = self.vocabulary.tokens
        initial = petrinet.prefix_marking(())

        # from_traces emits each case's examples from shortest to longest
        # prefix, so the last write per case_id keeps the longest one
        longest: dict[str, PrefixExample] = {}
        for example in self.examples:
            longest[example.case_id] = example

        sequences: dict[str, tuple[tuple[int, ...], ...]] = {}
        for case_id, example in longest.items():
            prefix = [vocab_tokens[token_id] for token_id in example.token_ids[1:]]
            sequences[case_id] = petrinet.marking_sequence(prefix)

        sequences_log: list[PrefixExample] = []
        for example in self.examples:
            events = len(example.token_ids) - 1  # START is not an event
            sequence = (initial,) + sequences[example.case_id][:events]
            sequences_log.append(replace(example, marking_sequence=sequence))

        return PrefixLog(tuple(sequences_log), self.vocabulary)

    def corrupt_targets(self, fraction: float, rng: random.Random) -> tuple["PrefixLog", int]:
        # Replace the target of "fraction" of examples with a random activity

        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be in [0, 1].")
        number_of_classes = len(self.vocabulary.activities)
        if fraction == 0.0 or number_of_classes < 2:
            return PrefixLog(self.examples, self.vocabulary), 0

        count = round(len(self.examples) * fraction)
        corrupt_indices = set(rng.sample(range(len(self.examples)), count))

        perturbed: list[PrefixExample] = []
        changed = 0
        for index, example in enumerate(self.examples):
            if index in corrupt_indices:
                new_target = rng.randrange(number_of_classes)
                if new_target == example.target_id:  # guarantee an actual change
                    new_target = (new_target + 1) % number_of_classes
                perturbed.append(replace(example, target_id=new_target))
                changed += 1
            else:
                perturbed.append(example)
        return PrefixLog(tuple(perturbed), self.vocabulary), changed
