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
from .loader import ACTIVITY, CASE_ID, INDEX, TIMESTAMP
if TYPE_CHECKING:
    from ..config import ExperimentConfig
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

    @classmethod
    def temporal(
        cls,
        traces: Mapping[str, Sequence[str]],
        case_order: Sequence[str],
        validation_fraction: float = 0.15,
        test_fraction: float = 0.15,
    ) -> "TraceSplits":
        # Split by position along a temporal case ordering: the oldest cases
        # train, the newest ones test, with no shuffling.
        #
        # This mirrors ``nirdizati_light.log.common.split_train_val_test`` with
        # ``shuffle=False`` -- that function also cuts the case list by index
        # position -- and matches the protocol of Mezini et al. (2026), who
        # "order the traces by the timestamp of the first event" before
        # splitting. The temporal semantics live entirely in ``case_order``;
        # build it with :meth:`TraceUtils.case_order_by_time`.
        #
        # Unlike :meth:`from_traces` this keeps every case: no shuffling, no
        # seed, no dropping. Later partitions may therefore contain activities
        # absent from ``train`` -- build the vocabulary over all partitions
        # (see :meth:`ActivityVocabulary.from_traces`) rather than filtering.

        held_out_fraction = validation_fraction + test_fraction
        if not 0.0 < held_out_fraction < 1.0:
            raise ValueError("Validation and test fractions must sum to a value in (0, 1).")

        ordered = [case_id for case_id in case_order if case_id in traces]
        missing = set(traces) - set(ordered)
        if missing:
            raise ValueError(
                f"case_order is missing {len(missing)} case(s) present in traces, "
                f"e.g. {sorted(missing)[:3]}"
            )
        if len(ordered) < 3:
            raise ValueError("At least three cases are required for train/validation/test splits.")

        total = len(ordered)
        train_end = int(total * (1.0 - held_out_fraction))
        validation_end = int(total * (1.0 - test_fraction))

        def select(ids: Sequence[str]) -> dict[str, tuple[str, ...]]:
            return {case_id: tuple(traces[case_id]) for case_id in ids}

        return cls(
            select(ordered[:train_end]),
            select(ordered[train_end:validation_end]),
            select(ordered[validation_end:]),
        )


@dataclass(frozen=True)
class TraceUtils:
    # Trace Utilities functions

    @classmethod
    def extract_traces(cls, events: pd.DataFrame) -> dict[str, tuple[str, ...]]:
        # Convert ordered event rows into immutable case traces

        ordered = events.sort_values([CASE_ID, INDEX])
        return {str(case_id): tuple(group[ACTIVITY].astype(str)) for case_id, group in ordered.groupby(CASE_ID, sort=False)}

    @classmethod
    def corrupt_traces(
        cls,
        traces: Mapping[str, Sequence[str]],
        fraction: float,
        rng: random.Random,
        activities: Sequence[str] | None = None,
    ) -> tuple[dict[str, tuple[str, ...]], int]:
        # Event-level label noise: replace the activity label of ``fraction`` of
        # the EVENTS with a different label from the vocabulary.
        #
        # This is the noise model of Mezini et al. (2026) Sec. 4.1 -- "noise is
        # applied by randomly replacing the activity label of selected events
        # with other labels from the activity vocabulary" -- and it differs from
        # :meth:`PrefixLog.corrupt_targets` in what it damages. Corrupting
        # targets leaves the traces intact, so every prefix the model conditions
        # on is still clean and the training log stays fully compliant with the
        # process knowledge. Corrupting *events* damages the trace itself, so
        # the corruption also shows up in later prefixes, in the Petri-net
        # replay, and in the log's compliance with the mined constraints -- the
        # regime the paper measures.
        #
        # Events are sampled globally rather than per trace, so the requested
        # fraction holds over the whole log rather than trace by trace.

        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be in [0, 1].")
        alphabet = tuple(activities) if activities is not None else tuple(
            sorted({activity for trace in traces.values() for activity in trace})
        )
        if fraction == 0.0 or len(alphabet) < 2:
            return {case_id: tuple(trace) for case_id, trace in traces.items()}, 0

        corrupted = {case_id: list(trace) for case_id, trace in traces.items()}
        positions = [
            (case_id, index)
            for case_id, trace in corrupted.items()
            for index in range(len(trace))
        ]
        count = round(len(positions) * fraction)
        changed = 0
        for case_id, index in rng.sample(positions, count):
            replacement = rng.choice(alphabet)
            if replacement == corrupted[case_id][index]:  # guarantee a real change
                replacement = alphabet[(alphabet.index(replacement) + 1) % len(alphabet)]
            corrupted[case_id][index] = replacement
            changed += 1
        return {case_id: tuple(trace) for case_id, trace in corrupted.items()}, changed

    @classmethod
    def case_order_by_time(cls, events: pd.DataFrame) -> tuple[str, ...]:
        # Case IDs ordered by the timestamp of their first event.
        #
        # This is the ordering the temporal split consumes. Ties (cases opening
        # in the same instant) are broken by the case's first row position in
        # the log, so the ordering is deterministic and stable across runs.

        first = (
            events.sort_values([CASE_ID, INDEX])
            .groupby(CASE_ID, sort=False)
            .agg(start=(TIMESTAMP, "first"), position=(INDEX, "first"))
            .sort_values(["start", "position"])
        )
        return tuple(str(case_id) for case_id in first.index)

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

# ------------------------------------------------------ config-driven preparation

def build_splits(events: pd.DataFrame, config: "ExperimentConfig") -> TraceSplits:
    # Dispatch to the split strategy named by the config.
    #
    # ``"random"`` reproduces the original seeded, shuffled, case-level split.
    # ``"temporal"`` orders cases by first-event timestamp and cuts by position,
    # keeping every case (no shuffling, no dropping, duplicates preserved).

    traces = TraceUtils.extract_traces(events)
    if config.data.split_strategy == "temporal":
        return TraceSplits.temporal(
            traces,
            case_order=TraceUtils.case_order_by_time(events),
            validation_fraction=config.data.validation_fraction,
            test_fraction=config.data.test_fraction,
        )
    return TraceSplits.from_traces(
        traces,
        validation_fraction=config.data.validation_fraction,
        test_fraction=config.data.test_fraction,
        seed=config.seed,
    )


def build_vocabulary(splits: TraceSplits, config: "ExperimentConfig") -> ActivityVocabulary:
    # Alphabet over the partitions named by ``vocabulary_scope``.
    #
    # ``"all"`` is what lets a temporal split keep every case: activities that
    # first appear late in the log are representable instead of forcing the
    # caller to drop the cases containing them.

    if config.data.vocabulary_scope == "all":
        return ActivityVocabulary.from_traces(
            [*splits.train.values(), *splits.validation.values(), *splits.test.values()]
        )
    return ActivityVocabulary.from_traces(splits.train.values())


def knowledge_traces(splits: TraceSplits, config: "ExperimentConfig") -> Mapping[str, tuple[str, ...]]:
    # Partition the background knowledge is discovered from.
    #
    # Returns the *traces*, so every knowledge artefact -- ``ProcessDFA``,
    # mined precedence constraints, discovered ``PetriNet`` -- is built from one
    # agreed-upon source instead of each call site picking its own.

    return splits.test if config.data.knowledge_source == "test" else splits.train


def unseen_activities(splits: TraceSplits, vocabulary: ActivityVocabulary) -> set[str]:
    # Held-out activities the vocabulary cannot represent (empty when
    # ``vocabulary_scope="all"``). Reported rather than filtered.

    known = set(vocabulary.class_to_id)
    return {
        activity
        for partition in (splits.validation, splits.test)
        for trace in partition.values()
        for activity in trace
        if activity not in known
    }


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
    #: State of a ``ReachabilityAutomaton`` after replaying this prefix. Unlike
    #: ``marking`` it is not a model input: no architecture reads it. It exists
    #: so a *loss* can index a state-aware constraint mask, which the
    #: last-token indexing of ``build_allowed_mask`` cannot express.
    automaton_state: int | None = None

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
    automaton_states: torch.Tensor | None = None

    def to(self, device: torch.device) -> "PrefixBatch":
        return PrefixBatch(
            case_ids=self.case_ids,
            tokens=self.tokens.to(device),
            lengths=self.lengths.to(device),
            targets=self.targets.to(device),
            markings=self.markings.to(device) if self.markings is not None else None,
            marking_sequences=self.marking_sequences.to(device) if self.marking_sequences is not None else None,
            automaton_states=self.automaton_states.to(device) if self.automaton_states is not None else None,
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
        automaton_states=None if examples[0].automaton_state is None else torch.tensor([example.automaton_state for example in examples], dtype=torch.long),
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
        prefixes = [[vocab_tokes[token_id] for token_id in example.token_ids[1:]]
                    for example in self.examples]
        # Il replay e' il costo dominante della costruzione di un log, e i
        # prefissi sono indipendenti: ``prefix_markings`` li distribuisce sui
        # core. Con pochi prefissi, o con un solo worker, ricade da sola sul
        # percorso seriale di ``prefix_marking``, quindi i valori sono gli stessi
        # in ogni caso.
        markings = petrinet.prefix_markings(prefixes)
        markings_log = [replace(example, marking=marking)
                        for example, marking in zip(self.examples, markings)]

        return PrefixLog(tuple(markings_log), self.vocabulary)

    def with_automaton_states(self, automaton) -> PrefixLog:
        """Attach the automaton state each prefix reaches, for a state-aware loss.

        The Petri-net reachability automaton distinguishes contexts that the
        directly-follows view collapses: the same last activity can leave the
        net in different markings, with different legal continuations. Indexing
        a constraint mask by *this* state instead of by the last token is what
        lets a loss use the net's memory. Replaying is cheap -- following edges
        in a deterministic automaton, not token replay on the net.

        A prefix that leaves the automaton's alphabet (no edge at all) is mapped
        to the trap state when there is one, otherwise to the initial state; the
        automaton built by :meth:`ReachabilityAutomaton.from_petri_net` with the
        full vocabulary as ``alphabet`` is complete, so this is a guard, not a
        path that normally runs.
        """
        tokens = self.vocabulary.tokens
        fallback = automaton.trap_state if automaton.trap_state is not None else automaton.init_state
        stated: list[PrefixExample] = []
        for example in self.examples:
            # token_ids[0] is START, which is not an automaton symbol.
            prefix = [tokens[token_id] for token_id in example.token_ids[1:]]
            state = automaton.state_after(prefix)
            stated.append(replace(
                example, automaton_state=fallback if state is None else state
            ))
        return PrefixLog(tuple(stated), self.vocabulary)

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

        # Una sequenza chiede il marking di ogni prefisso del suo caso. Quando
        # questa passata segue ``with_markings`` sono tutti gia' in cache; quando
        # non la segue, questo li calcola in parallelo invece che in serie.
        petrinet.prefix_markings(
            [prefix[:length]
             for example in longest.values()
             for prefix in ([vocab_tokens[t] for t in example.token_ids[1:]],)
             for length in range(1, len(prefix) + 1)]
        )

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
