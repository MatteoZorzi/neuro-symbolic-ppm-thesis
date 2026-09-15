# Whole-trace (suffix) prediction and its conformance/accuracy metrics

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
from torch import nn

from ..data.preparation import ActivityVocabulary
from ..process.automaton import START, ProcessDFA
from ..process.ltl_constraints import PrecedenceConstraint


# Optimal string-alignment Damerau-Levenshtein distance
def damerau_levenshtein_distance(
    a: Sequence[str], b: Sequence[str]
) -> int:

    a = list(a)
    b = list(b)
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    distance = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        distance[i][0] = i
    for j in range(m + 1):
        distance[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            distance[i][j] = min(
                distance[i - 1][j] + 1,  # deletion
                distance[i][j - 1] + 1,  # insertion
                distance[i - 1][j - 1] + cost,  # substitution
            )
            if (
                i > 1
                and j > 1
                and a[i - 1] == b[j - 2]
                and a[i - 2] == b[j - 1]
            ):
                distance[i][j] = min(
                    distance[i][j], distance[i - 2][j - 2] + 1
                )  # transposition
    return distance[n][m]


# 1 - normalised Damerau-Levenshtein distance (1.0 = identical)
def dl_similarity(a: Sequence[str], b: Sequence[str]) -> float:

    if not a and not b:
        return 1.0
    longest = max(len(a), len(b))
    return 1.0 - damerau_levenshtein_distance(a, b) / longest


# Keeps the symbolic marking channel in step with an autoregressive rollout
class _MarkingRollout:

    def __init__(self, petrinet, expects_sequences: bool) -> None:
        self._petrinet = petrinet
        self._expects_sequences = expects_sequences
        self._activities: list[str] = []
        self._markings: list[tuple[int, ...]] = []

    def reset(self, prefix_activities: Sequence[str]) -> None:
        self._activities = list(prefix_activities)
        self._markings = [
            self._petrinet.prefix_marking(()),
            *self._petrinet.marking_sequence(self._activities),
        ]

    def push(self, activity: str) -> None:
        self._activities.append(activity)
        self._markings.append(self._petrinet.prefix_marking(self._activities))

    def tensor(self, device: torch.device) -> torch.Tensor:
        if self._expects_sequences:  # (1, L, P), L aligned with the token count
            return torch.tensor([self._markings], dtype=torch.float32, device=device)
        return torch.tensor([self._markings[-1]], dtype=torch.float32, device=device)


# Greedily roll out ``n_steps`` activities, feeding predictions back in
@torch.no_grad()
def generate_continuation(
    model: nn.Module,
    vocabulary: ActivityVocabulary,
    prefix_token_ids: Sequence[int],
    n_steps: int,
    device: torch.device,
    allowed_mask: torch.Tensor | None = None,
    petrinet=None,
) -> list[str]:

    model.eval()
    token_to_id = vocabulary.token_to_id
    activities = vocabulary.activities
    if allowed_mask is not None:
        allowed_mask = allowed_mask.to(device)

    encoder = getattr(model, "marking_encoder", None)
    rollout: _MarkingRollout | None = None
    if encoder is not None:
        if petrinet is None:
            raise ValueError(
                "This model reads Petri-net markings, so free-running generation "
                "needs the net to replay its own predictions: pass petrinet=."
            )
        rollout = _MarkingRollout(petrinet, encoder.expects_sequences)
        # token_ids start with START, which is not an event
        rollout.reset([vocabulary.tokens[t] for t in prefix_token_ids[1:]])

    history = list(prefix_token_ids)
    generated: list[str] = []
    for _ in range(max(0, n_steps)):
        tokens = torch.tensor([history], dtype=torch.long, device=device)
        lengths = torch.tensor([len(history)], dtype=torch.long)
        markings = None if rollout is None else rollout.tensor(device)
        logits = model(tokens, lengths, markings)[0]
        if allowed_mask is not None:
            allowed_row = allowed_mask[history[-1]]
            logits = logits.masked_fill(~allowed_row, float("-inf"))
        class_id = int(logits.argmax())
        activity = activities[class_id]
        generated.append(activity)
        history.append(token_to_id[activity])
        if rollout is not None:
            rollout.push(activity)
    return generated


# Aggregate whole-trace metrics plus a few example traces for display
@dataclass
class TracePredictionResult:

    setup: str
    n_examples: int
    dl_similarity: float
    activity_accuracy: float
    exact_match_rate: float
    dfa_violation_rate: float
    precedence_violation_rate: float
    #: The same illegal transitions, but counted against the automaton projected
    #: from the Petri net instead of the empirical directly-follows one. NaN when
    #: the net automaton is not handed to the evaluator.
    #:
    #: It exists because the two structures do not forbid the same things: the
    #: projection of the net is an over-approximation, so it admits behaviour
    #: never observed but structurally possible. A method trained against the net
    #: has to be measured against the net, or the very generalisations it was
    #: asked to make get counted as violations.
    dfa_violation_rate_net: float = float("nan")
    #: Mean token-replay fitness of the WHOLE trace (real prefix + generated
    #: suffix) against the Petri net. NaN when the net is not handed to the
    #: evaluator. It is the only conformance metric that looks at the trace as a
    #: whole object rather than step by step: ``dfa_violation_rate`` counts the
    #: illegal transitions, this one says how much of what the model produced the
    #: net manages to replay.
    net_fitness: float = float("nan")
    examples: list[dict] = field(default_factory=list)

    def metrics(self) -> dict[str, float]:
        return {
            "dl_similarity": self.dl_similarity,
            "activity_accuracy": self.activity_accuracy,
            "exact_match_rate": self.exact_match_rate,
            "dfa_violation_rate": self.dfa_violation_rate,
            "dfa_violation_rate_net": self.dfa_violation_rate_net,
            "precedence_violation_rate": self.precedence_violation_rate,
            "net_fitness": self.net_fitness,
        }


# (violations, transitions) for the model-generated part of the trace
def _generated_dfa_violations(
    automaton: ProcessDFA, prefix: Sequence[str], generated: Sequence[str]
) -> tuple[int, int]:

    previous = prefix[-1] if prefix else START
    violations = 0
    transitions = 0
    for activity in generated:
        transitions += 1
        allowed = automaton.allowed_activities(previous)
        if allowed and activity not in allowed:
            violations += 1
        previous = activity
    return violations, transitions


# True if any *relevant* mined precedence rule is broken by the trace
def _violates_precedence(
    constraints: Sequence[PrecedenceConstraint], trace: Sequence[str]
) -> bool:

    present = set(trace)
    for constraint in constraints:
        if constraint.later in present and not constraint.is_satisfied(trace):
            return True
    return False


def _score_records(
    records: list[dict],
    setup: str,
    automaton: ProcessDFA,
    constraints: Sequence[PrecedenceConstraint] | None,
    n_examples: int,
    petrinet=None,
    automaton_net: ProcessDFA | None = None,
) -> TracePredictionResult:
    if not records:
        raise ValueError("Cannot score whole-trace prediction without examples.")

    constraints = list(constraints or ())
    total = len(records)
    dl_sum = 0.0
    position_hits = 0
    position_total = 0
    exact = 0
    dfa_violations = 0
    dfa_transitions = 0
    # The traces are already generated: running them past a second automaton is
    # a pass of string comparisons, not a second inference. Measuring against
    # both structures therefore costs almost nothing, and does not force a choice
    # of which of the two is "the" yardstick.
    net_violations = 0
    net_transitions = 0
    precedence_violations = 0
    full_traces: list[list[str]] = []

    for record in records:
        true = record["true"]
        predicted = record["predicted"]
        dl_sum += dl_similarity(true, predicted)
        for expected, got in zip(true, predicted):
            position_total += 1
            position_hits += int(expected == got)
        exact += int(list(true) == list(predicted))

        violations, transitions = _generated_dfa_violations(
            automaton, record["prefix"], predicted
        )
        dfa_violations += violations
        dfa_transitions += transitions

        if automaton_net is not None:
            violations, transitions = _generated_dfa_violations(
                automaton_net, record["prefix"], predicted
            )
            net_violations += violations
            net_transitions += transitions

        full_trace = [*record["prefix"], *predicted]
        full_traces.append(full_trace)
        if constraints:
            precedence_violations += int(_violates_precedence(constraints, full_trace))

    # A single call over all the reconstructed traces: this is a metric, not a
    # feature, so bulk replay is the right shape here.
    net_fitness = float("nan")
    if petrinet is not None:
        scores = petrinet.trace_fitness(full_traces)
        net_fitness = sum(scores) / len(scores) if scores else float("nan")

    return TracePredictionResult(
        setup=setup,
        n_examples=total,
        dl_similarity=dl_sum / total,
        activity_accuracy=position_hits / position_total if position_total else 0.0,
        exact_match_rate=exact / total,
        dfa_violation_rate=dfa_violations / dfa_transitions if dfa_transitions else 0.0,
        dfa_violation_rate_net=(
            net_violations / net_transitions if net_transitions
            else (0.0 if automaton_net is not None else float("nan"))
        ),
        precedence_violation_rate=(
            precedence_violations / total if constraints else float("nan")
        ),
        net_fitness=net_fitness,
        examples=records[:n_examples],
    )


# Generate each whole trace from ``START`` and score it against the truth
def evaluate_trace_from_start(
    model: nn.Module,
    traces: Mapping[str, Sequence[str]],
    vocabulary: ActivityVocabulary,
    automaton: ProcessDFA,
    device: torch.device,
    *,
    allowed_mask: torch.Tensor | None = None,
    constraints: Sequence[PrecedenceConstraint] | None = None,
    n_examples: int = 5,
    petrinet=None,
) -> TracePredictionResult:

    start_id = vocabulary.token_to_id[START]
    records: list[dict] = []
    for case_id, trace in traces.items():
        trace = list(trace)
        if not trace:
            continue
        predicted = generate_continuation(
            model, vocabulary, [start_id], len(trace), device, allowed_mask, petrinet
        )
        records.append(
            {"case_id": case_id, "prefix": [], "true": trace, "predicted": predicted}
        )
    return _score_records(records, "from_start", automaton, constraints, n_examples,
                          petrinet)


# Generate the suffix after each prefix length and score it (PPM protocol)
def evaluate_suffix_prediction(
    model: nn.Module,
    traces: Mapping[str, Sequence[str]],
    vocabulary: ActivityVocabulary,
    automaton: ProcessDFA,
    device: torch.device,
    *,
    prefix_lengths: Sequence[int] = (2, 4, 8),
    allowed_mask: torch.Tensor | None = None,
    constraints: Sequence[PrecedenceConstraint] | None = None,
    n_examples: int = 5,
    petrinet=None,
    fitness_net=None,
    automaton_net: ProcessDFA | None = None,
) -> TracePredictionResult:

    start_id = vocabulary.token_to_id[START]
    records: list[dict] = []
    for case_id, trace in traces.items():
        trace = list(trace)
        for k in prefix_lengths:
            if k < 1 or k >= len(trace):
                continue
            prefix = trace[:k]
            true_suffix = trace[k:]
            prefix_token_ids = [start_id, *vocabulary.encode_prefix(prefix)]
            predicted = generate_continuation(
                model, vocabulary, prefix_token_ids, len(true_suffix), device,
                allowed_mask, petrinet,
            )
            records.append(
                {
                    "case_id": case_id,
                    "prefix_length": k,
                    "prefix": prefix,
                    "true": true_suffix,
                    "predicted": predicted,
                }
            )
    return _score_records(records, "suffix", automaton, constraints, n_examples,
                          fitness_net, automaton_net)
