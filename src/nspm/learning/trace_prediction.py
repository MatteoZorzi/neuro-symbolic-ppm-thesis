"""Whole-trace (suffix) prediction and its conformance/accuracy metrics.

The next-activity task in :mod:`learning.evaluation` scores a single step from a
ground-truth prefix. This module evaluates the harder PPM *suffix-prediction*
task (Di Francescomarino, Donadello & Maggi, 2026, Ch. 5): given a prefix, the
model **autoregressively generates the remaining trace** from its own previous
predictions, so errors compound. The whole predicted trace is then scored against
the ground truth with the field-standard **Damerau-Levenshtein similarity**.

Two generation setups are provided:

* :func:`evaluate_trace_from_start` -- generate the entire trace from ``START``
  (the headline "whole predicted trace" story);
* :func:`evaluate_suffix_prediction` -- generate the suffix after several prefix
  lengths (the robust quantitative protocol, many samples per trace).

Both accept an optional ``allowed_mask`` that forbids directly-follows-illegal
next activities at every decoding step. This is the book's **output refinement**
mechanism (Ch. 7, mechanism A): symbolic knowledge applied *at inference time*
with no retraining, complementing the logic-in-loss models (checker/embedder).

Because the sequence models output activity classes only (no end-of-sequence
token, which the book notes a full suffix predictor needs, Ch. 5), generation is
**length-conditioned**: exactly ``len(true continuation)`` steps are produced.
This isolates ordering accuracy from length prediction and is stated as a
limitation rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
from torch import nn

from ..data.preparation import ActivityVocabulary
from ..process.automaton import START, ProcessDFA
from ..process.ltl_constraints import PrecedenceConstraint


def damerau_levenshtein_distance(
    a: Sequence[str], b: Sequence[str]
) -> int:
    """Optimal string-alignment Damerau-Levenshtein distance.

    Counts insertions, deletions, substitutions and adjacent transpositions
    between two activity sequences. This is the distance underlying the standard
    suffix-prediction similarity metric in predictive process monitoring.
    """

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


def dl_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    """1 - normalised Damerau-Levenshtein distance (1.0 = identical)."""

    if not a and not b:
        return 1.0
    longest = max(len(a), len(b))
    return 1.0 - damerau_levenshtein_distance(a, b) / longest


@torch.no_grad()
def generate_continuation(
    model: nn.Module,
    vocabulary: ActivityVocabulary,
    prefix_token_ids: Sequence[int],
    n_steps: int,
    device: torch.device,
    allowed_mask: torch.Tensor | None = None,
) -> list[str]:
    """Greedily roll out ``n_steps`` activities, feeding predictions back in.

    When ``allowed_mask`` is supplied (shape ``[n_tokens, n_classes]``), the
    classes forbidden by the directly-follows automaton for the current last
    token are masked before the argmax -- the inference-time *output-refinement*
    guardrail. Without it, decoding is unconstrained (free-running).
    """

    model.eval()
    token_to_id = vocabulary.token_to_id
    activities = vocabulary.activities
    if allowed_mask is not None:
        allowed_mask = allowed_mask.to(device)

    history = list(prefix_token_ids)
    generated: list[str] = []
    for _ in range(max(0, n_steps)):
        tokens = torch.tensor([history], dtype=torch.long, device=device)
        lengths = torch.tensor([len(history)], dtype=torch.long)
        logits = model(tokens, lengths)[0]
        if allowed_mask is not None:
            allowed_row = allowed_mask[history[-1]]
            logits = logits.masked_fill(~allowed_row, float("-inf"))
        class_id = int(logits.argmax())
        activity = activities[class_id]
        generated.append(activity)
        history.append(token_to_id[activity])
    return generated


@dataclass
class TracePredictionResult:
    """Aggregate whole-trace metrics plus a few example traces for display."""

    setup: str
    n_examples: int
    dl_similarity: float
    activity_accuracy: float
    exact_match_rate: float
    dfa_violation_rate: float
    precedence_violation_rate: float
    examples: list[dict] = field(default_factory=list)

    def metrics(self) -> dict[str, float]:
        return {
            "dl_similarity": self.dl_similarity,
            "activity_accuracy": self.activity_accuracy,
            "exact_match_rate": self.exact_match_rate,
            "dfa_violation_rate": self.dfa_violation_rate,
            "precedence_violation_rate": self.precedence_violation_rate,
        }


def _generated_dfa_violations(
    automaton: ProcessDFA, prefix: Sequence[str], generated: Sequence[str]
) -> tuple[int, int]:
    """(violations, transitions) for the model-generated part of the trace.

    A transition is a violation only when its source state is *constrained* --
    i.e. it has at least one observed real-activity successor -- and the chosen
    activity is not among them. Transitions out of unconstrained states (seen
    only at trace end, or never seen as a source) impose no directly-follows
    rule, so they are legal by definition. This matches exactly what the
    inference-time decoding mask can enforce (see
    :meth:`ProcessDFA.allowed_activities` and
    :func:`learning.logic.build_allowed_mask`), so masked decoding drives this
    rate to 0 by construction, and it is the same notion of "DFA violation"
    used for the single-step task in :mod:`learning.evaluation`.
    """

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


def _violates_precedence(
    constraints: Sequence[PrecedenceConstraint], trace: Sequence[str]
) -> bool:
    """True if any *relevant* mined precedence rule is broken by the trace."""

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
    precedence_violations = 0

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

        if constraints:
            full_trace = [*record["prefix"], *predicted]
            precedence_violations += int(_violates_precedence(constraints, full_trace))

    return TracePredictionResult(
        setup=setup,
        n_examples=total,
        dl_similarity=dl_sum / total,
        activity_accuracy=position_hits / position_total if position_total else 0.0,
        exact_match_rate=exact / total,
        dfa_violation_rate=dfa_violations / dfa_transitions if dfa_transitions else 0.0,
        precedence_violation_rate=(
            precedence_violations / total if constraints else float("nan")
        ),
        examples=records[:n_examples],
    )


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
) -> TracePredictionResult:
    """Generate each whole trace from ``START`` and score it against the truth."""

    start_id = vocabulary.token_to_id[START]
    records: list[dict] = []
    for case_id, trace in traces.items():
        trace = list(trace)
        if not trace:
            continue
        predicted = generate_continuation(
            model, vocabulary, [start_id], len(trace), device, allowed_mask
        )
        records.append(
            {"case_id": case_id, "prefix": [], "true": trace, "predicted": predicted}
        )
    return _score_records(records, "from_start", automaton, constraints, n_examples)


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
) -> TracePredictionResult:
    """Generate the suffix after each prefix length and score it (PPM protocol)."""

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
                model, vocabulary, prefix_token_ids, len(true_suffix), device, allowed_mask
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
    return _score_records(records, "suffix", automaton, constraints, n_examples)
