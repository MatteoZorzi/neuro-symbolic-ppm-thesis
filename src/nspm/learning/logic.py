# Differentiable process constraints derived from the empirical DFA

from __future__ import annotations

import random
from typing import Sequence

import torch

from ..data.preparation import ActivityVocabulary, PAD
from ..process.automaton import ProcessDFA, START
from ..process.graph_encoding import FeatureSpace
from ..process.ltl_constraints import PrecedenceConstraint, relevant_constraints
from .models import symbolic_input


# Map the last input token to the classes allowed by the automaton
def build_allowed_mask(
    automaton: ProcessDFA,
    vocabulary: ActivityVocabulary,
) -> torch.Tensor:

    mask = torch.zeros(
        (len(vocabulary.tokens), len(vocabulary.activities)), dtype=torch.bool
    )
    class_to_id = vocabulary.class_to_id
    for token_id, state in enumerate(vocabulary.tokens):
        if state == PAD:  # Padding is never a real state.
            mask[token_id] = True
            continue
        allowed_classes = [
            class_to_id[action]
            for action in automaton.allowed_activities(state)
            if action in class_to_id
        ]
        if allowed_classes:
            mask[token_id, allowed_classes] = True
        else:
            mask[token_id] = True

    # A prefix always starts with START, so this row must describe start actions.
    start_id = vocabulary.token_to_id[START]
    if not mask[start_id].any():
        raise ValueError("The automaton contains no valid starting action.")
    return mask


# Map a **reachability-automaton state** to the classes it allows
def build_state_mask(automaton, vocabulary: ActivityVocabulary) -> torch.Tensor:

    n_states = max(automaton.states) + 1
    mask = torch.zeros((n_states, len(vocabulary.activities)), dtype=torch.bool)
    class_to_id = vocabulary.class_to_id
    for state in automaton.states:
        allowed_classes = [
            class_to_id[activity]
            for activity in automaton.allowed_activities(state)
            if activity in class_to_id
        ]
        if allowed_classes and state != automaton.trap_state:
            mask[state, allowed_classes] = True
        else:
            mask[state] = True
    return mask


# Select the final non-padding input token for each prefix
def last_token_ids(tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:

    row_ids = torch.arange(tokens.size(0), device=tokens.device)
    return tokens[row_ids, lengths - 1]


# Mean probability assigned to DFA-forbidden next activities
def forbidden_probability_mass(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    allowed_mask: torch.Tensor,
    state_ids: torch.Tensor | None = None,
) -> torch.Tensor:

    if state_ids is None:
        state_ids = last_token_ids(tokens, lengths)
    allowed = allowed_mask[state_ids]
    probabilities = torch.softmax(logits, dim=1)
    return (probabilities * ~allowed).sum(dim=1).mean()


# Boolean vector indicating whether each top-1 prediction is forbidden
def prediction_violations(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    allowed_mask: torch.Tensor,
) -> torch.Tensor:

    state_ids = last_token_ids(tokens, lengths)
    predictions = logits.argmax(dim=1)
    allowed = allowed_mask[state_ids]
    row_ids = torch.arange(logits.size(0), device=logits.device)
    return ~allowed[row_ids, predictions]


# Fraction of next-activity predictions that break a precedence constraint
@torch.no_grad()
def precedence_violation_rate(
    model,
    data_loader,
    constraints: Sequence[PrecedenceConstraint],
    vocabulary: ActivityVocabulary,
    device: torch.device,
) -> float:

    required_earlier: dict[str, set[str]] = {}
    for constraint in constraints:
        required_earlier.setdefault(constraint.later, set()).add(constraint.earlier)
    token_strings = list(vocabulary.tokens)
    activities = vocabulary.activities

    model.eval()
    total = 0
    violations = 0
    for raw_batch in data_loader:
        batch = raw_batch.to(device)
        predictions = model(batch.tokens, batch.lengths, symbolic_input(model, batch)).argmax(dim=1)
        for i in range(predictions.size(0)):
            total += 1
            predicted = activities[int(predictions[i])]
            needed = required_earlier.get(predicted)
            if not needed:
                continue
            prefix = {
                token_strings[int(batch.tokens[i, position])]
                for position in range(int(batch.lengths[i]))
            }
            prefix.discard(PAD)
            prefix.discard(START)
            if not needed.issubset(prefix):
                violations += 1
    return violations / total if total else 0.0


# The learned-embedder T-LEAF logic loss ``‖q(A) − q(w_pred)‖²``
class EmbeddingLogicLoss:

    def __init__(
        self,
        embedder,
        space: FeatureSpace,
        constraints: Sequence[PrecedenceConstraint],
        vocabulary: ActivityVocabulary,
        *,
        max_examples: int = 16,
        max_constraints: int = 3,
        seed: int = 0,
    ) -> None:
        if tuple(space.activities) != tuple(vocabulary.activities):
            raise ValueError(
                "FeatureSpace activities must match the vocabulary activity order "
                "so that model logits align with activity features."
            )
        self.embedder = embedder.eval()
        for parameter in self.embedder.parameters():
            parameter.requires_grad_(False)
        self.space = space
        self.constraints = list(constraints)
        self.vocabulary = vocabulary
        self.max_examples = max_examples
        self.max_constraints = max_constraints
        self.rng = random.Random(seed)
        self._token_strings = list(vocabulary.tokens)

        # q(A) is constant while the embedder is frozen: cache it once (detached),
        # keyed by constraint for O(1) lookup of a relevant constraint's anchor.
        with torch.no_grad():
            self._anchor_by_constraint = {
                constraint: self.embedder.embed_dfa(constraint.to_dfa(), space).detach()
                for constraint in self.constraints
            }

    def _prefix_activities(self, token_row: torch.Tensor, length: int) -> list[str]:
        activities: list[str] = []
        for position in range(int(length)):
            token = self._token_strings[int(token_row[position])]
            if token in (PAD, START):
                continue
            activities.append(token)
        return activities

    def __call__(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = logits.size(0)
        indices = list(range(batch_size))
        if batch_size > self.max_examples:
            indices = self.rng.sample(indices, self.max_examples)

        probabilities = torch.softmax(logits, dim=1)
        distances: list[torch.Tensor] = []
        for i in indices:
            prefix = self._prefix_activities(tokens[i], int(lengths[i]))
            if not prefix:
                continue
            # The predicted activity is appended; ``later`` activities present in
            # the predicted continuation make a constraint relevant.
            top_activity = self.vocabulary.activities[int(probabilities[i].argmax())]
            context = prefix + [top_activity]
            relevant = relevant_constraints(self.constraints, context)
            if not relevant:
                continue
            if len(relevant) > self.max_constraints:
                relevant = self.rng.sample(relevant, self.max_constraints)

            soft_feature = self.space.soft_activity_feat(probabilities[i])
            predicted = self.embedder.embed_trace(
                tuple(prefix) + (top_activity,), self.space, soft_last=soft_feature
            )
            for constraint in relevant:
                anchor = self._anchor_by_constraint[constraint]
                distances.append((anchor - predicted).pow(2).sum())

        if not distances:
            return logits.sum() * 0.0  # No relevant constraints this step.
        return torch.stack(distances).mean()
