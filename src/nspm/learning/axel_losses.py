# The two logic losses of Mezini et al. (nesy-suffix-prediction-dfa), ported to
# this setup: the local one at one step, the global one over a whole rollout.

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..process.automaton import END, START, ProcessDFA


# Our ``ProcessDFA`` in tensor form, differentiable
@dataclass(frozen=True)
class TensorDFA:

    transitions: torch.Tensor      # (n_actions, n_states, n_states)
    accepting: torch.Tensor        # (n_states,) 1.0 sugli stati accettanti
    n_states: int
    n_actions: int
    end_action: int
    trap_state: int

    @classmethod
    def from_process_dfa(cls, automaton: ProcessDFA, activities: tuple[str, ...],
                         device) -> "TensorDFA":
        n = len(activities)
        state_of = {START: 0}
        for index, activity in enumerate(activities):
            state_of[activity] = index + 1
        end_state, trap_state = n + 1, n + 2
        n_states, n_actions = n + 3, n + 1
        end_action = n

        transitions = torch.zeros((n_actions, n_states, n_states), device=device)

        def wire(source_name: str, source_index: int) -> None:
            allowed = automaton.allowed_next.get(source_name, frozenset())
            for action, activity in enumerate(activities):
                target = state_of[activity] if activity in allowed else trap_state
                transitions[action, source_index, target] = 1.0
            transitions[end_action, source_index,
                        end_state if END in allowed else trap_state] = 1.0

        wire(START, 0)
        for activity in activities:
            wire(activity, state_of[activity])
        # Absorbing: once accepted or trapped, the trace does not move again.
        # Without this the rollout would walk off the matrix.
        for absorbing in (end_state, trap_state):
            transitions[:, absorbing, absorbing] = 1.0

        accepting = torch.zeros(n_states, device=device)
        accepting[end_state] = 1.0
        return cls(transitions, accepting, n_states, n_actions, end_action, trap_state)

    # One step over distributions: ``state`` (B, S), ``action`` (B, A) -> (B, S)
    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:

        return torch.einsum("ba,bs,ast->bt", action, state, self.transitions)

    def initial_state(self, state_indices: torch.Tensor) -> torch.Tensor:
        return F.one_hot(state_indices, num_classes=self.n_states).float()


# LLL: weighted cross-entropy plus a penalty on the mass the automaton rejects
class LocalLogicLoss(nn.Module):

    def __init__(self, allowed_mask: torch.Tensor, alpha: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
        self.register_buffer("allowed_mask", allowed_mask)
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                state_ids: torch.Tensor) -> torch.Tensor:
        rejects = ~self.allowed_mask[state_ids]           # (B, C)

        cross_entropy = F.cross_entropy(logits, targets, reduction="none")
        target_rejected = rejects.gather(1, targets.unsqueeze(1)).squeeze(1)
        weights = (~target_rejected).float()
        # If EVERY target in a batch is forbidden, no supervision is left: the
        # epsilon of the original code avoids the division by zero and the term
        # cancels on its own.
        weighted = (cross_entropy * weights).sum() / (weights.sum() + 1e-6)

        invalid_mass = (torch.softmax(logits, dim=1) * rejects).sum(dim=1)
        penalty = -torch.log(1.0 - invalid_mass + 1e-6).mean()

        return self.alpha * weighted + (1.0 - self.alpha) * penalty


# GLL: the model runs on by itself and the automaton judges the whole trace
class GlobalLogicLoss(nn.Module):

    def __init__(self, dfa: TensorDFA, *, horizon: int, alpha: float = 0.5,
                 temperature: float = 0.5, num_samples: int = 4) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
        self.dfa = dfa
        self.horizon = horizon
        #: Blended by the caller, not here: as in the reference runner, GLL
        #: returns the logic penalty alone and ``alpha * CE + (1-alpha) * GLL``
        #: happens in the training loop. The value only travels with its loss.
        self.alpha = alpha
        self.temperature = temperature
        self.num_samples = num_samples

    @staticmethod
    def _gumbel_softmax(logits: torch.Tensor, temperature: float,
                        eps: float = 1e-10) -> torch.Tensor:
        uniform = torch.rand_like(logits)
        noise = -torch.log(-torch.log(uniform + eps) + eps)
        return torch.softmax((F.log_softmax(logits, dim=-1) + noise) / temperature,
                             dim=-1)

    def forward(self, model, tokens: torch.Tensor, lengths: torch.Tensor,
                state_ids: torch.Tensor) -> torch.Tensor:
        samples = self.num_samples
        batch = tokens.size(0)

        tokens = tokens.repeat_interleave(samples, dim=0)
        lengths = lengths.repeat_interleave(samples, dim=0)
        # The automaton state after the prefix is known in hard form: in the
        # directly-follows view the state IS the last activity. Replaying the
        # prefix in relaxed form would spend gradient on what the model did not generate.
        state = self.dfa.initial_state(state_ids.repeat_interleave(samples, dim=0))

        logits, hidden = model.encode(tokens, lengths)
        for _ in range(self.horizon):
            soft_action = self._gumbel_softmax(logits, self.temperature)
            padded = F.pad(soft_action, (0, self.dfa.n_actions - soft_action.size(1)))
            state = self.dfa.step(state, padded)
            logits, hidden = model.forward_from_state(soft_action, hidden)

        end_action = F.one_hot(
            torch.full((tokens.size(0),), self.dfa.end_action, device=tokens.device),
            num_classes=self.dfa.n_actions,
        ).float()
        state = self.dfa.step(state, end_action)

        acceptance = (state * self.dfa.accepting).sum(dim=1).view(batch, samples)
        return -torch.log(acceptance.mean(dim=1).clamp(min=1e-10)).mean()
