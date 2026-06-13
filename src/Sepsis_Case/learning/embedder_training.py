"""Triplet training of the hierarchical embedder.

Implements the embedder objective from the T-LEAF paper (Eqs. 7-8): a hinge
triplet loss that pulls each constraint's DFA embedding ``z = q(A)`` towards the
embeddings of its *satisfying* traces and away from its *unsatisfying* traces.

    ℓ_triplet(A, w_T, w_F) = max{ d(z, z_T) − d(z, z_F) + m, 0 }

with ``d`` the squared Euclidean distance, ``z_T = q(w_T)`` a satisfying trace,
``z_F = q(w_F)`` an unsatisfying trace, and ``m`` the margin. (The paper's
printed formula has the two distances transposed; the sign here follows the
paper's stated intent -- anchor close to satisfying, far from unsatisfying.)

The paper trains ``qe`` first and then ``qm``; for a self-contained pipeline the
two are trained jointly end-to-end here, which the paper lists as an acceptable
alternative. Constraint DFAs are tiny, so triplets are processed one graph at a
time and accumulated into mini-batches before each optimiser step.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

import pandas as pd
import torch

from ..process.graph_encoding import FeatureSpace
from ..process.ltl_constraints import (
    PrecedenceConstraint,
    sample_satisfying_trace,
    sample_unsatisfying_trace,
)
from .embedder import HierarchicalEmbedder


@dataclass(frozen=True)
class EmbedderConfig:
    """Hyper-parameters for triplet training of the embedder."""

    epochs: int = 5
    learning_rate: float = 1e-3
    # With un-normalised embeddings the squared distances are O(10), so the hinge
    # margin is scaled accordingly to keep gradient pressure on easy triplets.
    margin: float = 5.0
    triplets_per_constraint: int = 24
    trace_length: int = 8
    batch_size: int = 16
    weight_decay: float = 0.0
    seed: int = 42


@dataclass(frozen=True)
class Triplet:
    constraint: PrecedenceConstraint
    satisfying: tuple[str, ...]
    unsatisfying: tuple[str, ...]


def build_triplets(
    constraints: Sequence[PrecedenceConstraint],
    alphabet: Sequence[str],
    config: EmbedderConfig,
    rng: random.Random,
) -> list[Triplet]:
    """Synthesize satisfying/unsatisfying trace pairs for every constraint."""

    triplets: list[Triplet] = []
    for constraint in constraints:
        for _ in range(config.triplets_per_constraint):
            length = rng.randint(max(3, config.trace_length - 2), config.trace_length + 2)
            satisfying = sample_satisfying_trace(constraint, alphabet, length=length, rng=rng)
            unsatisfying = sample_unsatisfying_trace(constraint, alphabet, length=length, rng=rng)
            triplets.append(Triplet(constraint, satisfying, unsatisfying))
    rng.shuffle(triplets)
    return triplets


def _triplet_distances(
    embedder: HierarchicalEmbedder,
    space: FeatureSpace,
    triplet: Triplet,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    anchor = embedder.embed_dfa(triplet.constraint.to_dfa(), space, rng=rng)
    positive = embedder.embed_trace(triplet.satisfying, space, rng=rng)
    negative = embedder.embed_trace(triplet.unsatisfying, space, rng=rng)
    d_pos = (anchor - positive).pow(2).sum()
    d_neg = (anchor - negative).pow(2).sum()
    return d_pos, d_neg


def train_embedder(
    embedder: HierarchicalEmbedder,
    space: FeatureSpace,
    constraints: Sequence[PrecedenceConstraint],
    alphabet: Sequence[str],
    config: EmbedderConfig | None = None,
    *,
    device: torch.device | str = "cpu",
    verbose: bool = True,
) -> pd.DataFrame:
    """Jointly train ``qe`` and ``qm`` with the triplet hinge loss."""

    config = config or EmbedderConfig()
    if not constraints:
        raise ValueError("Cannot train the embedder without any constraints.")
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)

    embedder = embedder.to(device)
    space = space.to(device)
    optimizer = torch.optim.Adam(
        embedder.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    margin = config.margin
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        embedder.train()
        triplets = build_triplets(constraints, alphabet, config, rng)
        running_loss = 0.0
        satisfied = 0
        batch_losses: list[torch.Tensor] = []
        optimizer.zero_grad(set_to_none=True)

        for index, triplet in enumerate(triplets, start=1):
            d_pos, d_neg = _triplet_distances(embedder, space, triplet, rng)
            loss = torch.clamp(d_pos - d_neg + margin, min=0.0)
            batch_losses.append(loss)
            running_loss += float(loss)
            satisfied += int(d_pos < d_neg)

            if len(batch_losses) == config.batch_size or index == len(triplets):
                torch.stack(batch_losses).mean().backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                batch_losses = []

        row = {
            "epoch": float(epoch),
            "triplet_loss": running_loss / len(triplets),
            "triplet_accuracy": satisfied / len(triplets),
        }
        history.append(row)
        if verbose:
            print(
                f"embedder epoch {epoch:02d} "
                f"loss={row['triplet_loss']:.4f} acc={row['triplet_accuracy']:.3f}"
            )

    return pd.DataFrame(history)


@torch.no_grad()
def evaluate_embedder(
    embedder: HierarchicalEmbedder,
    space: FeatureSpace,
    constraints: Sequence[PrecedenceConstraint],
    alphabet: Sequence[str],
    config: EmbedderConfig | None = None,
    *,
    device: torch.device | str = "cpu",
) -> float:
    """Fraction of held-out triplets ranked correctly (d_pos < d_neg)."""

    config = config or EmbedderConfig()
    rng = random.Random(config.seed + 1)
    embedder = embedder.to(device).eval()
    space = space.to(device)
    triplets = build_triplets(constraints, alphabet, config, rng)
    correct = 0
    for triplet in triplets:
        d_pos, d_neg = _triplet_distances(embedder, space, triplet, rng)
        correct += int(d_pos < d_neg)
    return correct / len(triplets)
