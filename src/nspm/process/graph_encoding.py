# Encode constraint DFAs and traces as graphs for the hierarchical embedder

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Data

from .ltl_constraints import COMMON, FINAL, INIT, ConstraintDFA, Guard


# Fixed feature vectors for activities, operators and node types
@dataclass
class FeatureSpace:

    activities: tuple[str, ...]
    prop_dim: int
    node_dim: int
    activity_feat: torch.Tensor  # [n_activities, prop_dim]
    and_feat: torch.Tensor       # [prop_dim]
    or_feat: torch.Tensor        # [prop_dim]
    true_feat: torch.Tensor      # [prop_dim]
    node_type_feat: torch.Tensor  # [3, node_dim] indexed by INIT/COMMON/FINAL

    @classmethod
    def create(
        cls,
        activities,
        *,
        prop_dim: int = 50,
        node_dim: int = 100,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> "FeatureSpace":
        activities = tuple(activities)
        generator = torch.Generator().manual_seed(seed)
        activity_feat = torch.rand(len(activities), prop_dim, generator=generator)
        and_feat = torch.rand(prop_dim, generator=generator)
        or_feat = torch.rand(prop_dim, generator=generator)
        true_feat = torch.ones(prop_dim)
        node_type_feat = torch.rand(3, node_dim, generator=generator)
        space = cls(
            activities=activities,
            prop_dim=prop_dim,
            node_dim=node_dim,
            activity_feat=activity_feat,
            and_feat=and_feat,
            or_feat=or_feat,
            true_feat=true_feat,
            node_type_feat=node_type_feat,
        )
        return space.to(device)

    def to(self, device: torch.device | str) -> "FeatureSpace":
        self.activity_feat = self.activity_feat.to(device)
        self.and_feat = self.and_feat.to(device)
        self.or_feat = self.or_feat.to(device)
        self.true_feat = self.true_feat.to(device)
        self.node_type_feat = self.node_type_feat.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self.activity_feat.device

    @property
    def index(self) -> dict[str, int]:
        return {activity: i for i, activity in enumerate(self.activities)}

    # Feature for ``activity`` (positive) or its negation ``1 - feat``
    def literal_feat(self, activity: str, positive: bool) -> torch.Tensor:

        vector = self.activity_feat[self.index[activity]]
        return vector if positive else (1.0 - vector)

    # Probability-weighted activity feature ``Σ_a p_a · feat(a)``
    def soft_activity_feat(self, probabilities: torch.Tensor) -> torch.Tensor:

        return probabilities @ self.activity_feat


# Return the guard as an OR of ANDs of literal feature vectors
def _guard_disjuncts(guard: Guard, space: FeatureSpace) -> list[list[torch.Tensor]]:

    if guard.is_true:
        return [[space.true_feat]]
    literals: list[torch.Tensor] = []
    for activity in guard.positives:
        literals.append(space.literal_feat(activity, positive=True))
    for activity in guard.negatives:
        literals.append(space.literal_feat(activity, positive=False))
    if not literals:  # Degenerate guard: treat as ``true``.
        return [[space.true_feat]]
    return [literals]  # A single conjunction (AND) of the literals.


# Build the OR -> AND -> literal proposition graph for one edge guard
def build_prop_graph(
    disjuncts: list[list[torch.Tensor]],
    space: FeatureSpace,
) -> Data:

    node_features: list[torch.Tensor] = [space.or_feat]  # node 0 is the OR root.
    sources: list[int] = []
    targets: list[int] = []
    counter = 0
    for conjunction in disjuncts:
        counter += 1
        and_index = counter
        node_features.append(space.and_feat)
        sources.append(0)
        targets.append(and_index)
        for literal in conjunction:
            counter += 1
            node_features.append(literal)
            sources.append(and_index)
            targets.append(counter)

    x = torch.stack(node_features).to(space.device).float()
    edge_index = torch.tensor([sources, targets], dtype=torch.long, device=space.device)
    return Data(x=x, edge_index=edge_index)


def guard_prop_graph(guard: Guard, space: FeatureSpace) -> Data:
    return build_prop_graph(_guard_disjuncts(guard, space), space)


# Proposition graph whose single literal is a (differentiable) soft feature
def soft_prop_graph(soft_literal: torch.Tensor, space: FeatureSpace) -> Data:

    return build_prop_graph([[soft_literal]], space)


# Lift every edge into a node connected to its endpoints (paper's trick)
def _edge2node(
    node_feats: torch.Tensor,
    edge_index: torch.Tensor,
    edge_feats: torch.Tensor,
) -> Data:

    n_nodes = node_feats.size(0)
    nodes = torch.cat([node_feats, edge_feats], dim=0)
    sources: list[int] = []
    targets: list[int] = []
    new_node = n_nodes
    for column in range(edge_index.size(1)):
        start = int(edge_index[0, column])
        end = int(edge_index[1, column])
        sources.extend([start, new_node])
        targets.extend([new_node, end])
        new_node += 1
    lifted_index = torch.tensor(
        [sources, targets], dtype=torch.long, device=node_feats.device
    )
    return Data(x=nodes, edge_index=lifted_index)


# Embed each edge's proposition graph and lift the result for ``qm``
def _lift_edges(
    node_feats: torch.Tensor,
    edges: list[tuple[int, int, Data]],
    edge_embedder,
    space: FeatureSpace,
) -> Data:

    sources = [src for src, _, _ in edges]
    targets = [dst for _, dst, _ in edges]
    edge_feats = [edge_embedder(graph) for _, _, graph in edges]
    edge_index = torch.tensor([sources, targets], dtype=torch.long, device=space.device)
    edge_feats_tensor = torch.cat(edge_feats, dim=0)
    return _edge2node(node_feats, edge_index, edge_feats_tensor)


# Encode a constraint DFA as the lifted graph consumed by ``qm``
def encode_dfa(dfa: ConstraintDFA, space: FeatureSpace, edge_embedder) -> Data:

    states = list(dfa.states)  # sorted; the initial state is 0 -> row 0.
    state_row = {state: row for row, state in enumerate(states)}
    node_feats = torch.stack(
        [space.node_type_feat[dfa.node_types[state]] for state in states]
    )

    edges = [
        (state_row[src], state_row[dst], guard_prop_graph(guard, space))
        for src, dst, guard in dfa.edges
    ]
    return _lift_edges(node_feats, edges, edge_embedder, space)


# Encode a trace as a linear automaton, lifted for ``qm``
def encode_trace(
    trace,
    space: FeatureSpace,
    edge_embedder,
    *,
    soft_last: torch.Tensor | None = None,
) -> Data:

    activities = list(trace)
    if not activities:
        raise ValueError("Cannot encode an empty trace.")

    n_steps = len(activities)
    node_types: list[int] = []
    for position in range(n_steps + 1):
        if position == 0:
            node_types.append(INIT)
        elif position == n_steps:
            node_types.append(FINAL)
        else:
            node_types.append(COMMON)
    node_feats = torch.stack([space.node_type_feat[label] for label in node_types])

    edges: list[tuple[int, int, Data]] = []
    for step, activity in enumerate(activities):
        is_last = step == n_steps - 1
        if is_last and soft_last is not None:
            graph = soft_prop_graph(soft_last, space)
        else:
            graph = guard_prop_graph(Guard.activity(activity), space)
        edges.append((step, step + 1, graph))
    return _lift_edges(node_feats, edges, edge_embedder, space)
