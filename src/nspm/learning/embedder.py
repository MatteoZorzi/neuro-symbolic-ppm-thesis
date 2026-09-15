# Hierarchical DFA embedder (edge embedder ``qe`` + meta embedder ``qm``)

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from ..process.graph_encoding import FeatureSpace, encode_dfa, encode_trace
from ..process.ltl_constraints import ConstraintDFA


def _neighbour_list(num_nodes: int, edge_index: torch.Tensor) -> list[list[int]]:
    neighbours: list[list[int]] = [[] for _ in range(num_nodes)]
    for column in range(edge_index.size(1)):
        source = int(edge_index[0, column])
        target = int(edge_index[1, column])
        neighbours[source].append(target)
    return neighbours


# Random walks from ``start_index`` (the initial automaton node)
def _sample_paths(
    neighbours: list[list[int]],
    *,
    start_index: int,
    num_paths: int,
    max_length: int,
    rng: random.Random,
) -> list[list[int]]:

    paths: list[list[int]] = []
    for _ in range(num_paths):
        current = start_index
        walk = [current]
        for _ in range(max_length - 1):
            if not neighbours[current]:
                break
            current = rng.choice(neighbours[current])
            walk.append(current)
        paths.append(walk)
    return paths


# ``qe``: embed an edge's proposition graph into a node-level vector
class EdgeEmbedder(torch.nn.Module):

    def __init__(self, prop_dim: int = 50, hidden_dim: int = 200, node_dim: int = 100,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = GCNConv(prop_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, node_dim)
        self.dropout = dropout

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        out = F.dropout(F.relu(self.conv1(x, edge_index)), p=self.dropout, training=self.training)
        out = F.dropout(F.relu(self.conv2(out, edge_index)), p=self.dropout, training=self.training)
        return torch.mean(out, dim=0, keepdim=True)  # [1, node_dim]


# ``qm``: embed the edge-lifted automaton graph via random-walk aggregation
class MetaEmbedder(torch.nn.Module):

    def __init__(self, node_dim: int = 100, hidden_dim: int = 256, out_dim: int = 200,
                 dropout: float = 0.0, num_paths: int = 10, max_length: int = 5,
                 aggregation: str = "mean", num_layers: int = 2,
                 normalize: bool = False) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        dims = [node_dim]
        for _ in range(num_layers - 1):
            dims.append(hidden_dim)
        dims.append(out_dim)
        self.convs = torch.nn.ModuleList(
            GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)
        )
        self.dropout = dropout
        self.num_paths = num_paths
        self.max_length = max_length
        self.normalize = normalize
        if aggregation not in {"mean", "random_walk"}:
            raise ValueError("aggregation must be 'mean' or 'random_walk'.")
        # 'random_walk' reproduces the paper's initial-node-seeded aggregation,
        # which preserves trace order (the signal separating satisfying from
        # unsatisfying traces); a generous ``max_length`` ensures the walk also
        # reaches the soft predicted edge at the end of a trace so the logic
        # loss keeps a non-vanishing gradient. 'mean' is the order-agnostic
        # global-pooling variant from the original ``Node_only``. Shallow
        # ``num_layers`` avoids over-smoothing these very small graphs.
        self.aggregation = aggregation

    def forward(self, data, *, start_index: int = 0, rng: random.Random | None = None) -> torch.Tensor:
        out, edge_index = data.x, data.edge_index
        for conv in self.convs:
            out = F.dropout(F.relu(conv(out, edge_index)), p=self.dropout, training=self.training)

        if self.aggregation == "mean":
            pooled = torch.mean(out, dim=0, keepdim=True)
        else:
            rng = rng or random
            neighbours = _neighbour_list(data.x.size(0), edge_index)
            paths = _sample_paths(
                neighbours,
                start_index=start_index,
                num_paths=self.num_paths,
                max_length=self.max_length,
                rng=rng,
            )
            visited = [out[node] for path in paths for node in path]
            pooled = torch.mean(torch.stack(visited), dim=0, keepdim=True)
        if self.normalize:
            pooled = pooled / torch.clamp(torch.norm(pooled), min=1e-8)
        return pooled


# ``q = (qe, qm)``: shared embedder for constraint DFAs and traces
class HierarchicalEmbedder(torch.nn.Module):

    def __init__(self, prop_dim: int = 50, node_dim: int = 100, out_dim: int = 200,
                 edge_hidden: int = 200, meta_hidden: int = 256, dropout: float = 0.0,
                 aggregation: str = "random_walk", num_paths: int = 8,
                 max_length: int = 32, meta_layers: int = 2, normalize: bool = False) -> None:
        super().__init__()
        self.edge_embedder = EdgeEmbedder(prop_dim, edge_hidden, node_dim, dropout)
        self.meta_embedder = MetaEmbedder(node_dim, meta_hidden, out_dim, dropout,
                                          num_paths=num_paths, max_length=max_length,
                                          aggregation=aggregation, num_layers=meta_layers,
                                          normalize=normalize)
        self.prop_dim = prop_dim
        self.node_dim = node_dim
        self.out_dim = out_dim

    def embed_dfa(self, dfa: ConstraintDFA, space: FeatureSpace,
                  rng: random.Random | None = None) -> torch.Tensor:
        data = encode_dfa(dfa, space, self.edge_embedder)
        return self.meta_embedder(data, start_index=0, rng=rng)

    def embed_trace(self, trace, space: FeatureSpace, *, soft_last: torch.Tensor | None = None,
                    rng: random.Random | None = None) -> torch.Tensor:
        data = encode_trace(trace, space, self.edge_embedder, soft_last=soft_last)
        return self.meta_embedder(data, start_index=0, rng=rng)
