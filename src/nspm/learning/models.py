# Sequence models for next-activity prediction, with checkpoint save/load

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch
from torch import nn, matmul
from torch.nn.utils.rnn import pack_padded_sequence

from ..config import ExperimentConfig, ModelConfig
from ..data.preparation import ActivityVocabulary
from ..process.petrinet import AdjacencyMatrix


ModelKind = Literal["gru", "gru_marking", "gru_gnn", "gru_seq", "gru_grnn",
                    "lstm", "lstm_marking", "lstm_gnn", "lstm_seq", "lstm_grnn"]

#: Encoder suffixes that need the Petri net adjacency matrices. Kept here so
#: that ``build_model`` and the scripts that build models test membership
#: against one list instead of each repeating its own tuple of suffixes.
GRAPH_SUFFIXES = ("_gnn", "_seq", "_grnn")


# Shared embedding and classification logic for recurrent encoders
class _NextActivityRecurrent(nn.Module):

    recurrent_type: type[nn.RNNBase]

    def __init__(self, vocabulary_size: int, number_of_classes: int, pad_id: int, config: ModelConfig, marking_encoder: nn.Module | None = None) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, config.embedding_dim, padding_idx=pad_id)
        # Torch's ``dropout`` acts BETWEEN layers, so with a single layer it
        # does nothing and prints a warning at every construction in exchange.
        # Dropout on the final state stays ``self.dropout``, which is always on.
        self.recurrent = self.recurrent_type(
            input_size=config.embedding_dim, hidden_size=config.hidden_dim,
            num_layers=config.recurrent_layers, batch_first=True,
            dropout=config.dropout if config.recurrent_layers > 1 else 0.0)
        self.dropout = nn.Dropout(config.dropout)
        self.marking_encoder = marking_encoder
        extra_dim = marking_encoder.output_dim if marking_encoder is not None else 0
        self.classifier = nn.Linear(config.hidden_dim + extra_dim, number_of_classes)

    # Encode each prefix and return unnormalised next-action scores
    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor, markings: torch.Tensor | None = None) -> torch.Tensor:

        embedded = self.embedding(tokens)
        # Packing ensures that padding cannot alter the final recurrent state.
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.recurrent(packed)
        if isinstance(hidden, tuple):  # LSTM returns (hidden_state, cell_state).
            hidden = hidden[0]
        final_hidden = hidden[-1]

        if self.marking_encoder is not None:
            if markings is None:
                raise ValueError("Markings cannot be None")
            final_hidden = torch.cat((final_hidden, self.marking_encoder(markings, lengths)), dim=1)

        return self.classifier(self.dropout(final_hidden))

    # Like :meth:`forward`, but it also returns the recurrent state
    def encode(self, tokens: torch.Tensor, lengths: torch.Tensor):

        if self.marking_encoder is not None:
            raise ValueError(
                "encode() e' definita solo sul tronco senza marking: un rollout "
                "dovrebbe rigiocare la rete di Petri a ogni passo generato, che "
                "e' il compito di _MarkingRollout, non di questa."
            )
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, hidden = self.recurrent(packed)
        state = hidden[0] if isinstance(hidden, tuple) else hidden
        return self.classifier(self.dropout(state[-1])), hidden

    # One recurrence step starting from a *soft* choice of activity
    def forward_from_state(self, activity_weights: torch.Tensor, hidden):

        offset = self.embedding.num_embeddings - self.classifier.out_features
        activity_embeddings = self.embedding.weight[offset:]
        embedded = (activity_weights @ activity_embeddings).unsqueeze(1)
        output, hidden = self.recurrent(embedded, hidden)
        return self.classifier(self.dropout(output[:, -1])), hidden

# Identity feature map: the raw marking is the feature vector
class FlatMarkingEncoder(nn.Module):

    expects_sequences = False

    def __init__(self, marking_dim: int) -> None:
        super().__init__()
        self.output_dim: int = marking_dim

    def forward(self, markings: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return markings.float()


# The net's graph and the message-passing pieces shared by rungs 4-6
class _PetriGraphEncoder(nn.Module):

    a_pt_t: torch.Tensor
    a_tp_t: torch.Tensor
    expects_sequences = False

    def __init__(self, a_pt: AdjacencyMatrix, a_tp: AdjacencyMatrix, hidden_dim: int) -> None:
        super().__init__()
        # Static graph structure: state that travels with .to(device) and
        # the checkpoint, but receives no gradient (the net is a fact).
        self.register_buffer("a_pt_t", torch.tensor(a_pt, dtype=torch.float32).T.contiguous())
        self.register_buffer("a_tp_t", torch.tensor(a_tp, dtype=torch.float32).T.contiguous())
        self.place_input = nn.Linear(1, hidden_dim)
        self.output_dim: int = hidden_dim

    # Token counts to node features: ``(..., P)`` -> ``(..., P, H)``
    def project(self, markings: torch.Tensor) -> torch.Tensor:

        return self.place_input(markings.float().unsqueeze(-1)).relu()

    # One P->T->P step with the given per-relation weights
    def hop(self, place_states: torch.Tensor, to_transition: nn.Linear, to_place: nn.Linear) -> torch.Tensor:

        transition_states = matmul(self.a_pt_t, place_states)
        transition_states = to_transition(transition_states).relu()
        place_states = matmul(self.a_tp_t, transition_states)
        return to_place(place_states).relu()

    # Max over the place axis, indexed from the right so a time axis is fine
    @staticmethod
    def readout(place_states: torch.Tensor) -> torch.Tensor:

        return place_states.max(dim=-2).values


# Two-hop message passing on the bipartite place/transition graph
class HeteroGraphEncoder(_PetriGraphEncoder):

    def __init__(self, a_pt: AdjacencyMatrix, a_tp: AdjacencyMatrix, hidden_dim: int) -> None:
        super().__init__(a_pt, a_tp, hidden_dim)
        self.place_to_transition = nn.Linear(hidden_dim, hidden_dim)
        self.transition_to_place = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, markings: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        place_states = self.hop(
            self.project(markings), self.place_to_transition, self.transition_to_place
        )
        return self.readout(place_states)

class MarkingSequenceEncoder(HeteroGraphEncoder):

    expects_sequences = True

    def __init__(self, a_pt: AdjacencyMatrix, a_tp: AdjacencyMatrix, hidden_dim: int) -> None:
        super().__init__(a_pt, a_tp, hidden_dim)
        self.recurrent = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, markings: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is None:
            raise ValueError("Marking sequences are padded, so the encoder needs the prefix \nlengths to keep padded steps out of the recurrence")
        # One graph readout per step: (B, L, P) -> (B, L, H). Time mixes only
        # in the recurrence below, never inside a step's message passing.
        graph_states = super().forward(markings)
        packed = pack_padded_sequence(graph_states, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.recurrent(packed)
        return hidden[-1]

# The two per-relation maps of one P->T->P hop
def _hop_weights(hidden_dim: int) -> tuple[nn.Linear, nn.Linear]:

    return nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)


# TACO's GRNN (Eq. 6): graph convolutions *inside* the GRU gates
class GraphRecurrentEncoder(_PetriGraphEncoder):

    expects_sequences = True

    def __init__(self, a_pt: AdjacencyMatrix, a_tp: AdjacencyMatrix, hidden_dim: int) -> None:
        super().__init__(a_pt, a_tp, hidden_dim)
        self.update_x_pt, self.update_x_tp = _hop_weights(hidden_dim)
        self.update_h_pt, self.update_h_tp = _hop_weights(hidden_dim)
        self.reset_x_pt, self.reset_x_tp = _hop_weights(hidden_dim)
        self.reset_h_pt, self.reset_h_tp = _hop_weights(hidden_dim)
        self.candidate_x_pt, self.candidate_x_tp = _hop_weights(hidden_dim)
        self.candidate_h_pt, self.candidate_h_tp = _hop_weights(hidden_dim)
        self.update_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.reset_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.candidate_bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, markings: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is None:
            raise ValueError("Marking sequences are padded, so the encoder needs the prefix \nlengths to keep padded steps out of the recurrence")

        projected = self.project(markings)  # (B, L, P, H)
        update_x = self.hop(projected, self.update_x_pt, self.update_x_tp)
        reset_x = self.hop(projected, self.reset_x_pt, self.reset_x_tp)
        candidate_x = self.hop(projected, self.candidate_x_pt, self.candidate_x_tp)

        batch_size, steps, places, _ = projected.shape
        hidden = projected.new_zeros(batch_size, places, self.output_dim)
        # A padded step carries an all-zero marking, which is a *legal* marking
        # (every place empty), so nothing in the values distinguishes it: only
        # the lengths do. Hence the explicit freeze instead of packing.
        alive_until = lengths.to(projected.device).view(-1, 1, 1)

        for step in range(steps):
            update = torch.sigmoid(
                update_x[:, step]
                + self.hop(hidden, self.update_h_pt, self.update_h_tp)
                + self.update_bias
            )
            reset = torch.sigmoid(
                reset_x[:, step]
                + self.hop(hidden, self.reset_h_pt, self.reset_h_tp)
                + self.reset_bias
            )
            candidate = torch.tanh(
                candidate_x[:, step]
                + reset * self.hop(hidden, self.candidate_h_pt, self.candidate_h_tp)
                + self.candidate_bias
            )
            stepped = update * hidden + (1.0 - update) * candidate
            hidden = torch.where(alive_until > step, stepped, hidden)

        return self.readout(hidden)


# GRU classifier used as the compact recurrent baseline
class NextActivityGRU(_NextActivityRecurrent):

    recurrent_type = nn.GRU


# LSTM classifier with an explicit memory cell for longer dependencies
class  NextActivityLSTM(_NextActivityRecurrent):

    recurrent_type = nn.LSTM


# Construct a sequence model from a validated symbolic name
def build_model(
    kind: ModelKind,
    vocabulary_size: int,
    number_of_classes: int,
    pad_id: int,
    config: ModelConfig,
    marking_dim: int = 0,
    adjacency: tuple[AdjacencyMatrix, AdjacencyMatrix] | None = None,
) -> nn.Module:

    model_types: dict[str, type[nn.Module]] = {
        "gru": NextActivityGRU,
        "gru_marking": NextActivityGRU,
        "gru_gnn": NextActivityGRU,
        "gru_seq": NextActivityGRU,
        "gru_grnn": NextActivityGRU,
        "lstm": NextActivityLSTM,
        "lstm_marking": NextActivityLSTM,
        "lstm_gnn": NextActivityLSTM,
        "lstm_seq": NextActivityLSTM,
        "lstm_grnn": NextActivityLSTM,
    }
    kind = kind.lower()
    if kind.endswith(("_marking",) + GRAPH_SUFFIXES) and marking_dim == 0:
        raise ValueError("Marking models need to know the number of places the network has")
    try:
        model_type = model_types[kind]
    except KeyError as error:
        raise ValueError(f"Unsupported model kind: {kind!r}") from error

    marking_encoder: nn.Module | None = None
    if kind.endswith("_marking"):
        marking_encoder = FlatMarkingEncoder(marking_dim)
    elif kind.endswith(GRAPH_SUFFIXES):
        if adjacency is None:
            raise ValueError("GNN models need the Petri net adjacency matrices")
        graph_encoders = {
            "_gnn": HeteroGraphEncoder,
            "_seq": MarkingSequenceEncoder,
            "_grnn": GraphRecurrentEncoder,
        }
        suffix = next(s for s in GRAPH_SUFFIXES if kind.endswith(s))
        marking_encoder = graph_encoders[suffix](adjacency[0], adjacency[1], config.hidden_dim)
    return model_type(vocabulary_size, number_of_classes, pad_id, config, marking_encoder)


# The symbolic stream the model's encoder expects, or None if it has none
def symbolic_input(model: nn.Module, batch) -> torch.Tensor | None:

    encoder = getattr(model, "marking_encoder", None)
    if encoder is None:
        return None
    if not encoder.expects_sequences:
        return batch.markings
    if batch.marking_sequences is None:
        raise ValueError(
            "This model reads marking sequences: build the log with "
            "PrefixLog.with_marking_sequences(petrinet)"
        )
    return batch.marking_sequences


# -------------------------------------------------------------------- checkpoints

# Save enough metadata to reconstruct the trained model
def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_kind: ModelKind,
    vocabulary: ActivityVocabulary,
    config: ExperimentConfig,
    logic_enabled: bool,
    best_epoch: int,
    stopped_early: bool,
) -> None:

    # Recover what build_model needs from the injected encoder: the flat
    # variant only knows its width, the GNN carries the graph in its buffers.
    marking_encoder = getattr(model, "marking_encoder", None)
    marking_dim = 0
    adjacency: tuple[AdjacencyMatrix, AdjacencyMatrix] | None = None
    if isinstance(marking_encoder, FlatMarkingEncoder):
        marking_dim = marking_encoder.output_dim
    elif isinstance(marking_encoder, _PetriGraphEncoder):
        a_pt = tuple(tuple(int(v) for v in row) for row in marking_encoder.a_pt_t.T.tolist())
        a_tp = tuple(tuple(int(v) for v in row) for row in marking_encoder.a_tp_t.T.tolist())
        adjacency = (a_pt, a_tp)
        marking_dim = len(a_pt)

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_kind": model_kind,
            "model_config": asdict(config.model),
            "activities": list(vocabulary.activities),
            "logic_enabled": logic_enabled,
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
            "experiment_config": config.to_dict(),
            "marking_dim": marking_dim,
            "adjacency": adjacency,
        },
        path,
    )


# Load a checkpoint created by :func:`save_checkpoint`
def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[nn.Module, ActivityVocabulary, dict[str, object]]:

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    vocabulary = ActivityVocabulary(tuple(payload["activities"]))
    model_config = ModelConfig(**payload["model_config"])
    model = build_model(
        payload["model_kind"],
        len(vocabulary.tokens),
        len(vocabulary.activities),
        vocabulary.pad_id,
        model_config,
        payload.get("marking_dim", 0),
        payload.get("adjacency"),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, vocabulary, payload
