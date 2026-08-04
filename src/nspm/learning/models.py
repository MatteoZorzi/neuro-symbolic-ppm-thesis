"""Sequence models for next-activity prediction, with checkpoint save/load."""

from __future__ import annotations

import math
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
                    "lstm", "lstm_marking", "lstm_gnn", "lstm_seq", "lstm_grnn",
                    "transformer"]

#: Encoder suffixes that need the Petri net adjacency matrices. Kept here so
#: callers (``build_model``, ``scripts/final_matrix.py``) test membership
#: against one list instead of each repeating its own tuple of suffixes.
GRAPH_SUFFIXES = ("_gnn", "_seq", "_grnn")


class _NextActivityRecurrent(nn.Module):
    """Shared embedding and classification logic for recurrent encoders.

    The marking variants differ only in the injected ``marking_encoder``
    (flat identity or graph message passing): it maps the marking to a
    feature vector concatenated to the final recurrent state.
    """

    recurrent_type: type[nn.RNNBase]

    def __init__(self, vocabulary_size: int, number_of_classes: int, pad_id: int, config: ModelConfig, marking_encoder: nn.Module | None = None) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, config.embedding_dim, padding_idx=pad_id)
        self.recurrent = self.recurrent_type(input_size=config.embedding_dim, hidden_size=config.hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.marking_encoder = marking_encoder
        extra_dim = marking_encoder.output_dim if marking_encoder is not None else 0
        self.classifier = nn.Linear(config.hidden_dim + extra_dim, number_of_classes)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor, markings: torch.Tensor | None = None) -> torch.Tensor:
        """Encode each prefix and return unnormalised next-action scores."""

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

class FlatMarkingEncoder(nn.Module):
    """Identity feature map: the raw marking is the feature vector.

    Ablation rung 2 ("state without structure"): same interface as
    :class:`HeteroGraphEncoder` so the two are interchangeable.
    """

    expects_sequences = False

    def __init__(self, marking_dim: int) -> None:
        super().__init__()
        self.output_dim: int = marking_dim

    def forward(self, markings: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return markings.float()


class _PetriGraphEncoder(nn.Module):
    """The net's graph and the message-passing pieces shared by rungs 4-6.

    Holds the adjacencies (buffers, no gradient), the projection from token
    counts to node features, one P->T->P hop parametrised on the two
    ``Linear`` layers it should use, and the readout. Subclasses differ only
    in *how many* hops they own and in where they put the recurrence.

    The parameter names of :class:`HeteroGraphEncoder` deliberately stay where
    they were -- ``state_dict`` keys follow attribute paths, so moving the
    buffers and ``place_input`` up here leaves checkpoints loadable.
    """

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

    def project(self, markings: torch.Tensor) -> torch.Tensor:
        """Token counts to node features: ``(..., P)`` -> ``(..., P, H)``."""

        return self.place_input(markings.float().unsqueeze(-1)).relu()

    def hop(self, place_states: torch.Tensor, to_transition: nn.Linear, to_place: nn.Linear) -> torch.Tensor:
        """One P->T->P step with the given per-relation weights.

        The adjacency products broadcast over every axis left of the place
        one, so the same code serves ``(B, P, H)`` and ``(B, L, P, H)``.
        """

        transition_states = matmul(self.a_pt_t, place_states)
        transition_states = to_transition(transition_states).relu()
        place_states = matmul(self.a_tp_t, transition_states)
        return to_place(place_states).relu()

    @staticmethod
    def readout(place_states: torch.Tensor) -> torch.Tensor:
        """Max over the place axis, indexed from the right so a time axis is fine."""

        return place_states.max(dim=-2).values


class HeteroGraphEncoder(_PetriGraphEncoder):
    """
    Two-hop message passing on the bipartite place/transition graph.
    Heterogeneous: the place->transition and transition->place relations
    have separate weights. Readout: max over places.

    Ablation rung 3 ("structure without time"): consumes one marking per
    prefix, ``(B, P)``. The readout indexes the place axis from the right
    so the same code also serves the sequential subclass, whose markings
    carry an extra time axis.
    """

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

def _hop_weights(hidden_dim: int) -> tuple[nn.Linear, nn.Linear]:
    """The two per-relation maps of one P->T->P hop."""

    return nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)


class GraphRecurrentEncoder(_PetriGraphEncoder):
    """TACO's GRNN (Eq. 6): graph convolutions *inside* the GRU gates.

    Ablation rung 6. Rung 5 reads each step out to a vector and only then
    recurs, so structure and time never mix inside the recurrence. Here the
    hidden state stays **per place**, ``(B, P, H)``, and every linear map of
    the GRU becomes a P->T->P hop::

        z_t = sigmoid( G_zx(x_t) + G_zh(h_{t-1}) + b_z )
        r_t = sigmoid( G_rx(x_t) + G_rh(h_{t-1}) + b_r )
        n_t = tanh(    G_nx(x_t) + r_t * G_nh(h_{t-1}) + b_n )
        h_t = z_t * h_{t-1} + (1 - z_t) * n_t

    Six hops, as in the paper: three gates times two arguments. The three
    input-side ones do not depend on the hidden state, so they run once over
    the whole padded sequence and only the three state-side ones sit inside
    the loop -- identical arithmetic, half the sequential work.

    Deliberately *unlike* TACO in one respect, and it is the one that keeps
    the ladder honest: the adjacency products stay unnormalised, exactly as
    in rungs 4-5, so the 6-5 delta isolates the recurrence and nothing else.
    The paper's ``D^-1 A`` stays a declared divergence (``MODELS.md`` 9.2).
    """

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


class NextActivityGRU(_NextActivityRecurrent):
    """GRU classifier used as the compact recurrent baseline."""

    recurrent_type = nn.GRU


class NextActivityLSTM(_NextActivityRecurrent):
    """LSTM classifier with an explicit memory cell for longer dependencies."""

    recurrent_type = nn.LSTM

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed positional information for variable-length activity prefixes."""

    def __init__(self, dimension: int, max_length: int) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / dimension)
        )
        encoding = torch.zeros(max_length, dimension)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        sequence_length = embedded.size(1)
        if sequence_length > self.encoding.size(1):
            raise ValueError(
                f"Prefix length {sequence_length} exceeds configured Transformer "
                f"limit {self.encoding.size(1)}."
            )
        return embedded + self.encoding[:, :sequence_length]


class NextActivityTransformer(nn.Module):
    """Causal Transformer encoder for next-activity classification."""

    def __init__(self, vocabulary_size: int, number_of_classes: int, pad_id: int, config: ModelConfig, marking_encoder: nn.Module | None = None) -> None:
        super().__init__()
        if marking_encoder is not None:
            raise ValueError("The Transformer variant does not support marking encoders")
        self.pad_id = pad_id
        self.embedding_scale = math.sqrt(config.embedding_dim)
        self.embedding = nn.Embedding(
            vocabulary_size, config.embedding_dim, padding_idx=pad_id
        )
        self.position = SinusoidalPositionalEncoding(
            config.embedding_dim, config.transformer_max_length
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.transformer_layers
        )
        self.normalization = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.embedding_dim, number_of_classes)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor, markings=None) -> torch.Tensor:
        embedded = self.embedding(tokens) * self.embedding_scale
        embedded = self.position(embedded)
        sequence_length = tokens.size(1)
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=tokens.device,
            ),
            diagonal=1,
        )
        encoded = self.encoder(
            embedded,
            mask=causal_mask,
            src_key_padding_mask=tokens.eq(self.pad_id),
        )
        row_ids = torch.arange(tokens.size(0), device=tokens.device)
        final_states = encoded[row_ids, lengths - 1]
        final_states = self.normalization(final_states)
        return self.classifier(self.dropout(final_states))

def build_model(
    kind: ModelKind,
    vocabulary_size: int,
    number_of_classes: int,
    pad_id: int,
    config: ModelConfig,
    marking_dim: int = 0,
    adjacency: tuple[AdjacencyMatrix, AdjacencyMatrix] | None = None,
) -> nn.Module:
    """Construct a sequence model from a validated symbolic name."""

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
        "transformer": NextActivityTransformer,
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


def symbolic_input(model: nn.Module, batch) -> torch.Tensor | None:
    """The symbolic stream the model's encoder expects, or None if it has none.

    Both streams travel in the batch, so the choice belongs to the encoder
    that declares its need, not to the call site: a log carrying marking
    sequences also carries the static snapshots, and picking "whichever is
    present" would silently feed histories to the static variants.
    """

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
    """Save enough metadata to reconstruct the trained model."""

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


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[nn.Module, ActivityVocabulary, dict[str, object]]:
    """Load a checkpoint created by :func:`save_checkpoint`."""

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
