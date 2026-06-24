"""Sequence models for next-activity prediction."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from ..config import ModelConfig


ModelKind = Literal["gru", "lstm", "transformer"]


class _NextActivityRecurrent(nn.Module):
    """Shared embedding and classification logic for recurrent encoders."""

    recurrent_type: type[nn.RNNBase]

    def __init__(
        self,
        vocabulary_size: int,
        number_of_classes: int,
        pad_id: int,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocabulary_size, config.embedding_dim, padding_idx=pad_id
        )
        self.recurrent = self.recurrent_type(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_dim, number_of_classes)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode each prefix and return unnormalised next-action scores."""

        embedded = self.embedding(tokens)
        # Packing ensures that padding cannot alter the final recurrent state.
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.recurrent(packed)
        if isinstance(hidden, tuple):  # LSTM returns (hidden_state, cell_state).
            hidden = hidden[0]
        final_hidden = hidden[-1]
        return self.classifier(self.dropout(final_hidden))


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

    def __init__(
        self,
        vocabulary_size: int,
        number_of_classes: int,
        pad_id: int,
        config: ModelConfig,
    ) -> None:
        super().__init__()
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

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
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
) -> nn.Module:
    """Construct a sequence model from a validated symbolic name."""

    model_types: dict[str, type[nn.Module]] = {
        "gru": NextActivityGRU,
        "lstm": NextActivityLSTM,
        "transformer": NextActivityTransformer,
    }
    try:
        model_type = model_types[kind.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported model kind: {kind!r}") from error
    return model_type(vocabulary_size, number_of_classes, pad_id, config)
