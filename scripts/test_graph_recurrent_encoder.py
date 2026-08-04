"""Gradino 6 -- la GRNN, e la garanzia che il refactor non abbia mosso i gradini 4-5.

Il test che conta e' :func:`test_padding_non_cambia_output`. La GRNN srotola il
tempo a mano invece di usare ``pack_padded_sequence``, quindi il masking e'
scritto a mano ed e' il punto dove si sbaglia; e un passo di padding porta un
marking tutto a zero, che e' un marking **legale** (ogni posto vuoto), percio'
nessun controllo sui valori puo' accorgersi dell'errore. Solo le lunghezze
distinguono i due casi.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nspm.learning.models import (  # noqa: E402
    GraphRecurrentEncoder,
    HeteroGraphEncoder,
    MarkingSequenceEncoder,
)

PLACES, TRANSITIONS, HIDDEN = 5, 4, 8


@pytest.fixture
def adjacency():
    """Una reticella giocattolo: P->T e T->P binarie, non simmetriche."""

    generator = torch.Generator().manual_seed(0)
    a_pt = (torch.rand(PLACES, TRANSITIONS, generator=generator) > 0.5).int()
    a_tp = (torch.rand(TRANSITIONS, PLACES, generator=generator) > 0.5).int()
    return (
        tuple(tuple(int(v) for v in row) for row in a_pt),
        tuple(tuple(int(v) for v in row) for row in a_tp),
    )


@pytest.fixture
def encoder(adjacency):
    torch.manual_seed(0)
    return GraphRecurrentEncoder(*adjacency, HIDDEN).eval()


def test_shape(encoder):
    """``(B, L, P)`` entra, ``(B, H)`` esce: stesso contratto degli altri encoder."""

    markings = torch.rand(3, 7, PLACES)
    lengths = torch.tensor([7, 4, 1])
    assert encoder(markings, lengths).shape == (3, HIDDEN)
    assert encoder.output_dim == HIDDEN


def test_padding_non_cambia_output(encoder):
    """Stessi prefissi, piu' padding a destra -> stesso identico output.

    Se il congelamento dello stato non funziona, i passi di padding continuano
    ad aggiornare ``h`` e le due chiamate divergono.
    """

    torch.manual_seed(1)
    short = torch.rand(2, 3, PLACES)
    lengths = torch.tensor([3, 2])

    padded = torch.zeros(2, 6, PLACES)
    padded[:, :3] = short
    # Rumore oltre la lunghezza dichiarata: se venisse letto, si vedrebbe.
    padded[0, 3:] = torch.rand(3, PLACES)
    padded[1, 2:] = torch.rand(4, PLACES)

    with torch.no_grad():
        assert torch.allclose(encoder(short, lengths), encoder(padded, lengths), atol=1e-6)


def test_un_solo_passo_dipende_solo_dal_primo_marking(encoder):
    """Con ``lengths = 1`` l'uscita e' funzione di ``x_0`` e dello stato iniziale."""

    torch.manual_seed(2)
    markings = torch.rand(2, 4, PLACES)
    lengths = torch.tensor([1, 1])

    altered = markings.clone()
    altered[:, 1:] = torch.rand(2, 3, PLACES)

    with torch.no_grad():
        assert torch.allclose(encoder(markings, lengths), encoder(altered, lengths), atol=1e-6)


def test_lunghezze_obbligatorie(encoder):
    with pytest.raises(ValueError, match="lengths"):
        encoder(torch.rand(2, 3, PLACES), None)


def test_adiacenze_fuori_dai_parametri(encoder):
    """Il grafo e' un fatto, non un peso: buffer, non ``parameters()``."""

    parameter_ids = {id(p) for p in encoder.parameters()}
    assert id(encoder.a_pt_t) not in parameter_ids
    assert id(encoder.a_tp_t) not in parameter_ids
    buffers = dict(encoder.named_buffers())
    assert set(buffers) == {"a_pt_t", "a_tp_t"}
    assert not any(b.requires_grad for b in buffers.values())


def test_sei_convoluzioni(encoder):
    """Tre gate x due argomenti, come TACO Eq. 6 -- niente pesi condivisi."""

    hop_weights = [n for n, _ in encoder.named_parameters() if n.endswith(("_pt.weight", "_tp.weight"))]
    assert len(hop_weights) == 12  # sei hop, due relazioni ciascuno


def test_gradini_4_e_5_invariati_dopo_il_refactor(adjacency):
    """La suddivisione in ``project``/``hop``/``readout`` non cambia l'aritmetica.

    Ricalcola a mano la catena originale di :class:`HeteroGraphEncoder` e la
    confronta con il ``forward`` attuale: e' la verifica di regressione che il
    piano chiedeva per il gradino 4, estesa al 5 perche' ne condivide il codice.
    """

    torch.manual_seed(3)
    static = HeteroGraphEncoder(*adjacency, HIDDEN).eval()
    markings = torch.rand(3, PLACES)

    with torch.no_grad():
        place_states = static.place_input(markings.float().unsqueeze(-1)).relu()
        transition_states = torch.matmul(static.a_pt_t, place_states)
        transition_states = static.place_to_transition(transition_states).relu()
        place_states = torch.matmul(static.a_tp_t, transition_states)
        place_states = static.transition_to_place(place_states).relu()
        expected = place_states.max(dim=-2).values

        assert torch.allclose(static(markings), expected, atol=1e-7)

    torch.manual_seed(3)
    sequential = MarkingSequenceEncoder(*adjacency, HIDDEN).eval()
    sequences = torch.rand(3, 6, PLACES)
    lengths = torch.tensor([6, 3, 1])
    with torch.no_grad():
        assert sequential(sequences, lengths).shape == (3, HIDDEN)
