# End-to-end oracle of rung 4 (kind *_seq) on real data

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from nspm.config import ExperimentConfig
from nspm.data.loader import read_log
from nspm.data.preparation import ActivityVocabulary, PrefixLog, TraceSplits, TraceUtils
from nspm.learning.models import (
    MarkingSequenceEncoder,
    build_model,
    load_checkpoint,
    save_checkpoint,
    symbolic_input,
)
from nspm.process.petrinet import PetriNet


def main() -> None:
    config = ExperimentConfig()

    log_path = next((ROOT / "datasets" / "Sepsis_Case").glob("*.xes"))
    traces = TraceUtils.extract_traces(read_log(log_path))
    splits = TraceSplits.from_traces(traces)
    vocabulary = ActivityVocabulary.from_traces(splits.train.values())
    petrinet = PetriNet.from_traces(splits.train)

    base = PrefixLog.from_traces(splits.train, vocabulary)
    sequence_log = base.with_markings(petrinet).with_marking_sequences(petrinet)
    static_log = base.with_markings(petrinet)

    places = len(petrinet.places)
    print(f"Sepsis: {len(vocabulary.activities)} activities, {places} places, {len(base)} prefixes")

    # ------------------------------------------------------------------ construction
    model = build_model(
        "gru_seq",
        len(vocabulary.tokens),
        len(vocabulary.activities),
        vocabulary.pad_id,
        config.model,
        marking_dim=places,
        adjacency=petrinet.adjacency_matrices,
    )
    assert isinstance(model.marking_encoder, MarkingSequenceEncoder)
    assert model.marking_encoder.expects_sequences
    print("build_model('gru_seq') -> encoder", type(model.marking_encoder).__name__)

    for kind, expected in (("gru_gnn", False), ("gru_marking", False), ("gru_seq", True)):
        other = build_model(kind, len(vocabulary.tokens), len(vocabulary.activities),
                            vocabulary.pad_id, config.model, places, petrinet.adjacency_matrices)
        assert other.marking_encoder.expects_sequences is expected, kind
    print("expects_sequences: marking=False, gnn=False, seq=True")

    # ---------------------------------------------------------------- forward/backward
    batch = next(iter(sequence_log.data_loader(32, shuffle=False)))
    chosen = symbolic_input(model, batch)
    assert chosen is batch.marking_sequences, "symbolic_input picked the wrong stream"
    print(f"symbolic_input -> sequences {tuple(chosen.shape)} (not the snapshot {tuple(batch.markings.shape)})")

    logits = model(batch.tokens, batch.lengths, chosen)
    assert logits.shape == (batch.targets.size(0), len(vocabulary.activities))
    loss = torch.nn.functional.cross_entropy(logits, batch.targets)
    loss.backward()

    encoder = model.marking_encoder
    gradients = {name: parameter.grad for name, parameter in encoder.named_parameters()}
    assert all(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients.values()), \
        "a piece of the encoder receives no gradient"
    assert encoder.a_pt_t.grad is None and encoder.a_tp_t.grad is None, "the adjacencies are not parameters"
    print(f"forward {tuple(logits.shape)}, loss {loss.item():.4f}, gradient on "
          f"{len(gradients)} tensors (graph + inner GRU), adjacencies without gradient")

    # ------------------------------------------------------------------- guards
    static_batch = next(iter(static_log.data_loader(8, shuffle=False)))
    try:
        symbolic_input(model, static_batch)
        raise AssertionError("no error on a log without sequences")
    except ValueError as error:
        print(f"data-family guard OK: {str(error).splitlines()[0][:60]}...")

    try:
        encoder(batch.marking_sequences, None)
        raise AssertionError("no error without lengths")
    except ValueError:
        print("lengths guard OK")

    # --------------------------------------------------------------- checkpoint
    path = ROOT / "runs" / "_check" / "seq_wiring.pt"
    save_checkpoint(path, model, "gru_seq", vocabulary, config,
                    logic_enabled=False, best_epoch=1, stopped_early=False)
    restored, restored_vocabulary, payload = load_checkpoint(path)
    assert isinstance(restored.marking_encoder, MarkingSequenceEncoder)
    assert payload["adjacency"] is not None and payload["marking_dim"] == places
    model.eval()
    with torch.no_grad():
        before = model(batch.tokens, batch.lengths, chosen)
        after = restored(batch.tokens, batch.lengths, symbolic_input(restored, batch))
    assert torch.allclose(before, after, atol=1e-6), "the reloaded model predicts differently"
    print(f"checkpoint round-trip OK (adjacencies saved, marking_dim={payload['marking_dim']}, "
          "identical predictions)")

    print("\nSEQ WIRING OK")


if __name__ == "__main__":
    main()
