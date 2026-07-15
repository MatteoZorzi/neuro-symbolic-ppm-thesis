"""Check rapido: init di HeteroGraphEncoder (buffer, shape, parametri)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nspm.data.loader import read_log
from nspm.data.preparation import TraceSplits, TraceUtils
from nspm.learning.models import HeteroGraphEncoder
from nspm.process.petrinet import PetriNet

log_path = next((ROOT / "datasets" / "Sepsis_Case").glob("*.xes"))
frame = read_log(log_path)
traces = TraceUtils.extract_traces(frame)
splits = TraceSplits.from_traces(traces)
net = PetriNet.from_traces(splits.train)
a_pt, a_tp = net.adjacency_matrices

enc = HeteroGraphEncoder(a_pt, a_tp, hidden_dim=16)
print("a_pt_t shape:", tuple(enc.a_pt_t.shape), "(atteso: (34, 26), trasposta di 26x34)")
print("a_tp_t shape:", tuple(enc.a_tp_t.shape), "(atteso: (26, 34))")
print("buffer nello state_dict:", [k for k in enc.state_dict() if k.startswith("a_")])
print("parametri trainabili:", sorted({n.split(".")[0] for n, _ in enc.named_parameters()}))
print("INIT OK")

# --- forward smoke test -----------------------------------------------------
import torch

batch = torch.randint(0, 3, (5, len(net.places)))  # 5 marking finti, valori 0-2
out = enc(batch)
assert out.shape == (5, 16), f"atteso (5, 16), ottenuto {tuple(out.shape)}"
print("forward:", tuple(batch.shape), "->", tuple(out.shape))

# Il gradiente deve arrivare ai Linear, non ai buffer (il grafo e' un fatto).
out.sum().backward()
for name, param in enc.named_parameters():
    assert param.grad is not None, f"nessun gradiente su {name}"
assert enc.a_pt_t.grad is None and enc.a_tp_t.grad is None
print("gradienti: tutti i Linear OK, buffer senza gradiente")

# Proprieta' del max readout: marking tutto zero != marking attivo.
zero = enc(torch.zeros(1, len(net.places), dtype=torch.long))
active = enc(batch[:1])
print("readout distinti (zero vs attivo):", not torch.allclose(zero, active))
print("FORWARD OK")

# --- wiring: build_model sui tre gradini + roundtrip checkpoint --------------
import tempfile

from nspm.config import ExperimentConfig
from nspm.data.preparation import ActivityVocabulary
from nspm.learning.models import build_model, load_checkpoint, save_checkpoint

vocab = ActivityVocabulary.from_traces(splits.train.values())
config = ExperimentConfig()
n_places = len(net.places)
adjacency = (a_pt, a_tp)

tokens = torch.randint(1, len(vocab.tokens), (4, 7))
lengths = torch.tensor([7, 5, 3, 2])
markings = torch.randint(0, 3, (4, n_places))

variants = {
    "gru": {},
    "gru_marking": {"marking_dim": n_places},
    "gru_gnn": {"marking_dim": n_places, "adjacency": adjacency},
}
for kind, kwargs in variants.items():
    model = build_model(kind, len(vocab.tokens), len(vocab.activities), vocab.pad_id, config.model, **kwargs)
    model.eval()
    logits = model(tokens, lengths, markings if kwargs else None)
    encoder_name = type(model.marking_encoder).__name__ if model.marking_encoder is not None else "None"
    assert logits.shape == (4, len(vocab.activities))
    print(f"{kind:13s} logits {tuple(logits.shape)}  encoder: {encoder_name}")

# Il checkpoint della variante gnn deve ricostruire grafo e pesi da solo.
ckpt = Path(tempfile.gettempdir()) / "_gnn_roundtrip.pt"
model = build_model("gru_gnn", len(vocab.tokens), len(vocab.activities), vocab.pad_id, config.model, n_places, adjacency)
model.eval()
save_checkpoint(ckpt, model, "gru_gnn", vocab, config, logic_enabled=False, best_epoch=1, stopped_early=False)
loaded, _, payload = load_checkpoint(ckpt)
assert payload["adjacency"] == adjacency, "le adiacenze nel payload non coincidono"
assert torch.allclose(model(tokens, lengths, markings), loaded(tokens, lengths, markings), atol=1e-6)
print("checkpoint roundtrip: adiacenze nel payload e predizioni identiche")
print("WIRING OK")
