"""Scratch check for process/petrinet.py against the probe's known-good numbers.

Expected on Sepsis train split (from scripts/_probe_petrinet_feasibility.py):
26 places, 34 transitions (19 silent), 80 arcs.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nspm.data.loader import read_log
from nspm.data.preparation import TraceSplits, TraceUtils
from nspm.process.petrinet import PetriNet

log_path = next((ROOT / "datasets" / "Sepsis_Case").glob("*.xes"))
events = read_log(log_path)
traces = TraceUtils.extract_traces(events)
splits = TraceSplits.from_traces(traces)

net = PetriNet.from_traces(splits.train)
silent = [t for t in net.transitions if t.label is None]
print(f"places: {len(net.places)} | transitions: {len(net.transitions)} "
      f"(silent: {len(silent)}) | arcs: {len(net.network.arcs)}")
print(f"first 5 transition labels: {[t.label for t in net.transitions[:5]]}")
print(f"init marking: {net.init_marking} | final marking: {net.final_marking}")
print(f"place_index size: {len(net.place_index)}")


def occupied(marking_vector):
    """Human-readable view: only the places holding tokens."""
    return {net.places[i].name: v for i, v in enumerate(marking_vector) if v}


# --- replay checks -----------------------------------------------------------
# 1. Empty prefix. NOT the initial marking: WALK_THROUGH_HIDDEN_TRANS also fires
#    already-enabled silent transitions at the end of the replay, so the initial
#    tau-split runs and the marking shows the ready parallel branches. Accepted
#    semantics: every prefix (empty included) goes through the same procedure.
#    What we require instead: deterministic and well-formed.
empty = net.prefix_marking([])
assert empty == net.prefix_marking([]), "empty prefix marking not deterministic"
assert len(empty) == len(net.places) and sum(empty) >= 1
print(f"[replay] empty prefix (tau-closure of initial marking): {occupied(empty)}")

# 2. Growing prefixes of a real train trace: marking defined at every length,
#    total token count always >= 1 (the net always has state).
example_trace = next(iter(splits.train.values()))
for k in (1, 3, len(example_trace)):
    vector = net.prefix_marking(list(example_trace[:k]))
    assert sum(vector) >= 1, f"no tokens at prefix length {k}"
    print(f"[replay] prefix len {k:>2} -> {occupied(vector)}")

# 3. Absurd prefix (illegal order, repeated end activities): must not crash,
#    must still return a well-formed vector.
absurd = net.prefix_marking(["Release A", "Release A", "IV Antibiotics"])
assert len(absurd) == len(net.places)
print(f"[replay] absurd prefix survives: {occupied(absurd)}")

print("ALL REPLAY CHECKS OK")

# --- full chain: PrefixLog -> with_markings -> data_loader -> batch ----------
import time

from nspm.data.preparation import ActivityVocabulary, PrefixLog

vocabulary = ActivityVocabulary.from_traces(splits.train.values())
log = PrefixLog.from_traces(splits.train, vocabulary)

t0 = time.perf_counter()
marked = log.with_markings(net)
elapsed = time.perf_counter() - t0
print(f"[chain] with_markings on {len(log)} train prefixes: {elapsed:.1f}s "
      f"({elapsed / len(log) * 1000:.2f} ms/prefix)")

# Same number of examples, every marking present and of the right size.
assert len(marked) == len(log)
assert all(e.marking is not None and len(e.marking) == len(net.places) for e in marked)
# Everything else untouched: same cases, same prefixes, same targets.
assert [(e.case_id, e.token_ids, e.target_id) for e in marked] == \
       [(e.case_id, e.token_ids, e.target_id) for e in log]

# Batch with markings: shape (batch, n_places), dtype float32.
batch = next(iter(marked.data_loader(batch_size=32, shuffle=False)))
assert batch.markings is not None
assert tuple(batch.markings.shape) == (32, len(net.places)), batch.markings.shape
assert batch.markings.dtype.is_floating_point
print(f"[chain] batch.markings: shape {tuple(batch.markings.shape)}, "
      f"dtype {batch.markings.dtype}")

# Old path untouched: a log without markings still yields markings=None.
plain_batch = next(iter(log.data_loader(batch_size=32, shuffle=False)))
assert plain_batch.markings is None
print("[chain] plain log still gives markings=None: OK")

print("ALL CHAIN CHECKS OK")

# --- model: gru_marking forward, fail-fast, checkpoint round-trip ------------
import tempfile

import torch

from nspm.config import ExperimentConfig
from nspm.learning.models import build_model, load_checkpoint, save_checkpoint

config = ExperimentConfig()
n_places = len(net.places)
model = build_model(
    "gru_marking",
    len(vocabulary.tokens),
    len(vocabulary.activities),
    vocabulary.pad_id,
    config.model,
    marking_dim=n_places,
)
model.eval()

# 1. Forward with markings: logits (batch, n_classes).
with torch.no_grad():
    logits = model(batch.tokens, batch.lengths, batch.markings)
assert tuple(logits.shape) == (32, len(vocabulary.activities)), logits.shape
print(f"[model] gru_marking forward: logits {tuple(logits.shape)}")

# 2. Fail fast: a marking model without markings must raise.
try:
    model(batch.tokens, batch.lengths)
    raise AssertionError("gru_marking accepted markings=None")
except ValueError:
    print("[model] gru_marking without markings raises ValueError: OK")

# 3. Baseline untouched: plain gru still works with markings=None.
baseline = build_model(
    "gru", len(vocabulary.tokens), len(vocabulary.activities),
    vocabulary.pad_id, config.model,
)
baseline.eval()
with torch.no_grad():
    baseline_logits = baseline(batch.tokens, batch.lengths)
assert tuple(baseline_logits.shape) == (32, len(vocabulary.activities))
print("[model] plain gru with markings=None: OK")

# 4. build_model refuses a marking kind without marking_dim.
try:
    build_model("gru_marking", len(vocabulary.tokens),
                len(vocabulary.activities), vocabulary.pad_id, config.model)
    raise AssertionError("build_model accepted gru_marking with marking_dim=0")
except ValueError:
    print("[model] build_model gru_marking without marking_dim raises: OK")

# 5. Checkpoint round-trip: same architecture, same outputs.
with tempfile.TemporaryDirectory() as tmp:
    ckpt = Path(tmp) / "gru_marking.pt"
    save_checkpoint(ckpt, model, "gru_marking", vocabulary, config,
                    logic_enabled=False, best_epoch=1, stopped_early=False)
    restored, restored_vocab, payload = load_checkpoint(ckpt)
    assert payload["marking_dim"] == n_places
    with torch.no_grad():
        restored_logits = restored(batch.tokens, batch.lengths, batch.markings)
    assert torch.allclose(logits, restored_logits), "round-trip changed outputs"
print("[model] checkpoint round-trip (save -> load -> same logits): OK")

print("ALL MODEL CHECKS OK")

# --- adjacency matrices (Step 2): bipartite P->T / T->P ----------------------
# Probe ground truth: 80 arcs total, split 40 P->T and 40 T->P.
a_pt, a_tp = net.adjacency_matrices
assert len(a_pt) == len(net.places) and len(a_pt[0]) == len(net.transitions)
assert len(a_tp) == len(net.transitions) and len(a_tp[0]) == len(net.places)
pt_arcs = sum(sum(row) for row in a_pt)
tp_arcs = sum(sum(row) for row in a_tp)
assert pt_arcs == 40, pt_arcs
assert tp_arcs == 40, tp_arcs
assert pt_arcs + tp_arcs == len(net.network.arcs)
# Deterministic: two accesses build identical matrices.
assert (a_pt, a_tp) == net.adjacency_matrices
# Not each other's transpose (P->T arcs are different arcs than T->P).
transposed = tuple(tuple(a_pt[i][j] for i in range(len(a_pt))) for j in range(len(a_pt[0])))
assert transposed != a_tp, "a_tp equals a_pt transposed: input and output arcs collapsed"
print(f"[adjacency] a_pt {len(a_pt)}x{len(a_pt[0])} sum {pt_arcs} | "
      f"a_tp {len(a_tp)}x{len(a_tp[0])} sum {tp_arcs}")

print("ALL ADJACENCY CHECKS OK")