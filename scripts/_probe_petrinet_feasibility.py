"""Feasibility probe: Petri net + marking + heterogeneous GNN (TACO-style).

Answers, with the tools actually installed in this environment:
1. Can pm4py discover a Petri net (inductive miner) from our logs?      -> sizes
2. Can we extract places / transitions / arcs / initial+final marking?  -> objects
3. Can we build the adjacency structures (bipartite hetero + place graph)?
4. Given a prefix, can we compute its marking (token replay w/ taus)?   -> vector
5. Is torch_geometric ready for heterogeneous graphs (HeteroData/HeteroConv)?
6. How fast is marking computation per prefix (rough throughput)?
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

# ---------------------------------------------------------------- 1-2. mine net
import pm4py
from nspm.data.preparation import TraceSplits, TraceUtils
from nspm.data.loader import read_xes

log_path = next((ROOT / "datasets" / "Sepsis_Case").glob("*.xes"))
events = read_xes(log_path)
traces = TraceUtils.extract_traces(events)
splits = TraceSplits.from_traces(traces)
print(f"log: {log_path.name} | cases: {len(traces)} | train cases: {len(splits.train)}")

t0 = time.perf_counter()
event_log = pm4py.convert_to_event_log(events[events["case:concept:name"].isin(splits.train)])
net, im, fm = pm4py.discover_petri_net_inductive(event_log, noise_threshold=0.2)
t_mine = time.perf_counter() - t0

silent = [t for t in net.transitions if t.label is None]
labeled = [t for t in net.transitions if t.label is not None]
print(f"[1] inductive miner (noise 0.2) in {t_mine:.1f}s: "
      f"{len(net.places)} places, {len(net.transitions)} transitions "
      f"({len(silent)} silent), {len(net.arcs)} arcs")
print(f"[2] initial marking: {im} | final marking: {fm}")

# --------------------------------------------------- 3. adjacency structures
places = sorted(net.places, key=lambda p: p.name)
transitions = sorted(net.transitions, key=lambda t: (t.label is None, str(t)))
p_index = {p: i for i, p in enumerate(places)}
t_index = {t: i for i, t in enumerate(transitions)}

# Heterogeneous bipartite: place -> transition and transition -> place edges.
pt_edges, tp_edges = [], []
for arc in net.arcs:
    if arc.source in p_index:                       # place -> transition
        pt_edges.append((p_index[arc.source], t_index[arc.target]))
    else:                                            # transition -> place
        tp_edges.append((t_index[arc.source], p_index[arc.target]))
print(f"[3] hetero edges: place->trans {len(pt_edges)}, trans->place {len(tp_edges)}")

# TACO place-graph reduction: place -> place through any transition.
A = np.zeros((len(places), len(places)), dtype=np.int8)
for t in net.transitions:
    sources = [a.source for a in t.in_arcs]
    targets = [a.target for a in t.out_arcs]
    for s in sources:
        for d in targets:
            A[p_index[s], p_index[d]] = 1
print(f"[3] place-graph adjacency: {A.shape}, {int(A.sum())} edges")

# ------------------------------------------- 4. prefix -> marking (token replay)
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.objects.log.obj import Event, EventLog, Trace

def prefix_marking(prefix, net, im):
    """Reached marking after replaying `prefix` (token replay handles taus)."""
    trace = Trace()
    for activity in prefix:
        trace.append(Event({"concept:name": activity}))
    log = EventLog([trace])
    result = token_replay.apply(
        log, net, im, fm,
        parameters={
            token_replay.Variants.TOKEN_REPLAY.value.Parameters.STOP_IMMEDIATELY_UNFIT: False,
            token_replay.Variants.TOKEN_REPLAY.value.Parameters.WALK_THROUGH_HIDDEN_TRANS: True,
            token_replay.Variants.TOKEN_REPLAY.value.Parameters.SHOW_PROGRESS_BAR: False,
        },
    )[0]
    marking = result["reached_marking"]
    vector = np.zeros(len(places), dtype=np.float32)
    for place, tokens in marking.items():
        vector[p_index[place]] = tokens
    return vector, result["trace_is_fit"], result["missing_tokens"]

example_trace = next(iter(splits.train.values()))
for k in (1, 3, len(example_trace)):
    prefix = list(example_trace[:k])
    vector, fit, missing = prefix_marking(prefix, net, im)
    on = {places[i].name: int(v) for i, v in enumerate(vector) if v > 0}
    print(f"[4] prefix len {k:>2} -> marking {on} (fit={fit}, missing={missing})")

# ----------------------------------------------------- 5. PyG hetero readiness
try:
    import torch
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import GCNConv, HeteroConv, SAGEConv  # noqa: F401

    data = HeteroData()
    data["place"].x = torch.eye(len(places))
    data["transition"].x = torch.eye(len(transitions))
    data["place", "to", "transition"].edge_index = torch.tensor(
        list(zip(*pt_edges)), dtype=torch.long)
    data["transition", "to", "place"].edge_index = torch.tensor(
        list(zip(*tp_edges)), dtype=torch.long)
    conv = HeteroConv({
        ("place", "to", "transition"): SAGEConv((len(places), len(transitions)), 16),
        ("transition", "to", "place"): SAGEConv((len(transitions), len(places)), 16),
    })
    out = conv(data.x_dict, data.edge_index_dict)
    print(f"[5] PyG HeteroData + HeteroConv forward OK: "
          f"{ {k: tuple(v.shape) for k, v in out.items()} }")
except Exception as error:  # noqa: BLE001
    print(f"[5] PyG hetero FAILED: {error}")

# ------------------------------------------------------------- 6. throughput
sample = list(splits.train.values())[:30]
t0 = time.perf_counter()
n_prefixes = 0
for trace in sample:
    for k in range(1, len(trace) + 1):
        prefix_marking(list(trace[:k]), net, im)
        n_prefixes += 1
per_prefix = (time.perf_counter() - t0) / n_prefixes
total_prefixes = sum(len(t) for t in splits.train.values())
print(f"[6] token replay: {per_prefix*1000:.1f} ms/prefix -> "
      f"~{per_prefix*total_prefixes/60:.1f} min for all {total_prefixes} train prefixes "
      f"(one-off, cacheable)")
