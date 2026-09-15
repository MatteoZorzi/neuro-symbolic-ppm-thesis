"""Checks for process/reachability.py, the Python port of DpnReachabilityWoGuards.java.

Structural checks run on hand-built nets (sequence, XOR, AND, silent skip,
silent loop); language equivalence is then cross-checked against pm4py
alignments, which are exact.

    python scripts/_check_reachability.py              # fast checks only
    python scripts/_check_reachability.py --alignments # + alignment cross-check (slow)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pm4py
from pm4py.objects.petri_net.obj import Marking, PetriNet as Pm4pyNet
from pm4py.objects.petri_net.utils import petri_utils

from nspm.data.loader import ACTIVITY, CASE_ID, read_log
from nspm.process.automaton import END, START
from nspm.process.petrinet import PetriNet
from nspm.process.reachability import ReachabilityAutomaton

WITH_ALIGNMENTS = "--alignments" in sys.argv


def build(places, transitions, arcs):
    """Assemble a pm4py net; ``transitions`` maps name -> label (None = silent)."""
    net = Pm4pyNet("check")
    nodes = {}
    for name in places:
        nodes[name] = Pm4pyNet.Place(name)
        net.places.add(nodes[name])
    for name, label in transitions.items():
        nodes[name] = Pm4pyNet.Transition(name, label)
        net.transitions.add(nodes[name])
    for source, target in arcs:
        petri_utils.add_arc_from_to(nodes[source], nodes[target], net)
    return net, nodes


def automaton(net, nodes, source, sink, **kwargs):
    return ReachabilityAutomaton.from_petri_net(
        net, Marking({nodes[source]: 1}), Marking({nodes[sink]: 1}), **kwargs
    )


# --- 1. sequence: a -> b -----------------------------------------------------
net, nodes = build(
    ["p0", "p1", "p2"],
    {"ta": "a", "tb": "b"},
    [("p0", "ta"), ("ta", "p1"), ("p1", "tb"), ("tb", "p2")],
)
seq = automaton(net, nodes, "p0", "p2")
assert seq.accepts(["a", "b"])
assert not seq.accepts(["a"]), "a prefix must not be accepted: only final markings accept"
assert not seq.accepts(["b", "a"])
assert not seq.accepts(["a", "b", "b"])
assert seq.is_conformant_prefix(["a"]) and not seq.is_conformant_prefix(["b"])
assert seq.trap_state is not None and seq.step(seq.trap_state, "a") == seq.trap_state
print(f"[seq]  states {seq.state_count}, accepting {sorted(seq.accepting)}, trap {seq.trap_state}")

# The projection the checker consumes: START -> a -> b -> END.
dfa = seq.to_process_dfa()
assert dfa.allowed_next[START] == frozenset({"a"})
assert dfa.allowed_next["a"] == frozenset({"b"})
assert dfa.allowed_next["b"] == frozenset({END})
assert dfa.start_actions == frozenset({"a"}) and dfa.end_actions == frozenset({"b"})
assert dfa.accepts(["a", "b"]) and not dfa.accepts(["b", "a"])
print(f"[seq]  projected ProcessDFA: {dfa.to_dict()['allowed_next']}")

# --- 2. XOR: a -> (b | c) -> d ----------------------------------------------
net, nodes = build(
    ["p0", "p1", "p2", "p3"],
    {"ta": "a", "tb": "b", "tc": "c", "td": "d"},
    [("p0", "ta"), ("ta", "p1"), ("p1", "tb"), ("tb", "p2"),
     ("p1", "tc"), ("tc", "p2"), ("p2", "td"), ("td", "p3")],
)
xor = automaton(net, nodes, "p0", "p3")
assert xor.accepts(["a", "b", "d"]) and xor.accepts(["a", "c", "d"])
assert not xor.accepts(["a", "b", "c", "d"]), "XOR must not allow both branches"
# b and c are interchangeable, so minimization must merge their target states.
assert xor.step(xor.init_state, "a") is not None
assert xor.state_count == 5, xor.state_count
print(f"[xor]  states {xor.state_count} (b/c branches merged by minimization)")

# --- 3. AND: a -> (b || c) -> d ---------------------------------------------
net, nodes = build(
    ["p0", "pb", "pc", "pb2", "pc2", "p3"],
    {"ta": "a", "tb": "b", "tc": "c", "td": "d"},
    [("p0", "ta"), ("ta", "pb"), ("ta", "pc"),
     ("pb", "tb"), ("tb", "pb2"), ("pc", "tc"), ("tc", "pc2"),
     ("pb2", "td"), ("pc2", "td"), ("td", "p3")],
)
par = automaton(net, nodes, "p0", "p3")
assert par.accepts(["a", "b", "c", "d"]) and par.accepts(["a", "c", "b", "d"])
assert not par.accepts(["a", "b", "d"]), "both parallel branches must complete"
print(f"[and]  states {par.state_count}, both interleavings accepted")

# --- 4. silent skip: a -> (b | tau) -> c ------------------------------------
# The Java original skips invisible transitions outright; this is the case that
# would break under that rule, and the reason tau-closure is implemented here.
net, nodes = build(
    ["p0", "p1", "p2", "p3"],
    {"ta": "a", "tb": "b", "tskip": None, "tc": "c"},
    [("p0", "ta"), ("ta", "p1"), ("p1", "tb"), ("tb", "p2"),
     ("p1", "tskip"), ("tskip", "p2"), ("p2", "tc"), ("tc", "p3")],
)
skip = automaton(net, nodes, "p0", "p3")
assert skip.activities == ("a", "b", "c"), "silent transitions must not enter the alphabet"
assert skip.accepts(["a", "b", "c"])
assert skip.accepts(["a", "c"]), "tau-closure broken: the skip branch is unreachable"
print(f"[tau]  states {skip.state_count}, skip branch reachable")

# --- 5. silent loop: a -> b+ -> c -------------------------------------------
net, nodes = build(
    ["p0", "p1", "p2", "p3"],
    {"ta": "a", "tb": "b", "tredo": None, "tc": "c"},
    [("p0", "ta"), ("ta", "p1"), ("p1", "tb"), ("tb", "p2"),
     ("p2", "tredo"), ("tredo", "p1"), ("p2", "tc"), ("tc", "p3")],
)
loop = automaton(net, nodes, "p0", "p3")
assert loop.accepts(["a", "b", "c"]) and loop.accepts(["a", "b", "b", "b", "b", "c"])
assert not loop.accepts(["a", "c"]), "b is mandatory at least once"
print(f"[loop] states {loop.state_count}, unbounded repetition folded into a cycle")

# --- 6. activities outside the net's alphabet self-loop (Java behaviour) -----
foreign = automaton(net, nodes, "p0", "p3", alphabet=["a", "b", "c", "z"])
assert foreign.activities == ("a", "b", "c", "z")
assert foreign.accepts(["z", "a", "z", "b", "c", "z"]), "foreign activity must self-loop"
assert not foreign.accepts(["z", "a", "c"]), "foreign self-loop must not relax the net"
print(f"[alpha] foreign activity ignored, alphabet {foreign.activities}")

print("ALL STRUCTURAL CHECKS OK")

# --- 7. discovered nets: minimization, projection, serialization -------------
log_path = next((ROOT / "datasets" / "Sepsis_Case").glob("*.xes"))
events = read_log(log_path)
traces = [tuple(group[ACTIVITY]) for _, group in events.groupby(CASE_ID, sort=False)]

net = PetriNet.from_traces({str(i): t for i, t in enumerate(traces)}, noise_threshold=0.0)
raw = ReachabilityAutomaton.from_process_net(net, minimize=False)
mini = ReachabilityAutomaton.from_process_net(net, minimize=True)
print(f"[sepsis] net: {len(net.places)} places, {len(net.transitions)} transitions "
      f"({sum(1 for t in net.transitions if t.label is None)} silent)")
print(f"[sepsis] reachability states {raw.state_count} -> minimized {mini.state_count}, "
      f"accepting {len(mini.accepting)}, edges {mini.transition_count}")

# Minimization preserves the language on real traces and on their prefixes.
words = list(traces[:400]) + [t[:k] for t in traces[:200] for k in range(1, len(t))]
assert all(raw.accepts(w) == mini.accepts(w) for w in words), "minimization changed the language"
print(f"[sepsis] minimization language-preserving on {len(words)} words")

# The projection must never be stricter than the automaton it comes from:
# anything the net permits at a given point must stay permitted by the DFA.
dfa = mini.to_process_dfa()
stricter = 0
for trace in traces[:400]:
    state, previous = mini.init_state, START
    for activity in trace:
        if mini.step(state, activity) != mini.trap_state and \
                activity not in dfa.allowed_activities(previous):
            stricter += 1
        state, previous = mini.step(state, activity), activity
assert stricter == 0, f"projection forbids {stricter} net-legal transitions"
print(f"[sepsis] projected ProcessDFA: {len(dfa.states)} states, "
      f"{dfa.transition_count} transitions, coverage on log "
      f"{dfa.transition_coverage(traces):.3f}, projection never stricter: OK")

# Round-trip through JSON.
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "reachability.json"
    mini.save(path)
    restored = ReachabilityAutomaton.load(path)
assert restored.activities == mini.activities
assert restored.transitions == {s: dict(r) for s, r in mini.transitions.items()}
assert restored.accepting == mini.accepting and restored.trap_state == mini.trap_state
assert restored.state_markings == mini.state_markings
assert all(restored.accepts(w) == mini.accepts(w) for w in words)
print("[sepsis] save/load round-trip identical: OK")

assert mini.to_dot().startswith('digraph "" {') and "doublecircle" in mini.to_dot()
print(f"[sepsis] DOT export: {len(mini.to_dot().splitlines())} lines (trap state omitted)")

print("ALL DISCOVERED-NET CHECKS OK")

# --- 8. exact cross-check against pm4py alignments ---------------------------
if WITH_ALIGNMENTS:
    from pm4py.objects.log.obj import Event, EventLog, Trace

    for name, folder in [("BPIC13", "BPIC_2013_incidents"),
                         ("Sepsis", "Sepsis_Case"),
                         ("BPIC20", "BPIC_2020_DomesticDeclarations")]:
        path = next((ROOT / "datasets" / folder).glob("*.xes"))
        log = read_log(path, max_cases=200)
        cases = [tuple(g[ACTIVITY]) for _, g in log.groupby(CASE_ID, sort=False)]
        pnet, initial, final = pm4py.discover_petri_net_inductive(log, noise_threshold=0.0)
        aut = ReachabilityAutomaton.from_petri_net(pnet, initial, final)

        probes = sorted({*cases[:60], *(t[:k] for t in cases[:40] for k in range(1, len(t)))})[:300]
        event_log = EventLog()
        for probe in probes:
            trace = Trace()
            for activity in probe:
                trace.append(Event({ACTIVITY: activity}))
            event_log.append(trace)

        # An alignment fitness of exactly 1.0 means the trace is in the net's
        # language (the raw 'cost' carries a per-move epsilon, so it is not 0).
        results = pm4py.conformance_diagnostics_alignments(event_log, pnet, initial, final)
        mismatches = [p for p, r in zip(probes, results) if (r["fitness"] == 1.0) != aut.accepts(p)]
        accepted = sum(aut.accepts(p) for p in probes)
        print(f"[align] {name}: {len(probes)} words, {accepted} accepted, "
              f"{len(mismatches)} mismatches")
        assert not mismatches, mismatches[:3]

    print("ALL ALIGNMENT CHECKS OK")
else:
    print("(skipped alignment cross-check; rerun with --alignments)")
