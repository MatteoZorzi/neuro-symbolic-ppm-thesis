"""Demo: the discovered Petri net and with_marking_sequences on a real case.

Shows (1) what the PetriNet object contains (places, transitions, adjacency,
initial marking / tau-closure), (2) a real case's examples before and after
with_marking_sequences, with markings shown compactly (occupied places only).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nspm.data.loader import read_log
from nspm.data.preparation import TraceSplits, TraceUtils, ActivityVocabulary, PrefixLog
from nspm.process.petrinet import PetriNet

log_path = next((ROOT / "datasets" / "Sepsis_Case").glob("*.xes"))
events = read_log(log_path)
traces = TraceUtils.extract_traces(events)
splits = TraceSplits.from_traces(traces)
net = PetriNet.from_traces(splits.train)

# ---------------------------------------------------------------- the net
print("=" * 70)
print("LA PETRI NET (scoperta dal solo train, inductive miner noise 0.2)")
print("=" * 70)
labelled = [t for t in net.transitions if t.label is not None]
silent = [t for t in net.transitions if t.label is None]
print(f"place:      {len(net.places)}")
print(f"transition: {len(net.transitions)} ({len(labelled)} etichettate + {len(silent)} silenti/tau)")
a_pt, a_tp = net.adjacency_matrices
print(f"adiacenze:  A_PT {len(a_pt)}x{len(a_pt[0])} ({sum(map(sum, a_pt))} archi), "
      f"A_TP {len(a_tp)}x{len(a_tp[0])} ({sum(map(sum, a_tp))} archi)")
print(f"\nprimi 6 place:      {[p.name for p in net.places[:6]]}")
print(f"prime 6 transition: {[t.label for t in labelled[:6]]}")
print(f"\ninitial marking (pm4py): {net.init_marking}")

initial = net.prefix_marking(())
def compact(m):
    """Only the occupied places: {place_index: tokens}."""
    return {i: v for i, v in enumerate(m) if v}
print(f"tau-closure del prefisso vuoto, vettore 26-dim: {initial}")
print(f"  in forma compatta (solo place occupati):     {compact(initial)}")

# ------------------------------------------------- one real case, in/out
vocab = ActivityVocabulary.from_traces(splits.train.values())
short_cases = {cid: t for cid, t in splits.train.items() if len(t) == 5}
case_id, trace = next(iter(sorted(short_cases.items())))
print()
print("=" * 70)
print(f"UN CASE REALE: {case_id!r}, traccia di {len(trace)} eventi")
print("=" * 70)
for k, activity in enumerate(trace, 1):
    print(f"  evento {k}: {activity}")

log = PrefixLog.from_traces({case_id: trace}, vocab)

print(f"\n--- INPUT: PrefixLog.from_traces -> {len(log)} esempi ---")
for n, ex in enumerate(log, 1):
    prefix = [vocab.tokens[t] for t in ex.token_ids]
    target = vocab.activities[ex.target_id]
    print(f"esempio {n}: token_ids={ex.token_ids}")
    print(f"           prefisso={prefix}")
    print(f"           target={target!r} | marking_sequence={ex.marking_sequence}")

seq_log = log.with_marking_sequences(net)

print(f"\n--- OUTPUT: with_marking_sequences (marking compatti: {{place: token}}) ---")
for n, ex in enumerate(seq_log, 1):
    seq = ex.marking_sequence
    print(f"esempio {n}: {len(ex.token_ids)} token <-> {len(seq)} marking")
    steps = [vocab.tokens[t] for t in ex.token_ids]
    for step_name, marking in zip(steps, seq):
        print(f"           dopo {step_name!r:<28} -> {compact(marking)}")

print("\nNota: l'esempio piu' lungo contiene i piu' corti come prefisso della")
print("sua sequenza (slicing); il primo marking e' sempre la tau-closure.")
