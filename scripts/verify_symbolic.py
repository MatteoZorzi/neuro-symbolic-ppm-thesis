"""Executable correctness checks for the symbolic core (T5 review evidence).

1. Property test: PrecedenceConstraint.is_satisfied(trace) == to_dfa().accepts(trace)
   on thousands of random traces (the DFA-vs-LTLf-semantics equivalence).
2. build_allowed_mask: PAD row all-true, forbidden transition masked, START row valid.
3. corrupt_targets: exact count, always-changed labels, prefixes untouched.
4. split_traces: partitions disjoint by case id and complete.
5. Vocabulary round-trip: encode_prefix then decode is identity.
6. Mined constraints hold on the traces they were mined from (confidence=1.0 rules).
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nspm.data.preparation import ActivityVocabulary, PrefixLog, TraceSplits
from nspm.learning.logic import build_allowed_mask
from nspm.process.automaton import START, ProcessDFA
from nspm.process.ltl_constraints import (
    PrecedenceConstraint,
    mine_precedence_constraints,
    sample_satisfying_trace,
    sample_unsatisfying_trace,
)

rng = random.Random(0)
failures = 0

# --- 1. DFA <-> LTLf semantics equivalence (the heart of the embedder branch) ---
alphabet = [f"act{i}" for i in range(6)]
n_checked = 0
for _ in range(200):
    a, b = rng.sample(alphabet, 2)
    constraint = PrecedenceConstraint(a, b, support=1, confidence=1.0)
    dfa = constraint.to_dfa()
    for _ in range(30):
        length = rng.randint(0, 12)
        trace = tuple(rng.choice(alphabet) for _ in range(length))
        n_checked += 1
        if constraint.is_satisfied(trace) != dfa.accepts(trace):
            failures += 1
            print(f"MISMATCH: {constraint.earlier}<{constraint.later} on {trace}")
    # Also check the dedicated generators land on the right side.
    sat = sample_satisfying_trace(constraint, alphabet, length=8, rng=rng)
    unsat = sample_unsatisfying_trace(constraint, alphabet, length=8, rng=rng)
    if not dfa.accepts(sat):
        failures += 1
        print(f"GENERATOR: satisfying trace rejected: {sat}")
    if dfa.accepts(unsat):
        failures += 1
        print(f"GENERATOR: unsatisfying trace accepted: {unsat}")
print(f"[1] DFA==LTLf on {n_checked} random traces + 400 generated: "
      f"{'OK' if failures == 0 else 'FAIL'}")

# --- 2. build_allowed_mask ---
traces = {"c1": ("a", "b", "c"), "c2": ("a", "c"), "c3": ("a", "b", "b", "c")}
vocab = ActivityVocabulary.from_traces(traces.values())
dfa = ProcessDFA.from_traces(traces.values())
mask = build_allowed_mask(dfa, vocab)
t2i, c2i = vocab.token_to_id, vocab.class_to_id
ok = (
    bool(mask[t2i["<PAD>"]].all())
    and bool(mask[t2i[START], c2i["a"]])
    and not bool(mask[t2i[START], c2i["b"]])       # no trace starts with b
    and not bool(mask[t2i["a"], c2i["a"]])          # a->a never observed
    and bool(mask[t2i["a"], c2i["b"]])              # a->b observed
    and bool(mask[t2i["c"]].all())                  # c only precedes END -> unconstrained
)
failures += 0 if ok else 1
print(f"[2] build_allowed_mask semantics: {'OK' if ok else 'FAIL'}")

# --- 3. corrupt_targets ---
examples = PrefixLog.from_traces(traces, vocab)
for fraction in (0.0, 0.3, 1.0):
    perturbed, changed = examples.corrupt_targets(fraction, random.Random(1))
    expected = round(len(examples) * fraction)
    same_prefix = all(p.token_ids == e.token_ids for p, e in zip(perturbed, examples))
    truly_changed = sum(p.target_id != e.target_id for p, e in zip(perturbed, examples))
    ok = changed == expected == truly_changed and same_prefix
    failures += 0 if ok else 1
    print(f"[3] corrupt_targets fraction={fraction}: changed={changed}/{expected} "
          f"{'OK' if ok else 'FAIL'}")

# --- 4. split disjointness ---
many = {f"case{i}": tuple(rng.choice(alphabet) for _ in range(5)) for i in range(50)}
splits = TraceSplits.from_traces(many, seed=7)
sets = [set(splits.train), set(splits.validation), set(splits.test)]
ok = (not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
      and sets[0] | sets[1] | sets[2] == set(many))
failures += 0 if ok else 1
print(f"[4] split_traces disjoint+complete: {'OK' if ok else 'FAIL'}")

# --- 5. vocabulary round-trip ---
prefix = (START, "a", "b", "b", "c")
decoded = tuple(vocab.tokens[i] for i in vocab.encode_prefix(prefix))
ok = decoded == prefix
failures += 0 if ok else 1
print(f"[5] vocabulary round-trip: {'OK' if ok else 'FAIL'}")

# --- 6. mined constraints hold on their own mining set ---
mined = mine_precedence_constraints(many.values(), min_support=2, min_confidence=1.0)
bad = [
    c for c in mined
    for t in many.values()
    if c.later in t and not c.is_satisfied(t)
]
ok = not bad
failures += 0 if ok else 1
print(f"[6] {len(mined)} mined confidence-1.0 rules hold on mining set: "
      f"{'OK' if ok else 'FAIL'}")

print()
print("ALL OK" if failures == 0 else f"{failures} FAILURES")
sys.exit(0 if failures == 0 else 1)
