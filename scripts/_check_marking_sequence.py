# Oracle for PetriNet.marking_sequence (Step 3, increment 1)

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nspm.data.loader import read_log
from nspm.data.preparation import ActivityVocabulary, PrefixLog, TraceSplits, TraceUtils
from nspm.process.petrinet import PetriNet


def main() -> None:
    log_path = next((ROOT / "datasets" / "Sepsis_Case").glob("*.xes"))
    events = read_log(log_path)
    traces = TraceUtils.extract_traces(events)
    splits = TraceSplits.from_traces(traces)
    net = PetriNet.from_traces(splits.train)
    n_places = len(net.places)

    sample = [tuple(t) for t in list(splits.train.values())[:5]]

    for trace in sample:
        seq = net.marking_sequence(trace)

        # 1. one marking per event
        assert len(seq) == len(trace), (len(seq), len(trace))

        # 2. the invariant that makes rung 4 comparable with rung 3
        assert seq[-1] == net.prefix_marking(trace), "seq[-1] != prefix_marking"

        # 3. element-wise coincidence with the static sub-prefix calls
        for i in range(len(trace)):
            assert seq[i] == net.prefix_marking(trace[: i + 1]), f"mismatch at step {i}"

        # 4. nested-prefix slicing (what increment 2 will exploit)
        k = max(1, len(trace) // 2)
        assert net.marking_sequence(trace[:k]) == seq[:k], "slice property broken"

        # 6. well-formed and deterministic
        assert all(len(m) == n_places for m in seq)
        assert all(isinstance(v, int) and v >= 0 for m in seq for v in m)
        assert all(sum(m) >= 1 for m in seq), "a step lost all tokens"
        assert net.marking_sequence(trace) == seq, "not deterministic"

        # 7. the state actually moves
        assert len(set(seq)) > 1 or len(trace) == 1, "marking never changes"

        print(f"[seq] trace len {len(trace):>2}: {len(seq)} markings, "
              f"{len(set(seq))} distinct - OK")

    # 5. empty prefix
    assert net.marking_sequence(()) == (), "empty prefix must give empty tuple"
    print("[seq] empty prefix -> (): OK")

    # cost measurement, to motivate increment 2's per-trace optimization
    longest = max(splits.train.values(), key=len)
    t0 = time.perf_counter()
    net.marking_sequence(tuple(longest))
    elapsed = time.perf_counter() - t0
    print(f"[cost] longest train trace (len {len(longest)}): {elapsed:.2f}s "
          f"({elapsed / len(longest) * 1000:.0f} ms/step)")

    print("ALL MARKING SEQUENCE CHECKS OK")

    # --- increment 2: PrefixLog.with_marking_sequences ---------------------------
    # Properties: (1) len(sequence) == len(token_ids) (tau-closure aligned with
    # START); (2) first element == tau-closure for every example; (3) last element
    # == the static marking of the same example (the sequential rung sees exactly
    # what the static rung saw, plus history); (4) everything else untouched.

    vocab = ActivityVocabulary.from_traces(splits.train.values())
    subset = dict(list(splits.train.items())[:30])
    log = PrefixLog.from_traces(subset, vocab)

    t0 = time.perf_counter()
    seq_log = log.with_marking_sequences(net)
    elapsed = time.perf_counter() - t0

    marked = log.with_markings(net)
    initial = net.prefix_marking(())

    assert len(seq_log) == len(log)
    for seq_example, static_example in zip(seq_log, marked):
        seq = seq_example.marking_sequence
        assert seq is not None
        assert len(seq) == len(seq_example.token_ids), \
            (len(seq), len(seq_example.token_ids))
        assert seq[0] == initial, "first element must be the tau-closure"
        assert seq[-1] == static_example.marking or len(seq) == 1, \
            "last element must equal the static marking"
        if len(seq) == 1:
            assert static_example.marking == initial
    # everything else untouched
    assert [(e.case_id, e.token_ids, e.target_id) for e in seq_log] == \
           [(e.case_id, e.token_ids, e.target_id) for e in log]

    n_events = sum(len(e.token_ids) - 1 for e in log)
    print(f"[data] with_marking_sequences on {len(log)} examples "
          f"({len(subset)} cases): {elapsed:.1f}s - len/first/last/untouched OK")
    print("ALL DATA CHECKS OK")


if __name__ == "__main__":
    main()
