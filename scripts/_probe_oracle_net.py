# Probe -- an "oracle" net mined from the test too (upper bound, NOT a protocol)

import csv
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

from nspm.config import ExperimentConfig
from nspm.data.loader import read_log
from nspm.data.preparation import ActivityVocabulary, PrefixLog, TraceSplits, TraceUtils
from nspm.learning.evaluation import evaluate_model
from nspm.learning.logic import build_allowed_mask
from nspm.learning.training import resolve_device, train_model
from nspm.process.automaton import ProcessDFA
from nspm.process.petrinet import REPLAY_PARAMETERS, PetriNet

SEEDS = (0, 1, 2, 3, 42)
VARIANTS = ("marking", "gnn", "seq")   # kind = "gru_" + variante
FAMILY = {"marking": "marked", "gnn": "marked", "seq": "sequences"}

out_dir = ROOT / "runs" / "_probe_oracle_net"
(out_dir / "ckpt").mkdir(parents=True, exist_ok=True)
results_csv = out_dir / "results.csv"

config = ExperimentConfig()
device = resolve_device(config.training.device)

log_path = next((ROOT / "datasets" / "Sepsis_Case").glob("*.xes"))
traces = TraceUtils.extract_traces(read_log(log_path))
splits = TraceSplits.from_traces(traces)
vocab = ActivityVocabulary.from_traces(splits.train.values())
mask = build_allowed_mask(ProcessDFA.from_traces(splits.train.values()), vocab)

# ----------------------------------------------------------------- the two nets
# honest: training only (the protocol). oracle: train + validation + test.
everything = {**splits.train, **splits.validation, **splits.test}
nets = {
    "train": PetriNet.from_traces(splits.train),
    "oracle": PetriNet.from_traces(everything),
}

print("=" * 70)
print("DIAGNOSTICA DELLE DUE RETI (test set)")
print("=" * 70)


# Share of test traces the replay considers fitting the net
def fit_fraction(net: PetriNet, partition) -> float:
    event_log = net._create_event_log(partition.values())
    results = token_replay.apply(event_log, net.network, net.init_marking,
                                 net.final_marking, parameters=REPLAY_PARAMETERS)
    return sum(bool(r["trace_is_fit"]) for r in results) / len(results)


sample = list(splits.test.items())[:30]  # per-prefix replay is expensive
for name, net in nets.items():
    distinct, events = [], []
    for _, trace in sample:
        sequence = net.marking_sequence(trace)
        distinct.append(len(set(sequence)))
        events.append(len(trace))
    print(f"[{name:6s}] {len(net.places):3d} posti, {len(net.transitions):3d} transizioni "
          f"({sum(t.label is None for t in net.transitions)} silenti) | "
          f"tracce di test conformi {fit_fraction(net, splits.test):.1%} | "
          f"marking distinti per traccia {mean(distinct):.1f} su {mean(events):.1f} eventi")

# --------------------------------------------------------------------- i dati
def loaders_for(net: PetriNet) -> dict:
    def build(traces_map, shuffle):
        base = PrefixLog.from_traces(traces_map, vocab)
        marked = base.with_markings(net)
        return {
            "plain": base.data_loader(config.data.batch_size, shuffle=shuffle),
            "marked": marked.data_loader(config.data.batch_size, shuffle=shuffle),
            "sequences": marked.with_marking_sequences(net).data_loader(
                config.data.batch_size, shuffle=shuffle),
        }
    return {"train": build(splits.train, True),
            "val": build(splits.validation, False),
            "test": build(splits.test, False)}


print("\ncostruzione dei prefissi marcati per entrambe le reti...", flush=True)
data = {name: loaders_for(net) for name, net in nets.items()}

# ------------------------------------------------------------------- training
# resume: cells already in the CSV are skipped (the run is long and closing
# the console kills the process).
done: set[tuple[str, str, int]] = set()
if results_csv.exists():
    with results_csv.open() as handle:
        for row in csv.DictReader(handle):
            done.add((row["net"], row["variant"], int(row["seed"])))
    print(f"resume: {len(done)} run gia' nel CSV")
else:
    with results_csv.open("w", newline="") as handle:
        csv.writer(handle).writerow(("net", "variant", "seed", "accuracy", "top3", "forbidden"))

for seed in SEEDS:
    seed_config = replace(config, seed=seed)

    for name, net in nets.items():
        for variant in VARIANTS:
            if (name, variant, seed) in done:
                continue
            kind = f"gru_{variant}"
            family = FAMILY[variant]
            train_res = train_model(
                model_kind=kind,
                vocabulary=vocab,
                train_loader=data[name]["train"][family],
                validation_loader=data[name]["val"][family],
                allowed_mask=mask,
                config=seed_config,
                checkpoint_path=out_dir / "ckpt" / f"{name}_{variant}_s{seed}.pt",
                use_logic=False,
                marking_dim=len(net.places),
                adjacency=net.adjacency_matrices if variant in ("gnn", "seq") else None,
            )
            eval_res = evaluate_model(
                model=train_res.model,
                data_loader=data[name]["test"][family],
                allowed_mask=mask.to(device),
                device=device,
            )
            with results_csv.open("a", newline="") as handle:
                csv.writer(handle).writerow([
                    name, variant, seed, f"{eval_res.accuracy:.6f}",
                    f"{eval_res.top_k_accuracy:.6f}", f"{eval_res.forbidden_mass:.6f}",
                ])
            print(f"[seed {seed}] rete {name:6s} {variant:<8} accuracy {eval_res.accuracy:.4f} "
                  f"top-3 {eval_res.top_k_accuracy:.4f} forbidden {eval_res.forbidden_mass:.4f}",
                  flush=True)

# -------------------------------------------------------------------- summary
# from the complete CSV, so the summary is right even after a resume
with results_csv.open() as handle:
    rows = list(csv.DictReader(handle))
accuracies: dict[tuple[str, str], dict[int, float]] = {}
for row in rows:
    accuracies.setdefault((row["net"], row["variant"]), {})[int(row["seed"])] = float(row["accuracy"])

print("\n" + "=" * 70)
print(f"ORACOLO - ONESTA, su {len(SEEDS)} seed (Sepsis, GRU)")
print("=" * 70)
for variant in VARIANTS:
    honest_by_seed = accuracies.get(("train", variant), {})
    oracle_by_seed = accuracies.get(("oracle", variant), {})
    seeds = sorted(set(honest_by_seed) & set(oracle_by_seed))
    honest = [honest_by_seed[s] for s in seeds]
    oracle = [oracle_by_seed[s] for s in seeds]
    deltas = [o - h for o, h in zip(oracle, honest)]
    print(f"{variant:<8} onesta {mean(honest):.4f} | oracolo {mean(oracle):.4f} | "
          f"delta {mean(deltas)*100:+.2f} pt (std {stdev(deltas)*100:.2f}, "
          f"positivi {sum(d > 0 for d in deltas)}/{len(deltas)})")
print("\nLettura: un delta dentro il rumore significa che la rete piu' ricca non "
      "aiuta,\nquindi il risultato negativo non dipende dalla qualita' della rete. "
      "Un delta\npositivo NON e' un miglioramento utilizzabile: misura il leakage.")
