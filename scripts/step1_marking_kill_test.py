# Kill test -- four-rung ablation ladder on the Petri net marking

import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nspm.config import ExperimentConfig
from nspm.data.loader import read_log
from nspm.data.preparation import TraceSplits, TraceUtils, PrefixLog, ActivityVocabulary
from nspm.process.petrinet import PetriNet
from nspm.process.automaton import ProcessDFA
from nspm.learning.training import train_model, resolve_device
from nspm.learning.logic import build_allowed_mask
from nspm.learning.evaluation import evaluate_model

log_path = next((ROOT / "datasets" / "Sepsis_Case").glob("*.xes"))
ckpt_dir = ROOT / "runs" / "_step1_kill_test"
ckpt_dir.mkdir(parents=True, exist_ok=True)

events = read_log(log_path)
traces = TraceUtils.extract_traces(events)
splits = TraceSplits.from_traces(traces)

# symbolic artifacts from the training set only; val/test stand for future data -> no leakage
vocab = ActivityVocabulary.from_traces(splits.train.values())
petrinet = PetriNet.from_traces(splits.train)
automaton = ProcessDFA.from_traces(splits.train.values())
mask = build_allowed_mask(automaton, vocab)

config = ExperimentConfig()

data_loader = {
    "gru": {
        "train": PrefixLog.from_traces(splits.train, vocab).data_loader(config.data.batch_size, shuffle=True),
        "val": PrefixLog.from_traces(splits.validation, vocab).data_loader(config.data.batch_size, shuffle=False),
        "test": PrefixLog.from_traces(splits.test, vocab).data_loader(config.data.batch_size, shuffle=False)
    },
    "gru_marking" : {
        "train": PrefixLog.from_traces(splits.train, vocab).with_markings(petrinet).data_loader(config.data.batch_size, shuffle=True),
        "val" : PrefixLog.from_traces(splits.validation, vocab).with_markings(petrinet).data_loader(config.data.batch_size, shuffle=False),
        "test" : PrefixLog.from_traces(splits.test, vocab).with_markings(petrinet).data_loader(config.data.batch_size, shuffle=False)
    }
}
# The gnn eats the same marked batches as the flat one: the encoder changes, not the data.
data_loader["gru_gnn"] = data_loader["gru_marking"]

# The sequential one is the only rung that changes the DATA too: it needs the
# k+1 markings of the whole prefix, not just the last one (the last step stays
# identical to the rung-3 snapshot, so the comparison is honest).
def sequence_log(traces_map, shuffle):
    log = PrefixLog.from_traces(traces_map, vocab).with_markings(petrinet).with_marking_sequences(petrinet)
    return log.data_loader(config.data.batch_size, shuffle=shuffle)

data_loader["gru_seq"] = {
    "train": sequence_log(splits.train, True),
    "val": sequence_log(splits.validation, False),
    "test": sequence_log(splits.test, False),
}

MODELS = ("gru", "gru_marking", "gru_gnn", "gru_seq")
SEEDS = (0, 1, 2, 3, 42)
device = resolve_device(config.training.device)

deltas_flat = []   # rung 2 - rung 1 (already known: noise)
deltas_gnn = []    # rung 3 - rung 2 (the question of Step 2)
deltas_seq = []    # rung 4 - rung 3 (the question of Step 3: time)
for seed in SEEDS:
    seed_config = replace(config, seed=seed)
    test_accuracy = {}
    for name in MODELS:
        train_res = train_model(
            model_kind = name,
            vocabulary = vocab,
            train_loader = data_loader[name]["train"],
            validation_loader = data_loader[name]["val"],
            allowed_mask = mask,
            config = seed_config,
            checkpoint_path = ckpt_dir / f"{name}_seed{seed}.pt",
            use_logic = False,
            marking_dim = len(petrinet.places) if name != "gru" else 0,
            adjacency = petrinet.adjacency_matrices if name in ("gru_gnn", "gru_seq") else None,
        )
        eval_res = evaluate_model(
            model = train_res.model,
            data_loader = data_loader[name]["test"],
            allowed_mask = mask.to(device),
            device = device
        )
        test_accuracy[name] = eval_res.accuracy
        print(f"[seed {seed}] {name:<12} test accuracy {eval_res.accuracy:.4f} "
              f"top-3 {eval_res.top_k_accuracy:.4f} "
              f"forbidden {eval_res.forbidden_mass:.4f} "
              f"best_epoch {train_res.best_epoch}")
    delta_flat = test_accuracy["gru_marking"] - test_accuracy["gru"]
    delta_gnn = test_accuracy["gru_gnn"] - test_accuracy["gru_marking"]
    delta_seq = test_accuracy["gru_seq"] - test_accuracy["gru_gnn"]
    deltas_flat.append(delta_flat)
    deltas_gnn.append(delta_gnn)
    deltas_seq.append(delta_seq)
    print(f"[seed {seed}] delta (flat - plain): {delta_flat*100:+.2f} pt "
          f"| delta (gnn - flat): {delta_gnn*100:+.2f} pt "
          f"| delta (seq - gnn): {delta_seq*100:+.2f} pt")

print(f"\nover {len(SEEDS)} seeds:")
for label, deltas in (("flat - plain", deltas_flat), ("gnn - flat", deltas_gnn),
                      ("seq - gnn", deltas_seq)):
    positives = sum(d > 0 for d in deltas)
    print(f"  {label:15s} {[f'{d*100:+.2f}' for d in deltas]} pt "
          f"| mean {mean(deltas)*100:+.2f} | std {stdev(deltas)*100:.2f} "
          f"| positive {positives}/{len(SEEDS)}")
