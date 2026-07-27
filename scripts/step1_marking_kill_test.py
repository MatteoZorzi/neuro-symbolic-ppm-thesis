"""Kill test — scala di ablazione a quattro gradini sul marking della Petri net.
Run gemelli (stessa config, stesso seed, stessi dati), cambia solo l'encoder:
  1. gru          — nessun marking (baseline);
  2. gru_marking  — marking piatto concatenato (stato senza struttura);
  3. gru_gnn      — marking dentro HeteroGraphEncoder (stato con struttura);
  4. gru_seq      — TUTTA la storia dei marking, una GNN per passo + GRU
                    (stato con struttura E tempo).
Lettura: delta(2-1) ~0 gia' noto (rumore, 14 lug); delta(3-2) ~0 e negativo
(15 lug: la struttura sullo snapshot finale non aggiunge nulla). La domanda
dello Step 3 e' delta(4-3): se la sequenza batte lo snapshot, il merito e'
del tempo — cioe' di cio' che il loop overwriting cancella dallo snapshot.
"""

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

# artefatti simbolici solo dal train;val/test simulano dati futuri → niente leakage
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
# La gnn mangia gli stessi batch marcati del piatto: cambia l'encoder, non i dati.
data_loader["gru_gnn"] = data_loader["gru_marking"]

# Il sequenziale e' l'unico che cambia anche i DATI: gli servono i k+1 marking
# dell'intero prefisso, non solo l'ultimo (l'ultimo passo resta pero' identico
# allo snapshot del gradino 3, quindi il confronto e' onesto).
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

deltas_flat = []   # gradino 2 - gradino 1 (gia' noto: rumore)
deltas_gnn = []    # gradino 3 - gradino 2 (la domanda dello Step 2)
deltas_seq = []    # gradino 4 - gradino 3 (la domanda dello Step 3: il tempo)
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
    print(f"[seed {seed}] delta (piatto - liscio): {delta_flat*100:+.2f} pt "
          f"| delta (gnn - piatto): {delta_gnn*100:+.2f} pt "
          f"| delta (seq - gnn): {delta_seq*100:+.2f} pt")

print(f"\nsu {len(SEEDS)} seed:")
for label, deltas in (("piatto - liscio", deltas_flat), ("gnn - piatto", deltas_gnn),
                      ("seq - gnn", deltas_seq)):
    positives = sum(d > 0 for d in deltas)
    print(f"  {label:15s} {[f'{d*100:+.2f}' for d in deltas]} pt "
          f"| media {mean(deltas)*100:+.2f} | std {stdev(deltas)*100:.2f} "
          f"| positivi {positives}/{len(SEEDS)}")