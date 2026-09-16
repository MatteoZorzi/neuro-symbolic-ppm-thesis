# T10 — Label noise sulla branch marking (domanda del prof)

import csv
import random
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nspm.config import ExperimentConfig
from nspm.data.loader import read_log
from nspm.data.preparation import (
    PrefixLog,
    build_splits,
    build_vocabulary,
    knowledge_traces,
)
from nspm.process.petrinet import PetriNet
from nspm.process.automaton import ProcessDFA
from nspm.learning.training import train_model, resolve_device
from nspm.learning.logic import build_allowed_mask
from nspm.learning.evaluation import evaluate_model

log_path = next((ROOT / "datasets" / "Sepsis_Case").glob("*.xes"))
out_dir = ROOT / "runs" / "_t10_noise_grid"
out_dir.mkdir(parents=True, exist_ok=True)
results_csv = out_dir / "results.csv"

config = ExperimentConfig()

events = read_log(log_path)
splits = build_splits(events, config)

# symbolic artifacts from the partition named by ``knowledge_source`` (training
# by default); label noise does not touch the traces, so net, DFA and markings
# stay identical at every level of corruption
vocab = build_vocabulary(splits, config)
knowledge = knowledge_traces(splits, config)
petrinet = PetriNet.from_traces(knowledge)
automaton = ProcessDFA.from_traces(knowledge.values())
mask = build_allowed_mask(automaton, vocab)

# base logs built ONCE (with_markings costs ~35s); per-cell corruption is cheap
# and preserves the markings (replace touches only target_id)
train_plain = PrefixLog.from_traces(splits.train, vocab)
train_marked = train_plain.with_markings(petrinet)
loaders_clean = {
    "gru": {
        "val": PrefixLog.from_traces(splits.validation, vocab).data_loader(config.data.batch_size, shuffle=False),
        "test": PrefixLog.from_traces(splits.test, vocab).data_loader(config.data.batch_size, shuffle=False),
    },
    "gru_marking": {
        "val": PrefixLog.from_traces(splits.validation, vocab).with_markings(petrinet).data_loader(config.data.batch_size, shuffle=False),
        "test": PrefixLog.from_traces(splits.test, vocab).with_markings(petrinet).data_loader(config.data.batch_size, shuffle=False),
    },
}
loaders_clean["gru_gnn"] = loaders_clean["gru_marking"]
train_base = {"gru": train_plain, "gru_marking": train_marked, "gru_gnn": train_marked}

MODELS = ("gru", "gru_marking", "gru_gnn")
NOISES = (0.0, 0.25, 0.5)
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 42)
device = resolve_device(config.training.device)

if not results_csv.exists():
    with results_csv.open("w", newline="") as handle:
        csv.writer(handle).writerow(
            ["seed", "noise", "model", "accuracy", "top3", "forbidden", "best_epoch", "corrupted"])

results = {}  # (seed, noise, model) -> (accuracy, forbidden)
for seed in SEEDS:
    seed_config = replace(config, seed=seed)
    for noise in NOISES:
        for name in MODELS:
            # same RNG seed for the three models -> same corrupted labels
            rng = random.Random(seed + int(noise * 1000))
            corrupted_log, changed = train_base[name].corrupt_targets(noise, rng)
            train_loader = corrupted_log.data_loader(config.data.batch_size, shuffle=True)

            train_res = train_model(
                model_kind=name,
                vocabulary=vocab,
                train_loader=train_loader,
                validation_loader=loaders_clean[name]["val"],
                allowed_mask=mask,
                config=seed_config,
                checkpoint_path=out_dir / f"{name}_seed{seed}_noise{int(noise * 100)}.pt",
                use_logic=False,
                marking_dim=len(petrinet.places) if name != "gru" else 0,
                adjacency=petrinet.adjacency_matrices if name == "gru_gnn" else None,
            )
            eval_res = evaluate_model(
                model=train_res.model,
                data_loader=loaders_clean[name]["test"],
                allowed_mask=mask.to(device),
                device=device,
            )
            results[(seed, noise, name)] = (eval_res.accuracy, eval_res.forbidden_mass)
            with results_csv.open("a", newline="") as handle:
                csv.writer(handle).writerow(
                    [seed, noise, name, f"{eval_res.accuracy:.6f}",
                     f"{eval_res.top_k_accuracy:.6f}", f"{eval_res.forbidden_mass:.6f}",
                     train_res.best_epoch, changed])
            print(f"[seed {seed} noise {noise:.2f}] {name:<12} "
                  f"acc {eval_res.accuracy:.4f} forbidden {eval_res.forbidden_mass:.4f} "
                  f"({changed} label corrotte, best_epoch {train_res.best_epoch})",
                  flush=True)

# --- riepilogo: per livello di noise, i due delta su accuracy e forbidden ----
print("\n================ RIEPILOGO ================")
for metric_idx, metric in ((0, "accuracy"), (1, "forbidden")):
    print(f"\n--- {metric} ---")
    for noise in NOISES:
        per_model = {
            name: [results[(seed, noise, name)][metric_idx] for seed in SEEDS]
            for name in MODELS
        }
        line = " | ".join(
            f"{name} {mean(vals):.4f}±{stdev(vals):.4f}" for name, vals in per_model.items())
        print(f"noise {noise:.2f}: {line}")
        for label, a, b in (("piatto - liscio", "gru_marking", "gru"),
                            ("gnn - piatto", "gru_gnn", "gru_marking")):
            deltas = [results[(seed, noise, a)][metric_idx]
                      - results[(seed, noise, b)][metric_idx] for seed in SEEDS]
            positives = sum(d > 0 for d in deltas)
            print(f"    {label:15s} media {mean(deltas) * 100:+.2f} pt "
                  f"| std {stdev(deltas) * 100:.2f} | positivi {positives}/{len(SEEDS)}")
