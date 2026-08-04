"""T12 — Matrice sperimentale finale per la presentazione.

Assi: dataset (3) x architettura (gru, lstm) x variante (baseline, checker,
marking, gnn, seq) x noise sulle etichette di train (0/25/50%) x seed 0-9.
Una loss diversa = un modello diverso: checker = stessa rete della baseline
con loss = cross_entropy + 0.5 * forbidden_mass (peso FISSO, pre-registrato
= quello del benchmark di corruzione; niente grid search).

Protocollo (identico al kill test / T10):
- artefatti simbolici (vocab, DFA, Petri net) dal SOLO train, una volta per
  dataset; il noise corrompe solo le etichette y di train (val/test puliti);
- twin-run: dentro una cella (dataset, seed, noise) tutte le varianti vedono
  le STESSE etichette corrotte (stesso RNG -> stessi indici e target);
- risultati appesi riga per riga a runs/final_matrix/results.csv con RESUME:
  le celle gia' presenti nel CSV vengono saltate, quindi lo script si puo'
  lanciare a fette (es. stanotte senza `seq`, e rilanciare identico quando
  gru_seq/lstm_seq esisteranno: fara' solo le celle nuove).

Uso:
  python scripts/final_matrix.py                       # tutto (salta seq se non implementato)
  python scripts/final_matrix.py --datasets Sepsis_Case --variants baseline checker
  python scripts/final_matrix.py --noises 0.0 --archs gru

Le tre varianti "checker_*" (loss del checker + encoder di marking insieme)
chiudono la griglia 2 canali x 4 encoder; non sono nel default perche' T12 e'
pre-registrata a cinque varianti, e vanno in un CSV separato perche'
runs/final_matrix/results.csv e' l'artefatto a 900 righe su cui il notebook
asserisce:
  python scripts/final_matrix.py --datasets Sepsis_Case --archs gru --noises 0.0 \
      --variants checker_marking checker_gnn checker_seq --out-dir channel_cross
"""

import argparse
import csv
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev
from typing import get_args

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nspm.config import ExperimentConfig
from nspm.data.loader import read_log
from nspm.data.preparation import TraceSplits, TraceUtils, PrefixLog, ActivityVocabulary
from nspm.process.petrinet import PetriNet
from nspm.process.automaton import ProcessDFA
from nspm.learning.models import GRAPH_SUFFIXES, ModelKind
from nspm.learning.training import train_model, resolve_device
from nspm.learning.logic import build_allowed_mask
from nspm.learning.evaluation import evaluate_model

DATASETS = ("Sepsis_Case", "BPIC_2013_incidents", "BPIC_2020_DomesticDeclarations")
ARCHS = ("gru", "lstm")
VARIANTS = ("baseline", "checker", "marking", "gnn", "seq")
#: Croce dei due canali: loss del checker E encoder di marking insieme. Fuori
#: dal default di ``--variants`` di proposito -- T12 e' pre-registrata a cinque
#: varianti e un lancio nudo dello script deve continuare a riprodurla.
CROSS_VARIANTS = ("checker_marking", "checker_gnn", "checker_seq")
#: Gradino 6: la GRNN di TACO, convoluzioni dentro le gate. Fuori dal default
#: per la stessa ragione della croce -- T12 e' pre-registrata a cinque varianti.
STEP6_VARIANTS = ("grnn",)
ALL_VARIANTS = VARIANTS + CROSS_VARIANTS + STEP6_VARIANTS
NOISES = (0.0, 0.25, 0.5)
SEEDS = tuple(range(10))
LOGIC_WEIGHT = 0.5  # pre-registrato: lo stesso del benchmark di corruzione (exp 2)

# variante -> (suffisso kind, use_logic, famiglia di dati)
# Le due assi sono indipendenti per costruzione: il suffisso sceglie l'encoder
# simbolico (feature), il booleano accende il termine di loss. Le varianti
# "checker_*" sono esattamente il prodotto delle due.
VARIANT_SPEC = {
    "baseline": ("", False, "plain"),
    "checker": ("", True, "plain"),
    "marking": ("_marking", False, "marked"),
    "gnn": ("_gnn", False, "marked"),
    "seq": ("_seq", False, "sequences"),
    "checker_marking": ("_marking", True, "marked"),
    "checker_gnn": ("_gnn", True, "marked"),
    "checker_seq": ("_seq", True, "sequences"),
    "grnn": ("_grnn", False, "sequences"),
}

# ``secs_per_epoch`` esiste per il gradino 6: la GRNN srotola il tempo a mano,
# quindi il costo e' parte del verdetto quanto l'accuratezza. Le vecchie righe
# non ce l'hanno e il resume legge per nome, quindi convivono.
CSV_FIELDS = ("dataset", "arch", "variant", "noise", "seed",
              "accuracy", "macro_f1", "top3", "violation_rate", "forbidden",
              "best_epoch", "corrupted", "secs_per_epoch")


def find_log(dataset: str) -> Path:
    folder = ROOT / "datasets" / dataset
    for pattern in ("*.xes", "*.csv"):
        try:
            return next(folder.glob(pattern))
        except StopIteration:
            continue
    raise FileNotFoundError(f"No .xes or .csv log in {folder}")


def build_dataset_artifacts(dataset: str, config: ExperimentConfig, families: set[str]) -> dict:
    """Everything derived from one log, built once: splits, symbols, loaders."""
    print(f"[{dataset}] loading and building artifacts...", flush=True)
    events = read_log(find_log(dataset))
    traces = TraceUtils.extract_traces(events)
    splits = TraceSplits.from_traces(traces)

    # stessa convenzione di pipeline/benchmark.py (drop_unseen): i case di
    # val/test con attivita' mai viste nel train non sono ne' predicibili
    # ne' valutabili con un vocabolario costruito sul solo train
    known = {activity for trace in splits.train.values() for activity in trace}
    def drop_unseen(partition):
        kept = {cid: t for cid, t in partition.items()
                if all(activity in known for activity in t)}
        return kept, len(partition) - len(kept)
    validation, dropped_val = drop_unseen(splits.validation)
    test, dropped_test = drop_unseen(splits.test)
    if dropped_val or dropped_test:
        print(f"[{dataset}] dropped {dropped_val} validation / {dropped_test} test "
              f"cases with unseen activities", flush=True)
    splits = replace(splits, validation=validation, test=test)

    vocab = ActivityVocabulary.from_traces(splits.train.values())
    petrinet = PetriNet.from_traces(splits.train)
    mask = build_allowed_mask(ProcessDFA.from_traces(splits.train.values()), vocab)

    def logs(traces_map):
        base = PrefixLog.from_traces(traces_map, vocab)
        out = {"plain": base}
        if "marked" in families:
            out["marked"] = base.with_markings(petrinet)
        if "sequences" in families:
            out["sequences"] = base.with_markings(petrinet).with_marking_sequences(petrinet)
        return out

    train_logs = logs(splits.train)
    batch = config.data.batch_size
    eval_loaders = {
        split_name: {family: log.data_loader(batch, shuffle=False)
                     for family, log in logs(traces_map).items()}
        for split_name, traces_map in (("val", splits.validation), ("test", splits.test))
    }
    print(f"[{dataset}] {len(vocab.activities)} activities, "
          f"{len(petrinet.places)} places, {len(train_logs['plain'])} train prefixes", flush=True)
    return {
        "vocab": vocab, "petrinet": petrinet, "mask": mask,
        "train_logs": train_logs, "eval_loaders": eval_loaders,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--archs", nargs="+", default=list(ARCHS), choices=ARCHS)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=ALL_VARIANTS)
    parser.add_argument("--noises", nargs="+", type=float, default=list(NOISES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--out-dir", default="final_matrix",
        help="sottocartella di runs/ per CSV e checkpoint. Gli esperimenti fuori "
             "da T12 (es. le varianti checker_*) vanno in una cartella propria: "
             "runs/final_matrix/results.csv e' pre-registrato a 900 righe e il "
             "notebook dei risultati ci asserisce sopra.",
    )
    args = parser.parse_args()

    # gate: seq resta fuori finche' i kind non esistono in models.py
    variants = list(args.variants)
    if "seq" in variants and "gru_seq" not in get_args(ModelKind):
        print("NOTA: kind 'gru_seq' non ancora implementato -> variante 'seq' saltata. "
              "Rilancia lo script identico quando esiste: fara' solo le celle mancanti.")
        variants.remove("seq")

    out_dir = ROOT / "runs" / args.out_dir
    (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "results.csv"

    # resume: celle gia' misurate
    done: set[tuple] = set()
    if results_csv.exists():
        with results_csv.open() as handle:
            for row in csv.DictReader(handle):
                done.add((row["dataset"], row["arch"], row["variant"],
                          float(row["noise"]), int(row["seed"])))
        print(f"resume: {len(done)} celle gia' nel CSV, verranno saltate")
    else:
        with results_csv.open("w", newline="") as handle:
            csv.writer(handle).writerow(CSV_FIELDS)

    base_config = ExperimentConfig()
    base_config = replace(base_config, training=replace(base_config.training, logic_weight=LOGIC_WEIGHT))
    device = resolve_device(base_config.training.device)
    families = {VARIANT_SPEC[v][2] for v in variants}

    total = len(args.datasets) * len(args.archs) * len(variants) * len(args.noises) * len(args.seeds)
    print(f"matrice: {len(args.datasets)} dataset x {len(args.archs)} arch x "
          f"{len(variants)} varianti x {len(args.noises)} noise x {len(args.seeds)} seed "
          f"= {total} run ({len(done)} celle gia' nel CSV)", flush=True)
    completed = 0  # solo le celle girate in QUESTO lancio, per la progress bar

    for dataset in args.datasets:
        pending = [c for c in (
            (dataset, arch, variant, noise, seed)
            for arch in args.archs for variant in variants
            for noise in args.noises for seed in args.seeds
        ) if c not in done]
        if not pending:
            print(f"[{dataset}] completo, salto")
            continue

        art = build_dataset_artifacts(dataset, base_config, families)
        vocab, petrinet, mask = art["vocab"], art["petrinet"], art["mask"]
        n_places = len(petrinet.places)

        for seed in args.seeds:
            seed_config = replace(base_config, seed=seed)
            for noise in args.noises:
                for arch in args.archs:
                    for variant in variants:
                        cell = (dataset, arch, variant, noise, seed)
                        if cell in done:
                            continue
                        suffix, use_logic, family = VARIANT_SPEC[variant]
                        kind = arch + suffix

                        # stesse etichette corrotte per tutte le varianti della cella
                        rng = random.Random(seed + int(noise * 1000))
                        train_log, corrupted = art["train_logs"][family].corrupt_targets(noise, rng)
                        train_loader = train_log.data_loader(base_config.data.batch_size, shuffle=True)

                        started = time.perf_counter()
                        train_res = train_model(
                            model_kind=kind,
                            vocabulary=vocab,
                            train_loader=train_loader,
                            validation_loader=art["eval_loaders"]["val"][family],
                            allowed_mask=mask,
                            config=seed_config,
                            checkpoint_path=out_dir / "ckpt" / f"{dataset}_{kind}_{variant}_n{int(noise*100)}_s{seed}.pt",
                            use_logic=use_logic,
                            marking_dim=n_places if family != "plain" else 0,
                            adjacency=petrinet.adjacency_matrices if suffix in GRAPH_SUFFIXES else None,
                        )
                        # Le epoche girate, non quelle configurate: l'early
                        # stopping ferma le varianti a lunghezze diverse.
                        secs_per_epoch = (time.perf_counter() - started) / max(len(train_res.history), 1)
                        eval_res = evaluate_model(
                            model=train_res.model,
                            data_loader=art["eval_loaders"]["test"][family],
                            allowed_mask=mask.to(device),
                            device=device,
                        )
                        with results_csv.open("a", newline="") as handle:
                            csv.writer(handle).writerow([
                                dataset, arch, variant, noise, seed,
                                f"{eval_res.accuracy:.6f}", f"{eval_res.macro_f1:.6f}",
                                f"{eval_res.top_k_accuracy:.6f}", f"{eval_res.violation_rate:.6f}",
                                f"{eval_res.forbidden_mass:.6f}",
                                train_res.best_epoch, corrupted, f"{secs_per_epoch:.2f}",
                            ])
                        done.add(cell)
                        completed += 1
                        print(f"[{dataset} | {arch} {variant:<15} | noise {noise:.2f} | seed {seed}] "
                              f"acc {eval_res.accuracy:.4f} forbidden {eval_res.forbidden_mass:.4f} "
                              f"({completed}/{total})", flush=True)

    # ---- riepilogo dal CSV completo (incluse run di lanci precedenti) -------
    with results_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    print("\n================ RIEPILOGO (medie su seed) ================")
    for dataset in args.datasets:
        for noise in args.noises:
            for arch in args.archs:
                cells = [r for r in rows if r["dataset"] == dataset and r["arch"] == arch
                         and float(r["noise"]) == noise]
                if not cells:
                    continue
                parts = []
                for variant in variants:
                    accs = [float(r["accuracy"]) for r in cells if r["variant"] == variant]
                    if accs:
                        spread = stdev(accs) if len(accs) > 1 else 0.0
                        parts.append(f"{variant} {mean(accs):.4f}±{spread:.4f} (n={len(accs)})")
                print(f"{dataset} | {arch} | noise {noise:.2f}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
