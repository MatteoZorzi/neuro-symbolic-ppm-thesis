"""Verifica che le metriche della matrice temporale misurino cio' che dicono.

Non allena niente e non tocca ``runs/temporal_matrix/results.csv``. Fa due
controlli indipendenti, entrambi in pochi minuti.

1. **Test dell'oracolo.** Si sostituisce la predizione con il suffisso VERO e si
   ricalcolano le metriche del task 2. Una metrica ben posta deve dare il suo
   valore perfetto: DL 1.0, exact match 1.0, violazioni 0. Se non lo fa, quella
   metrica sta misurando in parte i dati invece del modello, e il valore che
   restituisce e' un pavimento che il modello non puo' scendere sotto.

2. **Riproduzione da checkpoint.** Si ricarica un modello salvato, si rifanno
   split, vocabolario, rete di Petri, DFA e vincoli da zero, si rivaluta e si
   confronta con la riga corrispondente del CSV. Se i numeri coincidono alla
   sesta cifra, allora (a) la catena di valutazione e' deterministica e (b) i 900
   checkpoint sono davvero riutilizzabili: si possono aggiungere o cambiare
   metriche senza riaddestrare niente.

Uso
---
  python scripts/validate_metrics.py
  python scripts/validate_metrics.py --datasets Sepsis_Case
  python scripts/validate_metrics.py --skip-checkpoints     # solo l'oracolo
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nspm.config import temporal_protocol
from nspm.data.loader import read_log
from nspm.data.preparation import (
    PrefixLog,
    build_splits,
    build_vocabulary,
    knowledge_traces,
)
from nspm.learning.evaluation import evaluate_model
from nspm.learning.logic import build_allowed_mask
from nspm.learning.models import GRAPH_SUFFIXES, build_model
from nspm.learning.trace_prediction import (
    _score_records,
    _violates_precedence,
    evaluate_suffix_prediction,
)
from nspm.learning.training import resolve_device
from nspm.process.automaton import ProcessDFA
from nspm.process.ltl_constraints import mine_precedence_constraints
from nspm.process.petrinet import PetriNet

DATASETS = ("Sepsis_Case", "BPIC_2013_incidents", "BPIC_2020_DomesticDeclarations")

#: Coppie (model_kind, variante, famiglia di dati) da riprovare da checkpoint.
#: Una senza canale simbolico e una con la sequenza di marking, che e' il
#: percorso piu' fragile: in generazione libera i marking vanno rigiocati dalle
#: predizioni del modello e un disallineamento non darebbe errore, solo numeri
#: sbagliati.
CHECKPOINT_PROBES = (("gru", "checker", "plain"), ("gru_seq", "seq", "sequences"))


def find_log(dataset: str) -> Path:
    folder = ROOT / "datasets" / dataset
    for pattern in ("*.xes", "*.csv"):
        try:
            return next(folder.glob(pattern))
        except StopIteration:
            continue
    raise FileNotFoundError(f"nessun log in {folder}")


def rebuild(dataset: str, config) -> dict:
    """Ricostruisce da zero tutto cio' che la matrice ha usato per quel dataset.

    Se questa funzione non fosse deterministica, il confronto col CSV del
    controllo 2 fallirebbe -- il che e' esattamente il motivo per cui il
    controllo esiste.
    """
    splits = build_splits(read_log(find_log(dataset)), config)
    vocabulary = build_vocabulary(splits, config)
    knowledge = knowledge_traces(splits, config)
    petrinet = PetriNet.from_traces(knowledge)
    automaton = ProcessDFA.from_traces(knowledge.values())
    constraints = mine_precedence_constraints(
        knowledge.values(),
        min_support=max(3, len(knowledge) // 10),
        min_confidence=0.95,
        max_constraints=15,
    )
    lengths = sorted(len(trace) for trace in splits.test.values())
    median = lengths[len(lengths) // 2] if lengths else 0
    prefix_lengths = tuple(
        k for k in (median // 2, median // 2 + 1, median // 2 + 2) if k >= 1
    )
    return {
        "splits": splits, "vocabulary": vocabulary, "petrinet": petrinet,
        "automaton": automaton, "constraints": constraints,
        "mask": build_allowed_mask(automaton, vocabulary),
        "prefix_lengths": prefix_lengths,
    }


def oracle_check(dataset: str, art: dict) -> dict:
    """Metriche del task 2 con la predizione sostituita dalla verita'.

    Riporta anche quante volte il **solo prefisso vero** viola gia' un vincolo di
    precedenza: e' la fonte della contaminazione, perche'
    ``precedence_violation_rate`` valuta ``prefisso + suffisso generato`` e non
    sa distinguere una violazione introdotta dal modello da una che era gia'
    nella parte di traccia che gli e' stata data in pasto.
    """
    splits, constraints = art["splits"], art["constraints"]
    records = []
    prefix_only = 0
    for case_id, trace in splits.test.items():
        for k in art["prefix_lengths"]:
            if not 1 <= k < len(trace):
                continue
            records.append({
                "case_id": case_id, "prefix": trace[:k],
                "true": trace[k:], "predicted": trace[k:],
            })
            prefix_only += _violates_precedence(constraints, trace[:k])
    result = _score_records(records, "oracolo", art["automaton"], constraints, 0)
    return {
        "n": result.n_examples,
        "dl_similarity": result.dl_similarity,
        "exact_match": result.exact_match_rate,
        "dfa_violation": result.dfa_violation_rate,
        "precedence_violation": result.precedence_violation_rate,
        "prefix_only_violations": prefix_only / len(records) if records else 0.0,
    }


def checkpoint_check(dataset: str, art: dict, config, device, rows) -> list[dict]:
    """Rivaluta modelli salvati e confronta con il CSV, metrica per metrica."""
    checkpoints = ROOT / "runs" / "temporal_matrix" / "ckpt"
    vocabulary, petrinet = art["vocabulary"], art["petrinet"]
    outcomes = []
    for kind, variant, family in CHECKPOINT_PROBES:
        path = checkpoints / f"{dataset}_{kind}_{variant}_n50_s3.pt"
        if not path.exists():
            print(f"  [{variant}] checkpoint assente, salto: {path.name}")
            continue
        state = torch.load(path, map_location=device)
        model = build_model(
            kind,
            vocabulary_size=len(vocabulary.tokens),
            number_of_classes=len(vocabulary.activities),
            pad_id=vocabulary.pad_id,
            config=config.model,
            marking_dim=len(petrinet.places) if family != "plain" else 0,
            adjacency=petrinet.adjacency_matrices
            if kind[len(kind.split("_")[0]):] in GRAPH_SUFFIXES else None,
        ).to(device)
        model.load_state_dict(state["state_dict"])

        base = PrefixLog.from_traces(art["splits"].test, vocabulary)
        log = base if family == "plain" else (
            base.with_markings(petrinet).with_marking_sequences(petrinet)
        )
        evaluated = evaluate_model(
            model=model,
            data_loader=log.data_loader(config.data.batch_size, shuffle=False),
            allowed_mask=art["mask"].to(device),
            device=device,
        )
        suffix = evaluate_suffix_prediction(
            model, art["splits"].test, vocabulary, art["automaton"], device,
            prefix_lengths=art["prefix_lengths"], constraints=art["constraints"],
            petrinet=petrinet if family != "plain" else None,
        )
        row = next(
            r for r in rows
            if r["dataset"] == dataset and r["variant"] == variant
            and float(r["noise"]) == 0.5 and int(r["seed"]) == 3
            and r["arch"] == kind.split("_")[0]
        )
        for name, recomputed, stored in (
            ("accuracy", evaluated.accuracy, float(row["accuracy"])),
            ("forbidden", evaluated.forbidden_mass, float(row["forbidden"])),
            ("dl_similarity", suffix.dl_similarity, float(row["dl_similarity"])),
        ):
            outcomes.append({
                "variante": variant, "metrica": name,
                "ricalcolata": recomputed, "csv": stored,
                "ok": abs(recomputed - stored) < 1e-6,
            })
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--skip-checkpoints", action="store_true")
    args = parser.parse_args()

    config = temporal_protocol()
    device = resolve_device(config.training.device)

    rows = []
    results_csv = ROOT / "runs" / "temporal_matrix" / "results.csv"
    if results_csv.exists() and not args.skip_checkpoints:
        import csv as csv_module
        with results_csv.open() as handle:
            rows = list(csv_module.DictReader(handle))

    failures = 0
    print("=" * 78)
    print("1. TEST DELL'ORACOLO — predizione sostituita dal suffisso vero")
    print("=" * 78)
    for dataset in args.datasets:
        art = rebuild(dataset, config)
        check = oracle_check(dataset, art)
        perfect = (
            abs(check["dl_similarity"] - 1.0) < 1e-9
            and abs(check["exact_match"] - 1.0) < 1e-9
            and check["dfa_violation"] == 0.0
        )
        failures += 0 if perfect else 1
        print(f"\n[{dataset}] {check['n']} suffissi")
        print(f"  DL similarity        {check['dl_similarity']:.6f}   (atteso 1.0)")
        print(f"  exact match          {check['exact_match']:.6f}   (atteso 1.0)")
        print(f"  violazioni DFA       {check['dfa_violation']:.6f}   (atteso 0.0)")
        print(f"  violazioni precedenza {check['precedence_violation']:.6f}   "
              f"<-- PAVIMENTO: il modello non puo' scendere sotto")
        print(f"  di cui gia' nel solo prefisso vero: {check['prefix_only_violations']:.6f}")
        print(f"  esito: {'OK' if perfect else 'ATTENZIONE'}")

    if not args.skip_checkpoints and rows:
        print("\n" + "=" * 78)
        print("2. RIPRODUZIONE DA CHECKPOINT — noise 0.50, seed 3")
        print("=" * 78)
        for dataset in args.datasets:
            art = rebuild(dataset, config)
            print(f"\n[{dataset}]")
            for outcome in checkpoint_check(dataset, art, config, device, rows):
                failures += 0 if outcome["ok"] else 1
                print(f"  {outcome['variante']:8s} {outcome['metrica']:14s} "
                      f"ricalcolata {outcome['ricalcolata']:.6f} | CSV {outcome['csv']:.6f} "
                      f"| {'OK' if outcome['ok'] else 'DIVERSO'}")

    print("\n" + "=" * 78)
    print("nessuna anomalia" if failures == 0 else f"{failures} controlli da guardare")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
