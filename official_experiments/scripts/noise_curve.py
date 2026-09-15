"""Curve del rumore: un punto ogni 10%, da 0 a 80, per entrambi i protocolli.

Le griglie congelate hanno tre soli livelli (0, 0.25, 0.5) e la sonda
``probe_noise80.py`` ha trovato che su BPIC20 il crollo arriva fra 0.6 e 0.8 --
cioe' fuori da quell'intervallo. Tre punti non dicono se la caduta e' graduale o
se c'e' una soglia: questa curva ne mette nove.

Una run per livello, non una cella di griglia: serve la forma della curva, non
la stima puntuale.

I due protocolli
----------------
Stesso comando per entrambi: ``matrix.py --protocol {A,B}``, che misura next
activity e suffisso nella stessa run. Il protocollo e' solo una scelta di
config -- split, vocabolario, sorgente della conoscenza e modello di rumore.

``B`` (default)
    Split temporale, rumore sugli EVENTI, knowledge dal test.
``A``
    Split random, rumore sui TARGET degli esempi (le tracce restano intatte),
    knowledge dal train. Il rumore di A e' un modello piu' debole di quello di
    B: corrompe l'etichetta da predire, non la traccia su cui si condiziona.

Fino al 21/08/2026 il protocollo A girava su ``final_matrix.py`` +
``suffix_on_t12.py``, due stadi con un merge in mezzo. Quella strada e' stata
abbandonata: il secondo stadio doveva RICOSTRUIRE la rete di Petri per valutare
modelli gia' addestrati, e ``pm4py.discover_petri_net_inductive`` non e'
riproducibile fra processi (su BPIC20 da' 21 posti quasi sempre e 22 ogni
tanto), quindi la ricostruzione poteva non combaciare con l'addestramento --
come e' puntualmente successo.

Ogni protocollo ha la sua cartella e il suo CSV, non si mescolano mai: fra A e B
cambiano insieme split, filtri, vocabolario, sorgente della conoscenza e modello
di rumore, quindi i livelli non sono confrontabili.

    conda activate tleaf
    python scripts/noise_curve.py                      # protocollo B
    python scripts/noise_curve.py --protocol A         # protocollo A
    python scripts/noise_curve.py --report             # solo il riepilogo
    python scripts/noise_curve.py --seeds 0 1 2        # curva con piu' repliche

"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
#: This directory: the trainer it launches is a sibling, not a path from ROOT.
HERE = Path(__file__).resolve().parent

DATASET = "BPIC_2020_DomesticDeclarations"
NOISES = [round(0.1 * step, 1) for step in range(9)]  # 0.0 ... 0.8
VARIANTS = ("baseline", "checker", "checker_net", "checker_net_state",
            "marking", "gnn", "seq")

COLUMNS = ["accuracy", "macro_f1", "top3", "dl_similarity", "exact_match",
           "forbidden", "suffix_dfa_violation", "train_compliance",
           "corrupted", "best_epoch"]

#: Chiave di una cella: identifica una run in entrambi i protocolli.
KEY = ["dataset", "arch", "variant", "noise", "seed"]


@dataclass(frozen=True)
class Protocol:
    """Come si producono le righe di un protocollo, e dove finiscono."""

    #: lettera del protocollo, passata a ``matrix.py --protocol``
    name: str
    #: sottocartella di ``runs/`` con il CSV del primo compito e i checkpoint
    out_dir: str
    #: sottocartella di ``runs/`` con il CSV del suffisso; ``None`` se il primo
    #: script misura gia' entrambi i compiti
    suffix_dir: str | None
    #: script che allena
    trainer: str
    #: script che valuta il suffisso dai checkpoint; ``None`` se non serve
    suffix_scorer: str | None
    #: descrizione per titoli e messaggi
    label: str


PROTOCOLS = {
    "B": Protocol(
        name="B",
        out_dir="noise_curve_b",
        suffix_dir=None,
        trainer="matrix.py",
        suffix_scorer=None,
        label="temporal split · event noise · knowledge from test",
    ),
    "C": Protocol(
        name="C",
        out_dir="noise_curve_c",
        suffix_dir=None,
        trainer="matrix.py",
        suffix_scorer=None,
        label="temporal split · event noise · knowledge from train",
    ),
    "A": Protocol(
        name="A",
        out_dir="noise_curve_a",
        suffix_dir=None,
        trainer="matrix.py",
        suffix_scorer=None,
        label="random split · label noise · knowledge from train",
    ),
}


def results_path(protocol: Protocol, out_dir: str | None = None) -> Path:
    """Il CSV che le figure leggono.

    Per B e C e' l'output diretto del trainer; per A e' il merge dei due stadi,
    che ``merge_stages`` scrive accanto al CSV del primo compito.

    ``out_dir`` scavalca la cartella del protocollo. Serve a BPIC15, che gira
    con meno metodi e meno metriche degli altri quattro log e quindi non puo'
    finire nello stesso CSV: righe con le stesse colonne ma con dentro celle
    vuote per costruzione si sommano male, e chi legge il file dopo non ha modo
    di distinguere "non misurato" da "misurato male".
    """

    folder = out_dir or protocol.out_dir
    if protocol.suffix_dir is None:
        return ROOT / "runs" / folder / "results.csv"
    return ROOT / "runs" / folder / "merged.csv"


def run(command: list[str]) -> int:
    print(" ".join(command), flush=True)
    return subprocess.call(command, cwd=ROOT)


def launch(protocol: Protocol, dataset, noises, archs, variants, seeds,
           out_dir: str | None = None, extra: list[str] | None = None) -> int:
    common = [
        "--datasets", dataset,
        "--noises", *[str(n) for n in noises],
        "--archs", *archs,
        "--variants", *variants,
        "--seeds", *[str(s) for s in seeds],
    ]
    code = run([sys.executable, str(HERE / protocol.trainer),
                *common, "--out-dir", out_dir or protocol.out_dir,
                "--protocol", protocol.name, *(extra or [])])
    if code != 0 or protocol.suffix_scorer is None:
        return code

    # Stadio 2: il suffisso dai checkpoint appena scritti. ``--reference-csv``
    # va spostato insieme a ``--ckpt-dir``, altrimenti la rete di sicurezza
    # confronterebbe questi modelli con le righe di un'altra matrice.
    stage_one = ROOT / "runs" / protocol.out_dir
    code = run([sys.executable, str(HERE / protocol.suffix_scorer),
                *common,
                "--ckpt-dir", str(stage_one / "ckpt"),
                "--reference-csv", str(stage_one / "results.csv"),
                "--out-dir", protocol.suffix_dir])
    if code != 0:
        return code
    merge_stages(protocol)
    return 0


def merge_stages(protocol: Protocol) -> None:
    """Unisce primo compito e suffisso sulla chiave di cella.

    Le colonne che i due stadi hanno in comune (``best_epoch``, la provenienza)
    restano quelle del primo: il secondo le riscrive ricaricando lo stesso
    checkpoint, quindi sono ridondanti e non nuove.
    """

    first = ROOT / "runs" / protocol.out_dir / "results.csv"
    second = ROOT / "runs" / protocol.suffix_dir / "results.csv"
    for path in (first, second):
        if not path.exists():
            raise SystemExit(f"manca {path}: il merge del protocollo A ha "
                             f"bisogno di entrambi gli stadi")

    left, right = pd.read_csv(first), pd.read_csv(second)
    duplicated = [c for c in right.columns if c in set(left.columns) and c not in KEY]
    merged = left.merge(right.drop(columns=duplicated), on=KEY, how="left",
                        validate="one_to_one")

    missing = int(merged["dl_similarity"].isna().sum()) if "dl_similarity" in merged else len(merged)
    if missing:
        print(f"ATTENZIONE: {missing}/{len(merged)} righe senza metriche del "
              f"suffisso — lo stadio 2 non ha coperto tutte le celle")
    out = results_path(protocol)
    merged.to_csv(out, index=False)
    print(f"merge: {len(left)} righe x {len(right)} suffissi -> "
          f"{out.relative_to(ROOT)} ({len(merged)} righe, "
          f"{len(merged.columns)} colonne)")


def report(protocol: Protocol, out_dir: str | None = None) -> None:
    path = results_path(protocol, out_dir)
    if not path.exists():
        raise SystemExit(f"manca {path} — lancia prima senza --report")
    curve = pd.read_csv(path).sort_values(["dataset", "variant", "arch", "noise"])
    present = [c for c in COLUMNS if c in curve.columns]

    print(f"\n{'='*100}\ncurva del rumore — protocollo {protocol.label} — "
          f"{path.relative_to(ROOT)} ({len(curve)} run)\n{'='*100}\n")
    print(curve[KEY + present].to_string(index=False))

    # Ogni livello contro il livello pulito della stessa cella: il calo
    # cumulato, che e' cio' che la curva racconta.
    axis = ["dataset", "arch", "variant", "seed"]
    clean = (curve[curve.noise == 0.0]
             .set_index(axis)[["accuracy", "dl_similarity"]])
    if clean.empty:
        return
    joined = curve.set_index(axis).join(clean, rsuffix="_clean")
    joined["acc_%"] = 100 * (joined.accuracy - joined.accuracy_clean) / joined.accuracy_clean
    joined["dl_%"] = 100 * (joined.dl_similarity - joined.dl_similarity_clean) / joined.dl_similarity_clean

    print("\n--- calo cumulato rispetto a rumore 0, in % ---")
    print(joined.reset_index()
                .pivot_table(index="noise", values=["acc_%", "dl_%"], aggfunc="mean")
                .round(2).to_string())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), default="B")
    parser.add_argument("--report", action="store_true", help="salta il lancio")
    parser.add_argument("--merge-only", action="store_true",
                        help="protocollo A: rifa' solo il merge dei due stadi")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--noises", nargs="+", type=float, default=NOISES)
    parser.add_argument("--archs", nargs="+", default=["lstm"])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--out-dir",
                        help="sottocartella di runs/ al posto di quella del "
                             "protocollo. Serve ai log che girano con meno "
                             "metodi degli altri, come BPIC15.")
    parser.add_argument("--no-net-eval", action="store_true",
                        help="inoltrato a matrix.py: niente automa della rete, "
                             "quindi niente colonne _net. Obbligatorio dove "
                             "l'automa non si costruisce.")
    args = parser.parse_args()

    protocol = PROTOCOLS[args.protocol]
    if args.merge_only:
        if protocol.suffix_dir is None:
            raise SystemExit(f"il protocollo {args.protocol} non ha stadi da unire")
        merge_stages(protocol)
    elif not args.report:
        code = launch(protocol, args.dataset, args.noises, args.archs,
                      args.variants, args.seeds, out_dir=args.out_dir,
                      extra=["--no-net-eval"] if args.no_net_eval else None)
        if code != 0:
            print(f"lo stadio e' uscito con codice {code}")
            return code
    report(protocol, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
