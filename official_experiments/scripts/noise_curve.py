# Drives a whole noise curve: nine levels from 0 to 80%, one call to matrix.py
# each, for either protocol.

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

#: Key of a cell: it identifies one run under either protocol.
KEY = ["dataset", "arch", "variant", "noise", "seed"]


# How the rows of a protocol are produced, and where they end up
@dataclass(frozen=True)
class Protocol:

    #: protocol letter, passed to ``matrix.py --protocol``
    name: str
    #: subdirectory of ``runs/`` with the first task's CSV and the checkpoints
    out_dir: str
    #: subdirectory of ``runs/`` with the suffix CSV; ``None`` when the first
    #: script already measures both tasks
    suffix_dir: str | None
    #: the script that trains
    trainer: str
    #: the script that scores the suffix from the checkpoints; ``None`` if unused
    suffix_scorer: str | None
    #: description, for titles and messages
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


# The CSV the figures read
def results_path(protocol: Protocol, out_dir: str | None = None) -> Path:

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

    # Stage 2: the suffix, from the checkpoints just written. ``--reference-csv``
    # has to move together with ``--ckpt-dir``, or the safety net would compare
    # these models against the rows of a different grid.
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


# Join the first task and the suffix on the cell key
def merge_stages(protocol: Protocol) -> None:

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

    # Every level against the clean level of the same cell: the cumulative
    # drop, which is what the curve tells.
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
    parser = argparse.ArgumentParser(description="Noise curves: nine levels from 0 to 80%, both protocols.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), default="B")
    parser.add_argument("--report", action="store_true", help="skip the launch, print the summary only")
    parser.add_argument("--merge-only", action="store_true",
                        help="protocol A: redo only the merge of the two stages")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--noises", nargs="+", type=float, default=NOISES)
    parser.add_argument("--archs", nargs="+", default=["lstm"])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--out-dir",
                        help="subdirectory of runs/ in place of the protocol's own. "
                             "For logs that run with fewer methods than the "
                             "others.")
    parser.add_argument("--no-net-eval", action="store_true",
                        help="forwarded to matrix.py: no net automaton, therefore "
                             "no _net columns. Required where the automaton "
                             "cannot be built.")
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
