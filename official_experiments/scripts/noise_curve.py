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


# Where the rows of a protocol end up, and how the protocol is described
@dataclass(frozen=True)
class Protocol:

    #: protocol name, passed to ``matrix.py --protocol``
    name: str
    #: subdirectory of ``runs/`` with the CSV and the checkpoints
    out_dir: str
    #: description, for titles and messages
    label: str


#: The two protocols differ in one thing: the partition the knowledge is mined
#: from. The published grids were launched under the letters B (test) and
#: C (train), which survive in their ``protocol`` column.
PROTOCOLS = {
    "test": Protocol(
        name="test",
        out_dir="noise_curve_test",
        label="temporal split · event noise · knowledge from test",
    ),
    "train": Protocol(
        name="train",
        out_dir="noise_curve_train",
        label="temporal split · event noise · knowledge from train",
    ),
}


# The CSV the figures read
def results_path(protocol: Protocol, out_dir: str | None = None) -> Path:
    return ROOT / "runs" / (out_dir or protocol.out_dir) / "results.csv"


def run(command: list[str]) -> int:
    print(" ".join(command), flush=True)
    return subprocess.call(command, cwd=ROOT)


def launch(protocol: Protocol, dataset, noises, archs, variants, seeds,
           out_dir: str | None = None, extra: list[str] | None = None) -> int:
    return run([
        sys.executable, str(HERE / "matrix.py"),
        "--datasets", dataset,
        "--noises", *[str(n) for n in noises],
        "--archs", *archs,
        "--variants", *variants,
        "--seeds", *[str(s) for s in seeds],
        "--out-dir", out_dir or protocol.out_dir,
        "--protocol", protocol.name,
        *(extra or []),
    ])


def report(protocol: Protocol, out_dir: str | None = None) -> None:
    path = results_path(protocol, out_dir)
    if not path.exists():
        raise SystemExit(f"{path} is missing: launch first, without --report")
    curve = pd.read_csv(path).sort_values(["dataset", "variant", "arch", "noise"])
    present = [c for c in COLUMNS if c in curve.columns]

    print(f"\n{'='*100}\nnoise curve — {protocol.label} protocol — "
          f"{path.relative_to(ROOT)} ({len(curve)} runs)\n{'='*100}\n")
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

    print("\n--- cumulative drop against noise 0, in % ---")
    print(joined.reset_index()
                .pivot_table(index="noise", values=["acc_%", "dl_%"], aggfunc="mean")
                .round(2).to_string())


def main() -> int:
    parser = argparse.ArgumentParser(description="Noise curves: nine levels from 0 to 80%, both protocols.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), default="test",
                        help="partition the knowledge is mined from")
    parser.add_argument("--report", action="store_true", help="skip the launch, print the summary only")
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
    if not args.report:
        code = launch(protocol, args.dataset, args.noises, args.archs,
                      args.variants, args.seeds, out_dir=args.out_dir,
                      extra=["--no-net-eval"] if args.no_net_eval else None)
        if code != 0:
            print(f"the trainer exited with code {code}")
            return code
    report(protocol, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
