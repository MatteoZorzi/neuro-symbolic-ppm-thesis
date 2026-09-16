# One cell of a grid: trains one model and measures next activity and suffix
# prediction in the same run.

import argparse
import csv
import random
import sys
import time
import torch
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nspm.config import temporal_protocol
from nspm.data.loader import read_log
from nspm.data.preparation import (
    PrefixLog,
    TraceUtils,
    build_splits,
    build_vocabulary,
    knowledge_traces,
    unseen_activities,
)
from nspm.process import artifact_store
from nspm.process.petrinet import PetriNet
from nspm.process.automaton import ProcessDFA
from nspm.process.reachability import ReachabilityAutomaton
from nspm.process.ltl_constraints import compliance_ratio, mine_precedence_constraints
from nspm.learning.axel_losses import GlobalLogicLoss, LocalLogicLoss, TensorDFA
from nspm.learning.models import GRAPH_SUFFIXES
from nspm.learning.training import train_model, resolve_device
from nspm.learning.logic import build_allowed_mask, build_state_mask
from nspm.learning.evaluation import evaluate_model
from nspm.learning.trace_prediction import evaluate_suffix_prediction

DATASETS = ("Sepsis_Case", "BPIC_2013_incidents", "BPIC_2020_DomesticDeclarations")
#: Allowed but NOT in the default set: the earlier grids were run over the
#: three logs above, and a launch without arguments keeps reproducing them.
#: BPIC 2012 is asked for by name.
EXTRA_DATASETS = ("BPI_Challenge_2012",)
ALL_DATASETS = DATASETS + EXTRA_DATASETS
ARCHS = ("gru", "lstm")
VARIANTS = ("baseline", "checker", "marking", "gnn", "seq")
CROSS_VARIANTS = ("checker_marking", "checker_gnn", "checker_seq")
STEP6_VARIANTS = ("grnn",)
#: Checkers driven by the automaton of the PETRI NET instead of the empirical
#: DFA (see ``MASK_SPEC``). Out of the default: own directory, merged in
#: analysis.
NETDFA_VARIANTS = ("checker_net", "checker_net_state")
#: The two methods of Mezini et al. Out of the default for the same reason as
#: the others: a launch without arguments has to keep reproducing the frozen
#: grid.
AXEL_VARIANTS = ("lll", "gll")
ALL_VARIANTS = (VARIANTS + CROSS_VARIANTS + STEP6_VARIANTS + NETDFA_VARIANTS
                + AXEL_VARIANTS)
NOISES = (0.0, 0.25, 0.5)
SEEDS = tuple(range(10))
LOGIC_WEIGHT = 0.5  # pre-registered: the same as the earlier grids

# variant -> (kind suffix, use_logic, data family, automaton of the loss).
# The axes are independent: the suffix picks the symbolic encoder, the boolean
# turns the loss term on, the last field picks the automaton behind the mask.
VARIANT_SPEC = {
    "baseline": ("", False, "plain", "dfa"),
    "checker": ("", True, "plain", "dfa"),
    "marking": ("_marking", False, "marked", "dfa"),
    "gnn": ("_gnn", False, "marked", "dfa"),
    "seq": ("_seq", False, "sequences", "dfa"),
    "checker_marking": ("_marking", True, "marked", "dfa"),
    "checker_gnn": ("_gnn", True, "marked", "dfa"),
    "checker_seq": ("_seq", True, "sequences", "dfa"),
    "grnn": ("_grnn", False, "sequences", "dfa"),
    "checker_net": ("", True, "plain", "net"),
    "checker_net_state": ("", True, "stated", "net_state"),
    # The two methods of Mezini et al.: same trunk as the baseline and the same
    # automaton as ``checker_net``, only HOW the constraint enters the loss
    # changes. The direct term of comparison is ``checker_net``.
    "lll": ("", True, "plain", "axel_local"),
    "gll": ("", True, "plain", "axel_global"),
}

#: Hyperparameters of the two losses of Mezini et al., fixed in advance and the
#: same for every seed. ``alpha`` weighs the supervision, one value per method.
#: ``AXEL_SAMPLES`` must stay THE SAME on every cell, or the cells would not compare.
AXEL_ALPHA_LOCAL = 0.25
AXEL_ALPHA_GLOBAL = 0.75
AXEL_TEMPERATURE = 0.5
AXEL_SAMPLES = 10

#: Size of the recurrent trunk, the numbers of Mezini et al. (2026): two layers
#: of a hundred units. It holds for ALL NINE variants: with two different trunks
#: the grid would measure architectures instead of knowledge channels.
RECURRENT_HIDDEN = 100
RECURRENT_LAYERS = 2

#: Ceiling on the epochs. Twelve were too few for the larger trunk; early
#: stopping is unchanged, so the cells that converge early keep stopping there.
MAX_EPOCHS = 60

#: The automata the loss mask can come from: ``dfa`` is the empirical
#: directly-follows relation, ``net`` the net's reachability automaton projected
#: onto it, ``net_state`` the same automaton unprojected, one row per state.
MASK_SPEC = ("dfa", "net", "net_state")
#: Every variant is EVALUATED on this one, so the columns compare across rows.
EVAL_MASK = "dfa"

#: Who needs the structure extracted from the Petri net. Two methods compare
#: only if they see THE SAME symbolic object: ``lll`` and ``gll`` read the net
#: too, projected for the local loss and tensorised for the global one.
NET_DFA_USERS = frozenset({"net", "axel_local", "axel_global"})
#: Who has the automaton built. The per-state mask does not go through the
#: projection, so it asks for the automaton and nothing else.
AUTOMATON_USERS = NET_DFA_USERS | {"net_state"}

CSV_FIELDS = (
    "dataset", "arch", "variant", "noise", "seed",
    # --- task 1: next activity. Every conformance metric exists in TWO copies:
    #     against the empirical directly-follows relation, and with ``_net``
    #     against the projected net automaton, which is wider but not nested.
    "accuracy", "macro_f1", "top3", "violation_rate", "forbidden",
    "violation_rate_net", "forbidden_net",
    # --- task 2: suffix prediction, the model regenerates from its own output.
    #     The same pair of yardsticks on the same generated traces.
    "dl_similarity", "suffix_accuracy", "exact_match",
    "suffix_dfa_violation", "suffix_dfa_violation_net",
    "suffix_precedence_violation",
    # Token-replay fitness of the WHOLE trace (real prefix + generated suffix)
    # against the Petri net.
    "suffix_net_fitness",
    "n_suffix", "prefix_lengths",
    # --- noise. ``corrupted`` counts the EVENTS whose label was replaced;
    #     ``train_compliance`` is the fraction of training traces that still
    #     satisfy the constraints after the injection.
    "best_epoch", "corrupted", "train_compliance", "secs_per_epoch",
    # --- cost. ``secs_train`` is the wall time of training; the other four are
    #     the symbolic part, which the neural network does not pay;
    #     ``secs_artifacts`` holds all of them plus reading the log.
    "secs_train", "secs_petrinet", "secs_dfa", "secs_automaton",
    "secs_markings", "secs_artifacts",
    # Which card the cell ran on: two datasets that ended up on different cards
    # are not comparable on the seconds.
    "gpu",
    # --- reproducibility. ``knowledge_key`` is the fingerprint of the traces
    #     mined from, ``net_fingerprint`` that of the net; ``artifacts_cached``
    #     says whether the symbolic seconds are those of the original construction.
    "knowledge_key", "net_fingerprint", "artifacts_cached",
    "split", "knowledge", "vocab_scope", "train_frac", "val_frac", "test_frac",
)


def find_log(dataset: str) -> Path:
    folder = ROOT / "datasets" / dataset
    # ``*.xes.gz`` is in the list because public logs often arrive compressed;
    # pm4py reads them without unpacking.
    for pattern in ("*.xes", "*.xes.gz", "*.csv", "*.csv.gz"):
        try:
            return next(folder.glob(pattern))
        except StopIteration:
            continue
    raise FileNotFoundError(f"No .xes or .csv log in {folder}")


# Everything derived from a log, built once per dataset
def build_dataset_artifacts(dataset: str, config, families: set[str],
                            masks_needed: set[str] = frozenset({"dfa"}),
                            cache_dir: Path | None = None) -> dict:
    print(f"[{dataset}] loading the log and building the artifacts...", flush=True)
    # The times of the symbolic part, reported in the CSV beside those of
    # training: they say what the knowledge costs, which under event noise is
    # paid again at every level.
    timings = {"petrinet": 0.0, "dfa": 0.0, "automaton": 0.0,
               "markings": 0.0, "artifacts": 0.0}
    build_started = time.perf_counter()

    events = read_log(find_log(dataset))
    splits = build_splits(events, config)

    # Under ``vocabulary_scope="train"`` held-out cases with activities never
    # seen in training are dropped, as the earlier random-split runs did.
    # Under ``"all"`` nothing is dropped and the set is empty by construction.
    if config.data.vocabulary_scope == "train":
        known = {a for trace in splits.train.values() for a in trace}

        def keep(partition):
            kept = {cid: t for cid, t in partition.items()
                    if all(a in known for a in t)}
            return kept, len(partition) - len(kept)

        validation, dropped_val = keep(splits.validation)
        test, dropped_test = keep(splits.test)
        if dropped_val or dropped_test:
            print(f"[{dataset}] dropped {dropped_val} validation / "
                  f"{dropped_test} test cases with activities never seen in training",
                  flush=True)
        splits = replace(splits, validation=validation, test=test)

    vocab = build_vocabulary(splits, config)
    unseen = unseen_activities(splits, vocab)
    if unseen:
        raise ValueError(
            f"[{dataset}] activities the vocabulary cannot represent in the held-out "
            f"partitions: {sorted(unseen)}. The temporal protocol drops no trace, so "
            f"the vocabulary has to cover them: check data.vocabulary_scope."
        )

    # Background knowledge from the partition named by ``knowledge_source``.
    knowledge = knowledge_traces(splits, config)

    # The mined objects come from disk when they are already there, keyed by
    # the fingerprint of the knowledge traces. Without them a run cannot be
    # RELOADED: the inductive miner generates fresh names at every call.
    store_key = artifact_store.knowledge_fingerprint(knowledge, noise_threshold=0.2)
    store_file = artifact_store.cache_path(cache_dir, dataset, store_key) \
        if cache_dir is not None else None
    cached = artifact_store.load(store_file) if store_file is not None else {}
    fresh = False

    started = time.perf_counter()
    if "petrinet" in cached:
        petrinet = cached["petrinet"]
        # The seconds are those of the REAL construction, read back from the
        # cache: the column has to say what discovering the net costs, not what
        # opening a file costs.
        timings["petrinet"] = cached.get("secs_petrinet", 0.0)
    else:
        petrinet = PetriNet.from_traces(knowledge)
        timings["petrinet"] = time.perf_counter() - started
        fresh = True

    started = time.perf_counter()
    if "automaton" in cached:
        automaton = cached["automaton"]
        timings["dfa"] = cached.get("secs_dfa", 0.0)
        mask = build_allowed_mask(automaton, vocab)
    else:
        automaton = ProcessDFA.from_traces(knowledge.values())
        mask = build_allowed_mask(automaton, vocab)
        timings["dfa"] = time.perf_counter() - started
        fresh = True
    masks = {"dfa": mask}
    reachability = cached.get("reachability")
    net_dfa = state_mask = None
    started = time.perf_counter()
    if reachability is not None and masks_needed & AUTOMATON_USERS:
        timings["automaton"] = cached.get("secs_automaton", 0.0)
        print(f"[{dataset}] net automaton (from cache): "
              f"{reachability.state_count} states "
              f"({len(reachability.accepting)} accepting)", flush=True)
    elif masks_needed & AUTOMATON_USERS:
        # The same net the marking/gnn/seq variants read. ``alphabet`` keeps
        # the activities pruned by the miner: about their position the net
        # says nothing, and saying nothing is not forbidding them.
        reachability = ReachabilityAutomaton.from_process_net(
            petrinet, alphabet=vocab.activities
        )
        timings["automaton"] = time.perf_counter() - started
        fresh = True
        print(f"[{dataset}] net automaton: {reachability.state_count} states "
              f"({len(reachability.accepting)} accepting)", flush=True)
    if masks_needed & NET_DFA_USERS:
        # Projection onto directly-follows: it loses the memory of the markings,
        # but it is the only shape a mask on the last token accepts.
        net_dfa = reachability.to_process_dfa()
        masks["net"] = build_allowed_mask(net_dfa, vocab)
        print(f"[{dataset}] projected: {net_dfa.transition_count} transitions | "
              f"allowed cells: net {int(masks['net'].sum())} vs empirical "
              f"{int(mask.sum())} of {mask.numel()}", flush=True)
    if "net_state" in masks_needed:
        # No projection: one row per automaton state, the memory survives.
        state_mask = build_state_mask(reachability, vocab)
        print(f"[{dataset}] per-state mask: {tuple(state_mask.shape)} "
              f"({int(state_mask.sum())} allowed cells)", flush=True)
    # The precedence constraints give conformance at TRACE level. They are
    # cached too: ``mine_precedence_constraints`` is not deterministic across
    # processes, and two runs must not measure against different rules.
    if "constraints" in cached:
        constraints = cached["constraints"]
    else:
        constraints = mine_precedence_constraints(
            knowledge.values(),
            min_support=max(3, len(knowledge) // 10),
            min_confidence=0.95,
            max_constraints=15,
        )
        fresh = True

    # Saved only if something was mined just now. The cache grows by addition:
    # a later launch that asks for the automaton adds it without redoing the rest.
    if store_file is not None and fresh:
        artifact_store.save(store_file, {
            "dataset": dataset, "knowledge_key": store_key,
            "petrinet": petrinet, "automaton": automaton,
            "reachability": reachability, "constraints": constraints,
            "secs_petrinet": timings["petrinet"], "secs_dfa": timings["dfa"],
            "secs_automaton": timings["automaton"],
        })

    # Prefixes: half the median test length, +1, +2 -- computed on the test set
    # because that is where suffix prediction is scored.
    lengths = sorted(len(t) for t in splits.test.values())
    median_length = lengths[len(lengths) // 2] if lengths else 0
    prefix_lengths = tuple(
        k for k in (median_length // 2, median_length // 2 + 1, median_length // 2 + 2)
        if k >= 1
    )

    # The global loss needs the automaton in tensor form: the SAME projection
    # ``checker_net`` reads, written as transition matrices.
    tensor_dfa = gll_horizon = None
    started = time.perf_counter()
    if "axel_global" in masks_needed:
        tensor_dfa = TensorDFA.from_process_dfa(
            net_dfa, vocab.activities, resolve_device(config.training.device)
        )
        # How far the rollout runs on: the training prefixes start around half
        # the median length, so the tail to generate is of the order of the
        # other half, the same scale as the suffix task.
        gll_horizon = max(1, median_length // 2 + 2)
        print(f"[{dataset}] tensor automaton: {tensor_dfa.n_states} states x "
              f"{tensor_dfa.n_actions} actions | rollout {gll_horizon} steps x "
              f"{AXEL_SAMPLES} samples", flush=True)
    # Tensorisation is the checker's own automaton rewritten as matrices, so its
    # cost belongs with the DFA's and not with the net's.
    timings["dfa"] += time.perf_counter() - started

    def logs(traces_map):
        base = PrefixLog.from_traces(traces_map, vocab)
        out = {"plain": base}
        if "marked" in families:
            out["marked"] = base.with_markings(petrinet)
        if "sequences" in families:
            out["sequences"] = base.with_markings(petrinet).with_marking_sequences(petrinet)
        if "stated" in families:
            # The same model as the baseline: the state is not an input, it
            # only indexes the row of the loss mask.
            out["stated"] = base.with_automaton_states(reachability)
        return out

    batch = config.data.batch_size
    started = time.perf_counter()
    eval_loaders = {
        split_name: {family: log.data_loader(batch, shuffle=False)
                     for family, log in logs(traces_map).items()}
        for split_name, traces_map in (("val", splits.validation), ("test", splits.test))
    }
    # Token replay of validation and test, the only part built once: the
    # training logs are rebuilt at every cell because the noise changes them.
    timings["markings"] = time.perf_counter() - started

    variants_train = len({tuple(t) for t in splits.train.values()})
    clean_ratio = compliance_ratio(constraints, splits.train.values())
    print(f"[{dataset}] {len(vocab.activities)} activities, {len(petrinet.places)} places\n"
          f"[{dataset}] split {config.data.split_strategy} "
          f"{len(splits.train)}/{len(splits.validation)}/"
          f"{len(splits.test)} cases (duplicates kept: {variants_train} distinct variants "
          f"over {len(splits.train)} training traces)\n"
          f"[{dataset}] knowledge from {config.data.knowledge_source}: "
          f"{len(knowledge)} traces -> DFA {len(automaton.states)} states/"
          f"{automaton.transition_count} transitions, {len(constraints)} constraints\n"
          f"[{dataset}] compliance of the clean training log: {clean_ratio:.3f}\n"
          f"[{dataset}] suffix: median test length {median_length}, "
          f"prefixes {prefix_lengths}", flush=True)
    timings["artifacts"] = time.perf_counter() - build_started
    print(f"[{dataset}] construction: net {timings['petrinet']:.1f}s | "
          f"dfa {timings['dfa']:.1f}s | automaton {timings['automaton']:.1f}s | "
          f"held-out markings {timings['markings']:.1f}s | "
          f"total {timings['artifacts']:.1f}s", flush=True)
    return {
        "timings": timings,
        "vocab": vocab, "petrinet": petrinet, "mask": mask, "masks": masks,
        "reachability": reachability, "state_mask": state_mask,
        # The two fingerprints that end up in the CSV: which traces were mined
        # from, and which net came out. A changed net is noticed here, not
        # discovered from the numbers.
        "knowledge_key": store_key,
        "net_fingerprint": artifact_store.net_fingerprint(petrinet),
        "cached": not fresh,
        # The net automaton projected onto directly-follows, the structure the
        # ``_net`` columns are measured against; ``None`` where there is none.
        "net_dfa": net_dfa,
        "tensor_dfa": tensor_dfa, "gll_horizon": gll_horizon,
        "build_logs": logs, "eval_loaders": eval_loaders,
        "train_traces": splits.train,
        "automaton": automaton, "constraints": constraints,
        "test_traces": splits.test, "prefix_lengths": prefix_lengths,
    }


# How the variant injects the knowledge into the loss
def loss_arguments(mask_key: str, art: dict) -> dict:
    if mask_key == "net_state":
        return {"logic_mode": "checker_state",
                "allowed_mask": art["masks"][EVAL_MASK],
                "logic_mask": art["state_mask"]}
    if mask_key == "axel_local":
        # The same mask as ``checker_net``. ``allowed_mask`` stays the
        # empirical one: it is the conformance ``train_model`` writes into the
        # history, not the constraint.
        return {"logic_mode": "axel_local",
                "allowed_mask": art["masks"][EVAL_MASK],
                "axel_loss": LocalLogicLoss(art["masks"]["net"],
                                            alpha=AXEL_ALPHA_LOCAL)}
    if mask_key == "axel_global":
        return {"logic_mode": "axel_global",
                "allowed_mask": art["masks"][EVAL_MASK],
                "axel_loss": GlobalLogicLoss(
                    art["tensor_dfa"], horizon=art["gll_horizon"],
                    alpha=AXEL_ALPHA_GLOBAL, temperature=AXEL_TEMPERATURE,
                    num_samples=AXEL_SAMPLES,
                )}
    return {"logic_mode": "checker", "allowed_mask": art["masks"][mask_key]}


def main() -> None:
    parser = argparse.ArgumentParser(description="One cell of a noise grid: next activity and suffix, same run.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", choices=("test", "train"), default="test",
                        help="partition the knowledge is mined from. Temporal "
                             "split and event noise in both: only the config "
                             "changes.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS),
                        choices=ALL_DATASETS)
    parser.add_argument("--archs", nargs="+", default=list(ARCHS), choices=ARCHS)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=ALL_VARIANTS)
    parser.add_argument("--noises", nargs="+", type=float, default=list(NOISES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--out-dir", default="temporal_matrix",
                        help="subdirectory of runs/ for the CSV and the checkpoints")
    parser.add_argument("--no-suffix", action="store_true",
                        help="skip suffix prediction (its columns stay empty)")
    parser.add_argument("--no-artifact-cache", action="store_true",
                        help="always mine net, DFA, automaton and constraints "
                             "instead of reusing those saved in runs/_artifacts")
    parser.add_argument("--no-net-eval", action="store_true",
                        help="do not build the net automaton for evaluation only: "
                             "the _net columns stay empty. Needed where the "
                             "automaton does not terminate, and to be used "
                             "with the variants that do not require it.")
    args = parser.parse_args()

    variants = list(args.variants)

    out_dir = ROOT / "runs" / args.out_dir
    (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "results.csv"
    # Outside the protocol's own directory, deliberately: the key is the
    # fingerprint of the knowledge traces, so two experiments mining from the
    # same traces share the objects even when they write different CSVs.
    artifact_cache = None if args.no_artifact_cache else ROOT / "runs" / "_artifacts"

    done: set[tuple] = set()
    if results_csv.exists():
        with results_csv.open() as handle:
            reader = csv.DictReader(handle)
            header = tuple(reader.fieldnames or ())
            for row in reader:
                done.add((row["dataset"], row["arch"], row["variant"],
                          float(row["noise"]), int(row["seed"])))
        # A CSV written with different columns cannot be extended by append: the
        # new rows would be misaligned with the header.
        if header != CSV_FIELDS:
            print(f"ERROR: {results_csv} has a header different from the "
                  f"expected one.\n"
                  f"  missing: {[f for f in CSV_FIELDS if f not in header]}\n"
                  f"  Use --out-dir with a fresh directory.")
            raise SystemExit(1)
        print(f"resume: {len(done)} cells already in the CSV, they are skipped")
    else:
        with results_csv.open("w", newline="") as handle:
            csv.writer(handle).writerow(CSV_FIELDS)

    # The protocol is ONLY a choice of config: one field of DataConfig. The
    # rest of the pipeline does not know which of the two is running.
    base_config = temporal_protocol(
        validation_fraction=args.val_fraction, test_fraction=args.test_fraction
    )
    if args.protocol == "train":
        # The knowledge comes from the training partition instead of the test
        # one, and nothing else changes. ``temporal_protocol`` DEFINES the
        # protocol, so the change is made afterwards, on a single field.
        base_config = replace(
            base_config, data=replace(base_config.data, knowledge_source="train"))
    base_config = replace(
        base_config,
        training=replace(base_config.training, logic_weight=LOGIC_WEIGHT,
                         epochs=MAX_EPOCHS),
        model=replace(base_config.model, hidden_dim=RECURRENT_HIDDEN,
                      recurrent_layers=RECURRENT_LAYERS))
    data = base_config.data
    device = resolve_device(base_config.training.device)
    # ``cpu`` when there is no card: a value like any other, and it says by
    # itself why that row has seconds out of scale with the others.
    gpu_name = (torch.cuda.get_device_name(device)
                if getattr(device, "type", str(device)) == "cuda" else "cpu")
    families = {VARIANT_SPEC[v][2] for v in variants}
    # The evaluation mask is always needed: it is the one forbidden mass and
    # violation rate are measured on for EVERY variant.
    masks_needed = {VARIANT_SPEC[v][3] for v in variants} | {EVAL_MASK}
    # The net mask is needed to EVALUATE all nine variants, not only by those
    # that use it in the loss: it is the primary yardstick and has to be the
    # same object on every row, baseline included.
    if not args.no_net_eval:
        masks_needed |= {"net"}

    total = len(args.datasets) * len(args.archs) * len(variants) * len(args.noises) * len(args.seeds)
    print(f"protocol {args.protocol}: split={data.split_strategy} "
          f"{data.train_fraction:.2f}/{data.validation_fraction:.2f}/{data.test_fraction:.2f}, "
          f"vocabulary={data.vocabulary_scope}, knowledge={data.knowledge_source}, "
          f"noise={data.noise_model}\n"
          f"suffix prediction: {'OFF' if args.no_suffix else 'ON'} | output: {out_dir}\n"
          f"grid: {len(args.datasets)} datasets x {len(args.archs)} archs x "
          f"{len(variants)} variants x {len(args.noises)} noise levels x {len(args.seeds)} seeds "
          f"= {total} runs ({len(done)} already done)", flush=True)
    completed = 0

    for dataset in args.datasets:
        pending = [c for c in (
            (dataset, arch, variant, noise, seed)
            for arch in args.archs for variant in variants
            for noise in args.noises for seed in args.seeds
        ) if c not in done]
        if not pending:
            print(f"[{dataset}] complete, skipped")
            continue
        # Under event noise, preparing a level costs as much as training all its
        # variants. A resumed grid must not pay it again for the closed levels,
        # so the (noise, seed) pairs still to do are decided before touching the logs.
        pending_levels = {(cell[3], cell[4]) for cell in pending}

        art = build_dataset_artifacts(dataset, base_config, families, masks_needed,
                                      cache_dir=artifact_cache)
        vocab, petrinet = art["vocab"], art["petrinet"]
        n_places = len(petrinet.places)

        # Two noise regimes. ``event`` corrupts the TRACES, so the logs are
        # built once per (noise, seed) and shared by every variant of the cell.
        # ``target`` corrupts only the labels, so the logs are built ONCE per dataset.
        event_noise = base_config.data.noise_model == "event"
        noisy_cache: dict = {}
        clean_logs = None if event_noise else art["build_logs"](art["train_traces"])
        clean_ratio_train = compliance_ratio(art["constraints"],
                                             art["train_traces"].values())

        def noisy_logs(noise: float, seed: int):
            if not event_noise:
                rng = random.Random(seed + int(noise * 1000))
                logs, changed = {}, 0
                for family, log in clean_logs.items():
                    logs[family], changed = log.corrupt_targets(noise, rng)
                return logs, changed, clean_ratio_train

            key = ("clean",) if noise == 0.0 else (noise, seed)
            if key not in noisy_cache:
                rng = random.Random(seed + int(noise * 1000))
                traces, changed = TraceUtils.corrupt_traces(
                    art["train_traces"], noise, rng, art["vocab"].activities
                )
                ratio = compliance_ratio(art["constraints"], traces.values())
                print(f"[{dataset}] noise {noise:.2f} seed {seed}: {changed} corrupted events, "
                      f"training compliance {ratio:.3f} -- rebuilding the markings...", flush=True)
                noisy_cache.clear()  # one key at a time: the logs are large
                noisy_cache[key] = (art["build_logs"](traces), changed, ratio)
            return noisy_cache[key]

        for seed in args.seeds:
            seed_config = replace(base_config, seed=seed)
            for noise in args.noises:
                if (noise, seed) not in pending_levels:
                    print(f"[{dataset}] noise {noise:.2f} seed {seed}: already complete, "
                          f"the rebuild is skipped", flush=True)
                    continue
                # twin-run: inside a cell (dataset, seed, noise) every variant
                # sees THE SAME corrupted log
                train_logs, corrupted, ratio = noisy_logs(noise, seed)
                for arch in args.archs:
                    for variant in variants:
                        cell = (dataset, arch, variant, noise, seed)
                        if cell in done:
                            continue
                        suffix, use_logic, family, mask_key = VARIANT_SPEC[variant]
                        kind = arch + suffix
                        train_loader = train_logs[family].data_loader(data.batch_size, shuffle=True)

                        started = time.perf_counter()
                        train_res = train_model(
                            model_kind=kind,
                            vocabulary=vocab,
                            train_loader=train_loader,
                            validation_loader=art["eval_loaders"]["val"][family],
                            config=seed_config,
                            checkpoint_path=out_dir / "ckpt" /
                                f"{dataset}_{kind}_{variant}_n{int(noise*100)}_s{seed}.pt",
                            use_logic=use_logic,
                            # Which knowledge enters the loss, and how.
                            **loss_arguments(mask_key, art),
                            # The marking is a FEATURE: it depends on the encoder
                            # (the kind suffix), not on the data family.
                            marking_dim=n_places if suffix else 0,
                            adjacency=petrinet.adjacency_matrices if suffix in GRAPH_SUFFIXES else None,
                        )
                        secs_train = time.perf_counter() - started
                        secs_per_epoch = secs_train / max(len(train_res.history), 1)

                        # --- task 1: next activity
                        eval_res = evaluate_model(
                            model=train_res.model,
                            data_loader=art["eval_loaders"]["test"][family],
                            # The empirical mask, the same on every row: the
                            # yardstick that always exists.
                            allowed_mask=art["masks"][EVAL_MASK].to(device),
                            device=device,
                        )
                        # The same model against the net automaton: a second pass
                        # over the test set, not a second training.
                        if "net" in art["masks"]:
                            eval_net = evaluate_model(
                                model=train_res.model,
                                data_loader=art["eval_loaders"]["test"][family],
                                allowed_mask=art["masks"]["net"].to(device),
                                device=device,
                            )
                            net_cells = [f"{eval_net.violation_rate:.6f}",
                                         f"{eval_net.forbidden_mass:.6f}"]
                        else:
                            net_cells = ["", ""]

                        # --- task 2: suffix prediction. FREE decoding: a mask
                        # would zero the violations by construction. The marking
                        # variants need the net to replay their own predictions.
                        if args.no_suffix:
                            suffix_cells = ["", "", "", "", "", "", "", "", ""]
                        else:
                            sfx = evaluate_suffix_prediction(
                                train_res.model, art["test_traces"], vocab,
                                art["automaton"], device,
                                prefix_lengths=art["prefix_lengths"],
                                constraints=art["constraints"],
                                petrinet=petrinet if suffix else None,
                                # Fitness is measured for EVERY variant: it is a
                                # property of the trace produced.
                                fitness_net=petrinet,
                                # The second yardstick on the same generated traces.
                                automaton_net=art["net_dfa"],
                            )
                            net_violation = ("" if art["net_dfa"] is None
                                             else f"{sfx.dfa_violation_rate_net:.6f}")
                            suffix_cells = [
                                f"{sfx.dl_similarity:.6f}", f"{sfx.activity_accuracy:.6f}",
                                f"{sfx.exact_match_rate:.6f}", f"{sfx.dfa_violation_rate:.6f}",
                                net_violation,
                                f"{sfx.precedence_violation_rate:.6f}",
                                f"{sfx.net_fitness:.6f}", sfx.n_examples,
                                "|".join(str(k) for k in art["prefix_lengths"]),
                            ]

                        with results_csv.open("a", newline="") as handle:
                            csv.writer(handle).writerow([
                                dataset, arch, variant, noise, seed,
                                f"{eval_res.accuracy:.6f}", f"{eval_res.macro_f1:.6f}",
                                f"{eval_res.top_k_accuracy:.6f}", f"{eval_res.violation_rate:.6f}",
                                f"{eval_res.forbidden_mass:.6f}",
                                *net_cells,
                                *suffix_cells,
                                train_res.best_epoch, corrupted, f"{ratio:.6f}",
                                f"{secs_per_epoch:.2f}",
                                f"{secs_train:.2f}",
                                f"{art['timings']['petrinet']:.2f}",
                                f"{art['timings']['dfa']:.2f}",
                                f"{art['timings']['automaton']:.2f}",
                                f"{art['timings']['markings']:.2f}",
                                f"{art['timings']['artifacts']:.2f}",
                                gpu_name,
                                art["knowledge_key"], art["net_fingerprint"],
                                int(art["cached"]),
                                data.split_strategy, data.knowledge_source, data.vocabulary_scope,
                                f"{data.train_fraction:.2f}", f"{data.validation_fraction:.2f}",
                                f"{data.test_fraction:.2f}",
                            ])
                        done.add(cell)
                        completed += 1
                        dl_note = "" if args.no_suffix else f" DL {sfx.dl_similarity:.4f}"
                        print(f"[{dataset} | {arch} {variant:<15} | noise {noise:.2f} | seed {seed}] "
                              f"acc {eval_res.accuracy:.4f} forbidden {eval_res.forbidden_mass:.4f}"
                              f"{dl_note} ({completed}/{total})", flush=True)

    # ---- summary from the complete CSV (earlier launches included) ----------
    with results_csv.open() as handle:
        rows = list(csv.DictReader(handle))

    def summarise(column: str, title: str) -> None:
        print(f"\n================ {title} (mean over seeds) ================")
        for dataset in args.datasets:
            for noise in args.noises:
                for arch in args.archs:
                    cells = [r for r in rows if r["dataset"] == dataset
                             and r["arch"] == arch and float(r["noise"]) == noise]
                    parts = []
                    for variant in variants:
                        vals = [float(r[column]) for r in cells
                                if r["variant"] == variant and r.get(column)]
                        if vals:
                            spread = stdev(vals) if len(vals) > 1 else 0.0
                            parts.append(f"{variant} {mean(vals):.4f}±{spread:.4f} (n={len(vals)})")
                    if parts:
                        print(f"{dataset} | {arch} | noise {noise:.2f}: " + " | ".join(parts))

    summarise("accuracy", "NEXT ACTIVITY — accuracy")
    summarise("dl_similarity", "SUFFIX PREDICTION — normalised Damerau-Levenshtein")


if __name__ == "__main__":
    main()
