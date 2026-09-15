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
from typing import get_args

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nspm.config import DataConfig, ExperimentConfig, temporal_protocol
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
from nspm.learning.models import GRAPH_SUFFIXES, ModelKind
from nspm.learning.training import train_model, resolve_device
from nspm.learning.logic import build_allowed_mask, build_state_mask
from nspm.learning.evaluation import evaluate_model
from nspm.learning.trace_prediction import evaluate_suffix_prediction

DATASETS = ("Sepsis_Case", "BPIC_2013_incidents", "BPIC_2020_DomesticDeclarations")
#: Allowed but NOT in the default set: the frozen grid is 1260 rows over the
#: three logs above, and a launch without arguments has to keep reproducing it.
#: These two are asked for by name.
EXTRA_DATASETS = ("BPI_Challenge_2012", "BPI_Challenge_2015_Municipality")
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
# The axes are independent: the suffix picks the symbolic encoder (feature), the
# boolean turns the loss term on, the last field picks which automaton the mask
# the loss penalises comes from.
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
    # The two methods of Mezini et al. (nesy-suffix-prediction-dfa). Same trunk
    # as the baseline and same structure as ``checker_net``, that is the
    # automaton extracted from the Petri net: only HOW the constraint enters the
    # loss changes, which is the one thing the experiment wants to compare. The
    # direct term of comparison is ``checker_net``, not ``checker``.
    "lll": ("", True, "plain", "axel_local"),
    "gll": ("", True, "plain", "axel_global"),
}

#: Hyperparameters of the two losses of Mezini et al. They sweep ten values of
#: alpha over fifteen runs; here the grid has a single seed, so the two values
#: are chosen by hand and declared.
#:
#: In both losses ``alpha`` weighs the supervision and ``1 - alpha`` the logic,
#: so the two methods run on opposite blends: the local one at 75% logic, the
#: global one at 75% cross-entropy. One value per method, not a single one: the
#: local penalty is a probability mass at one step, the global one the ``-log``
#: of the acceptance of a whole rollout, and the two are not on the same scale.
#:
#: ``AXEL_SAMPLES`` is their full value. The rollout costs ``horizon``
#: sequential steps over a batch replicated as many times, so ten samples did
#: not fit on a 16 GB card; the grid runs on 80 GB A100s, where they do. It has
#: to be kept THE SAME on every cell: with a different number of samples the
#: estimate of the acceptance has a different variance, and the cells would no
#: longer be comparable with each other.
AXEL_ALPHA_LOCAL = 0.25
AXEL_ALPHA_GLOBAL = 0.75
AXEL_TEMPERATURE = 0.5
AXEL_SAMPLES = 10

#: Size of the recurrent trunk. These are the numbers of Mezini et al. (2026):
#: two layers of a hundred units, Adam, batches of 64. The compact baseline of
#: this repo was one layer of 64, which is enough for a GRU and not for an LSTM:
#: the whole advantage of the memory cell is in being able to stack it.
#:
#: It holds for ALL NINE variants, not only for the two taken from them. The
#: cell-by-cell comparison stands only if the trunk is the same: if their two
#: methods ran on their trunk and our seven on ours, the grid would measure the
#: difference between architectures instead of between knowledge channels.
RECURRENT_HIDDEN = 100
RECURRENT_LAYERS = 2

#: Ceiling on the epochs. Twelve were too few: on the GRU grid a third of the
#: cells stopped at eleven or twelve, that is because the budget ran out and not
#: because they had converged, and a larger network asks for more, not fewer.
#: Mezini et al. reach 590-1607 epochs.
#:
#: The ceiling is not the cost: early stopping is unchanged, so the cells that
#: already converged in four epochs keep stopping there. Raising it pays only
#: where it was needed.
MAX_EPOCHS = 60

#: The two automata the loss mask can come from. ``dfa`` is the empirical
#: directly-follows relation counted on the knowledge traces; ``net`` is the
#: reachability automaton of the Petri net discovered from those same traces,
#: projected onto directly-follows form. Same source information, different
#: channel: the net generalises, so its mask is wider.
#:
#: ``net_state`` is the same net WITHOUT the projection: one row per automaton
#: state instead of per activity, indexed by replaying the prefix. It is the
#: only way the memory of the markings reaches the loss.
#:
#: Every variant is EVALUATED on ``dfa`` -- including the suffix metrics, which
#: use ``art["automaton"]``. If each variant measured conformance against its
#: own automaton, the columns would no longer be comparable across rows.
MASK_SPEC = ("dfa", "net", "net_state")
EVAL_MASK = "dfa"

#: Who needs the structure extracted from the Petri net.
#:
#: The criterion is the supervisor's: two methods are comparable only if they
#: see THE SAME symbolic object. Whoever uses the net uses the same net, whoever
#: uses the DFA uses the same DFA -- otherwise the difference between two cells
#: measures which structure the knowledge comes from instead of how that
#: knowledge enters the loss, which is the one thing the experiment isolates.
#:
#: ``lll`` and ``gll`` sit on this side together with marking, gnn, seq and the
#: two ``checker_net``: their knowledge is the net, projected onto
#: directly-follows for the local loss and tensorised for the global one. The
#: price is that they now depend on the reachability automaton, so on a net with
#: too many silent transitions they fall with it.
NET_DFA_USERS = frozenset({"net", "axel_local", "axel_global"})
#: Who has the automaton built. The per-state mask does not go through the
#: projection, so it asks for the automaton and nothing else.
AUTOMATON_USERS = NET_DFA_USERS | {"net_state"}

CSV_FIELDS = (
    "dataset", "arch", "variant", "noise", "seed",
    # --- task 1: next activity, one step from the true prefix.
    #     Every conformance metric exists in TWO copies: the same definition,
    #     computed against two different objects. Without a suffix it is the
    #     empirical directly-follows relation, mined from the traces; with
    #     ``_net`` it is the reachability automaton projected from the Petri net.
    #     Both hold for all nine variants, baseline included: the yardstick has
    #     to be the same on every row, or the rows do not compare.
    #
    #     They are not redundant, and that is a measured fact and not a
    #     precaution: on Sepsis the net admits 199 cells and the empirical DFA
    #     159, but the two masks are NOT nested -- the inductive miner with
    #     ``noise_threshold=0.2`` filters infrequent behaviour, so the net also
    #     forbids pairs that really are in the log. A method trained against the
    #     net and measured with the empirical yardstick has the very
    #     generalisations it was asked to make counted as violations, and the
    #     other way round.
    #
    #     The ``_net`` columns stay empty where the automaton cannot be built.
    #     The others are always there, and that is the only reason such a dataset
    #     stays measurable.
    "accuracy", "macro_f1", "top3", "violation_rate", "forbidden",
    "violation_rate_net", "forbidden_net",
    # --- task 2: suffix prediction, the model regenerates from its own output.
    #     The same pair of yardsticks on the same generated traces: counting them
    #     twice is a pass of string comparisons, not a second inference.
    "dl_similarity", "suffix_accuracy", "exact_match",
    "suffix_dfa_violation", "suffix_dfa_violation_net",
    "suffix_precedence_violation",
    # Token-replay fitness of the WHOLE trace (real prefix + generated suffix)
    # against the Petri net: the only conformance measured on the entire trace
    # instead of on the single transition.
    "suffix_net_fitness",
    "n_suffix", "prefix_lengths",
    # --- noise, cost and provenance. ``corrupted`` counts the EVENTS whose
    #     label was replaced; ``train_compliance`` is the fraction of training
    #     traces still satisfying the constraints after the injection -- Table 2
    #     of the paper, without which the noise axis cannot be read.
    "best_epoch", "corrupted", "train_compliance", "secs_per_epoch",
    # --- cost. ``secs_train`` is the wall time of training, ``secs_per_epoch``
    #     the same divided by the epochs actually run. The other four are the
    #     symbolic part, which the neural network does not pay: discovery of the
    #     Petri net, the DFA with its tensorisation, the reachability automaton,
    #     and the token replay of the held-out partitions. ``secs_artifacts``
    #     holds all of them plus reading the log.
    "secs_train", "secs_petrinet", "secs_dfa", "secs_automaton",
    "secs_markings", "secs_artifacts",
    # Which card the cell ran on. Without this column the seconds cannot be
    # read: one job is one dataset and one node, so the nine variants of a given
    # dataset are always comparable with each other, but two datasets that ended
    # up on different cards are not.
    "gpu",
    # --- reproducibility. ``knowledge_key`` is the fingerprint of the traces
    #     mined from, ``net_fingerprint`` that of the net that came out. They
    #     exist so that this can be VERIFIED and not merely hoped: two rows with
    #     the same pair saw the same symbolic object, and a future reconstruction
    #     is compared against what is written here. Without them, a change of net
    #     is invisible until it shows up in the results. ``artifacts_cached``
    #     says whether the objects were read back from disk in that run: when it
    #     is 1, the seconds of the symbolic part are those of the original
    #     construction, not of this run.
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
    raise FileNotFoundError(f"No .xes o .csv log in {folder}")


# Everything derived from a log, built once per dataset
def build_dataset_artifacts(dataset: str, config, families: set[str],
                            masks_needed: set[str] = frozenset({"dfa"}),
                            cache_dir: Path | None = None) -> dict:
    print(f"[{dataset}] caricamento e costruzione artefatti...", flush=True)
    # The times of the symbolic part, timed piece by piece and reported in the
    # CSV beside those of training. They exist to say what the knowledge costs,
    # which is a different question from what the neural network costs: under the
    # event-noise protocols these seconds are paid again at every noise level,
    # because the traces change and the knowledge has to be rediscovered.
    timings = {"petrinet": 0.0, "dfa": 0.0, "automaton": 0.0,
               "markings": 0.0, "artifacts": 0.0}
    build_started = time.perf_counter()

    events = read_log(find_log(dataset))
    splits = build_splits(events, config)

    # Under ``vocabulary_scope="train"`` the alphabet comes from the training
    # block alone, so held-out cases with activities never seen are neither
    # predictable nor scoreable and have to be dropped -- the convention of
    # protocol A. Under ``"all"`` nothing is dropped and the set is empty by
    # construction.
    if config.data.vocabulary_scope == "train":
        known = {a for trace in splits.train.values() for a in trace}

        def keep(partition):
            kept = {cid: t for cid, t in partition.items()
                    if all(a in known for a in t)}
            return kept, len(partition) - len(kept)

        validation, dropped_val = keep(splits.validation)
        test, dropped_test = keep(splits.test)
        if dropped_val or dropped_test:
            print(f"[{dataset}] scartati {dropped_val} case di validation / "
                  f"{dropped_test} di test con attivita' mai viste nel train",
                  flush=True)
        splits = replace(splits, validation=validation, test=test)

    vocab = build_vocabulary(splits, config)
    unseen = unseen_activities(splits, vocab)
    if unseen:
        raise ValueError(
            f"[{dataset}] attivita' non rappresentabili nelle partizioni held-out: "
            f"{sorted(unseen)}. Il protocollo temporale non scarta tracce, quindi "
            f"il vocabolario deve coprirle: controlla data.vocabulary_scope."
        )

    # Background knowledge dalla partizione indicata da ``knowledge_source``
    # (test, in questo protocollo).
    knowledge = knowledge_traces(splits, config)

    # The mined objects come from disk when they are already there. The key is
    # the fingerprint of the knowledge traces, so B and C of the same dataset do
    # not touch: they mine from different partitions and have different keys.
    # This is what makes it possible to RELOAD a run -- re-score the checkpoints,
    # redo the figures, reproduce a number -- which without the saved objects is
    # not possible, because the inductive miner generates fresh names at every
    # call and the marking vectors depend on the order of the places.
    store_key = artifact_store.knowledge_fingerprint(knowledge, noise_threshold=0.2)
    store_file = artifact_store.cache_path(cache_dir, dataset, store_key) \
        if cache_dir is not None else None
    cached = artifact_store.load(store_file) if store_file is not None else {}
    fresh = False

    started = time.perf_counter()
    if "petrinet" in cached:
        petrinet = cached["petrinet"]
        # The seconds are those of the REAL construction, read back from the
        # cache: the column has to say what discovering that net costs, not what
        # opening a file costs. If it reported zero, the cost of the symbolic
        # part -- one of the required metrics -- would vanish on the second
        # launch.
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
        print(f"[{dataset}] automa della rete (da cache): "
              f"{reachability.state_count} stati "
              f"({len(reachability.accepting)} accettanti)", flush=True)
    elif masks_needed & AUTOMATON_USERS:
        # The same net as the marking/gnn/seq variants -- the same object, not a
        # reconstruction. ``alphabet`` keeps in the activities pruned by the
        # inductive miner: about their position the net says nothing, and saying
        # nothing is not the same as forbidding them.
        reachability = ReachabilityAutomaton.from_process_net(
            petrinet, alphabet=vocab.activities
        )
        timings["automaton"] = time.perf_counter() - started
        fresh = True
        print(f"[{dataset}] automa della rete: {reachability.state_count} stati "
              f"({len(reachability.accepting)} accettanti)", flush=True)
    if masks_needed & NET_DFA_USERS:
        # Projection onto directly-follows: it loses the memory of the markings,
        # but it is the only shape a mask on the last token accepts.
        net_dfa = reachability.to_process_dfa()
        masks["net"] = build_allowed_mask(net_dfa, vocab)
        print(f"[{dataset}] proiettato: {net_dfa.transition_count} transizioni | "
              f"celle ammesse: net {int(masks['net'].sum())} vs empirico "
              f"{int(mask.sum())} su {mask.numel()}", flush=True)
    if "net_state" in masks_needed:
        # No projection: one row per automaton state, the memory survives.
        state_mask = build_state_mask(reachability, vocab)
        print(f"[{dataset}] maschera per stato: {tuple(state_mask.shape)} "
              f"({int(state_mask.sum())} celle ammesse)", flush=True)
    # The precedence constraints give conformance at TRACE level:
    # dfa_violation_rate counts the individual directly-follows steps, this says
    # whether the generated trace as a whole breaks an ordering rule.
    #
    # These are saved too: ``mine_precedence_constraints`` has a known
    # nondeterminism across processes, so two runs can mine different
    # constraints from the same traces and the violation column would not be
    # comparable between them.
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

    # Saved only if something was mined just now. The cache grows by addition: a
    # launch without the automaton writes net, DFA and constraints, and the next
    # launch that does ask for the automaton adds it without redoing the rest.
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

    # The global loss needs the automaton in tensor form, so that it can be
    # traversed differentiably. It is the SAME projection ``checker_net`` reads,
    # only written as transition matrices: same net, same automaton, same
    # projection onto directly-follows.
    tensor_dfa = gll_horizon = None
    started = time.perf_counter()
    if "axel_global" in masks_needed:
        tensor_dfa = TensorDFA.from_process_dfa(
            net_dfa, vocab.activities, resolve_device(config.training.device)
        )
        # How far the rollout runs on. The reference work goes to the end of the
        # longest trace plus a margin; here the training prefixes start around
        # half the median length, so the tail to generate is of the order of the
        # other half. Tying it to the same scale as the suffix task avoids paying
        # for steps no trace would use.
        gll_horizon = max(1, median_length // 2 + 2)
        print(f"[{dataset}] automa tensoriale: {tensor_dfa.n_states} stati x "
              f"{tensor_dfa.n_actions} azioni | rollout {gll_horizon} passi x "
              f"{AXEL_SAMPLES} campioni", flush=True)
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
    # Token replay of validation and test. The training one is not here:
    # ``logs`` is a closure called at every cell, because the noise changes the
    # training traces and the markings have to be replayed. This column therefore
    # measures the replay of the held-out part, the only one built once.
    timings["markings"] = time.perf_counter() - started

    variants_train = len({tuple(t) for t in splits.train.values()})
    clean_ratio = compliance_ratio(constraints, splits.train.values())
    print(f"[{dataset}] {len(vocab.activities)} attivita, {len(petrinet.places)} posti\n"
          f"[{dataset}] split {config.data.split_strategy} "
          f"{len(splits.train)}/{len(splits.validation)}/"
          f"{len(splits.test)} case (duplicati tenuti: {variants_train} varianti distinte "
          f"su {len(splits.train)} tracce di train)\n"
          f"[{dataset}] knowledge da {config.data.knowledge_source}: "
          f"{len(knowledge)} tracce -> DFA {len(automaton.states)} stati/"
          f"{automaton.transition_count} transizioni, {len(constraints)} vincoli\n"
          f"[{dataset}] conformita' del train pulito: {clean_ratio:.3f}\n"
          f"[{dataset}] suffix: lunghezza mediana test {median_length}, "
          f"prefissi {prefix_lengths}", flush=True)
    timings["artifacts"] = time.perf_counter() - build_started
    print(f"[{dataset}] costruzione: rete {timings['petrinet']:.1f}s | "
          f"dfa {timings['dfa']:.1f}s | automa {timings['automaton']:.1f}s | "
          f"marking held-out {timings['markings']:.1f}s | "
          f"totale {timings['artifacts']:.1f}s", flush=True)
    return {
        "timings": timings,
        "vocab": vocab, "petrinet": petrinet, "mask": mask, "masks": masks,
        "reachability": reachability, "state_mask": state_mask,
        # The two fingerprints that end up in the CSV. ``knowledge_key`` says
        # which traces were mined from, ``net_fingerprint`` which net came out:
        # two rows with the same pair saw the same object, two rows with
        # different pairs did not. That is what makes a changed net noticeable,
        # rather than something discovered from the numbers.
        "knowledge_key": store_key,
        "net_fingerprint": artifact_store.net_fingerprint(petrinet),
        "cached": not fresh,
        # The net automaton projected onto directly-follows: it is the structure
        # the ``_net`` columns are measured against, and it is ``None`` only
        # where the net does not give a finite automaton.
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
        # The same mask as ``checker_net``: the knowledge entering the loss is
        # the projected Petri net, the same one marking, gnn and seq see.
        # ``allowed_mask`` stays the empirical one because it is the conformance
        # ``train_model`` writes into the history, not the constraint -- as for
        # ``checker_state``.
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
    parser.add_argument("--protocol", choices=("A", "B", "C"), default="B",
                        help="B: split temporale, rumore sugli eventi, knowledge "
                             "dal test. A: split random, rumore sui target, "
                             "knowledge dal train. Cambia solo il config.")
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

    # gate: seq stays out until the kinds exist in models.py
    variants = list(args.variants)
    if "seq" in variants and "gru_seq" not in get_args(ModelKind):
        print("NOTA: kind 'gru_seq' non ancora implementato -> variante 'seq' saltata.")
        variants.remove("seq")

    out_dir = ROOT / "runs" / args.out_dir
    (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "results.csv"
    # Outside the protocol's own directory, and deliberately so: the key is the
    # fingerprint of the knowledge traces, so two experiments mining from the
    # same traces share the objects even when they write different CSVs.
    # Deleting runs/noise_curve_* does not throw away an hour-long automaton.
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

    # The protocol is ONLY a choice of config: four fields of DataConfig. The
    # rest of the pipeline does not know which of the two is running.
    if args.protocol == "B":
        base_config = temporal_protocol(
            validation_fraction=args.val_fraction, test_fraction=args.test_fraction
        )
    elif args.protocol == "C":
        # B with the knowledge taken from the training partition instead of the
        # test one, and nothing else different: same temporal split, same
        # vocabulary, same noise on the events. It isolates the source of the
        # knowledge, which Mezini et al. declare to be the test set -- almost
        # certainly a slip, and this protocol measures it instead of arguing
        # about it.
        # ``temporal_protocol`` accepts only batch_size and num_workers as
        # overrides: it drops the other fields silently, and rightly so, because
        # it is the function that DEFINES the protocol. So the change is made
        # afterwards, in the open, on a single field.
        temporal = temporal_protocol(
            validation_fraction=args.val_fraction, test_fraction=args.test_fraction
        )
        base_config = replace(
            temporal, data=replace(temporal.data, knowledge_source="train"))
    else:
        base_config = ExperimentConfig(data=DataConfig(
            validation_fraction=args.val_fraction, test_fraction=args.test_fraction,
            split_strategy="random", vocabulary_scope="train",
            knowledge_source="train", noise_model="target",
        ))
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
    # that use it in the loss: it is the primary yardstick, and it has to be the
    # same object on every row. So it is always asked for, even when no variant
    # would use it -- for the baseline the number is informative anyway, it says
    # how much off-net behaviour a model that was never shown the net produces.
    if not args.no_net_eval:
        masks_needed |= {"net"}

    total = len(args.datasets) * len(args.archs) * len(variants) * len(args.noises) * len(args.seeds)
    print(f"protocollo {args.protocol}: split={data.split_strategy} "
          f"{data.train_fraction:.2f}/{data.validation_fraction:.2f}/{data.test_fraction:.2f}, "
          f"vocabolario={data.vocabulary_scope}, knowledge={data.knowledge_source}, "
          f"rumore={data.noise_model}\n"
          f"suffix prediction: {'OFF' if args.no_suffix else 'ON'} | output: {out_dir}\n"
          f"matrice: {len(args.datasets)} dataset x {len(args.archs)} arch x "
          f"{len(variants)} varianti x {len(args.noises)} noise x {len(args.seeds)} seed "
          f"= {total} run ({len(done)} gia' fatte)", flush=True)
    completed = 0

    for dataset in args.datasets:
        pending = [c for c in (
            (dataset, arch, variant, noise, seed)
            for arch in args.archs for variant in variants
            for noise in args.noises for seed in args.seeds
        ) if c not in done]
        if not pending:
            print(f"[{dataset}] completo, salto")
            continue
        # Under event noise, preparing a level costs as much as training all its
        # variants (the markings are replayed from scratch). Whoever resumes an
        # interrupted grid must not pay that cost again for the levels already
        # closed, so the (noise, seed) pairs still to do are decided here, before
        # touching the logs.
        pending_levels = {(cell[3], cell[4]) for cell in pending}

        art = build_dataset_artifacts(dataset, base_config, families, masks_needed,
                                      cache_dir=artifact_cache)
        vocab, petrinet = art["vocab"], art["petrinet"]
        n_places = len(petrinet.places)

        # Two noise regimes, and they change the COST too, not only the effect.
        #
        # ``event`` corrupts the TRACES, so the markings have to be replayed at
        # every level: the logs are built once per (noise, seed) and reused
        # across all the arch x variant combinations of the cell -- the twin-run
        # property. At noise 0 the traces are the clean ones, the same for every
        # seed, so once only.
        #
        # ``target`` leaves the traces intact and replaces only the label to be
        # predicted: the markings never change, so the logs are built ONCE per
        # dataset and the targets are corrupted on the copy. Training compliance
        # stays the clean one at every level -- which is exactly why this noise
        # bites so much less.
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
                print(f"[{dataset}] noise {noise:.2f} seed {seed}: {changed} eventi corrotti, "
                      f"conformita' del train {ratio:.3f} -- ricostruzione marking...", flush=True)
                noisy_cache.clear()  # one key at a time: the logs are large
                noisy_cache[key] = (art["build_logs"](traces), changed, ratio)
            return noisy_cache[key]

        for seed in args.seeds:
            seed_config = replace(base_config, seed=seed)
            for noise in args.noises:
                if (noise, seed) not in pending_levels:
                    print(f"[{dataset}] noise {noise:.2f} seed {seed}: gia' completo, "
                          f"salto la ricostruzione", flush=True)
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
                            # The marking is a FEATURE: it depends on the
                            # encoder (the kind suffix), not on the data family
                            # -- the "stated" family carries the states for the
                            # loss but the model stays the baseline's.
                            marking_dim=n_places if suffix else 0,
                            adjacency=petrinet.adjacency_matrices if suffix in GRAPH_SUFFIXES else None,
                        )
                        secs_train = time.perf_counter() - started
                        secs_per_epoch = secs_train / max(len(train_res.history), 1)

                        # --- task 1: next activity
                        eval_res = evaluate_model(
                            model=train_res.model,
                            data_loader=art["eval_loaders"]["test"][family],
                            # The empirical mask, the same on every row: it is
                            # the yardstick that always exists, even where the
                            # net automaton cannot be built.
                            allowed_mask=art["masks"][EVAL_MASK].to(device),
                            device=device,
                        )
                        # The same model against the net automaton. It is a
                        # second pass over the test set, not a second training:
                        # next to the minutes of training it is noise. Absent
                        # only when the net gives no automaton.
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

                        # --- task 2: suffix prediction. FREE decoding: the
                        # allowed_mask would zero the dfa_violation by
                        # construction, making conformance uninformative. The
                        # marking variants need the net to replay their own
                        # predictions.
                        if args.no_suffix:
                            suffix_cells = ["", "", "", "", "", "", "", "", ""]
                        else:
                            sfx = evaluate_suffix_prediction(
                                train_res.model, art["test_traces"], vocab,
                                art["automaton"], device,
                                prefix_lengths=art["prefix_lengths"],
                                constraints=art["constraints"],
                                petrinet=petrinet if suffix else None,
                                # Fitness is measured for EVERY variant: it is
                                # a property of the trace produced, not of the
                                # model that produced it.
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
        print(f"\n================ {title} (medie su seed) ================")
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
    summarise("dl_similarity", "SUFFIX PREDICTION — Damerau-Levenshtein normalizzata")


if __name__ == "__main__":
    main()
