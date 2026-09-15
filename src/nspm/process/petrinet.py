from __future__ import annotations
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Mapping, Sequence, Iterable

import pm4py
from pm4py.objects.log.obj import Event, EventLog, Trace
from pm4py.objects.petri_net.obj import PetriNet as Pm4pyNet, Marking
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

from ..data.loader import ACTIVITY

AdjacencyMatrix = tuple[tuple[int, ...], ...]

_REPLAY_PARAM = token_replay.Variants.TOKEN_REPLAY.value.Parameters
REPLAY_PARAMETERS = {
    _REPLAY_PARAM.STOP_IMMEDIATELY_UNFIT: False,
    _REPLAY_PARAM.WALK_THROUGH_HIDDEN_TRANS: True,
    _REPLAY_PARAM.SHOW_PROGRESS_BAR: False
}

#: Below this threshold the pool costs more than it returns: on Windows every
#: worker starts with ``spawn`` and re-imports pm4py, which takes seconds.
PARALLEL_THRESHOLD = 2000

#: Prefixes per task. It only amortises the cost of IPC: inside the task the
#: prefixes are replayed ONE BY ONE anyway (see ``_replay_chunk``), so the size
#: does not change the values. Measured irrelevant between 4 and 64; 64 keeps
#: the number of round-trips low without unbalancing the load.
CHUNK_SIZE = 64


# How many processes to use for the replay
def replay_workers(workers: int | None = None) -> int:

    if workers is not None:
        return max(1, workers)
    override = os.environ.get("NSPM_REPLAY_WORKERS")
    if override:
        return max(1, int(override))
    return max(1, (os.cpu_count() or 2) // 2)


#: Per-worker state. The net arrives once at process start-up (50 KB, one
#: millisecond) instead of with every task.
_WORKER: dict = {}


def _init_replay_worker(payload) -> None:
    network, init_marking, final_marking, places = payload
    _WORKER["network"] = network
    _WORKER["init"] = init_marking
    _WORKER["final"] = final_marking
    # Places survive the pickle as the same objects that live inside the net,
    # because they are serialised along with it: the index stays valid.
    _WORKER["index"] = {place: position for position, place in enumerate(places)}
    _WORKER["size"] = len(places)


# Replay a chunk of prefixes, ONE PER CALL
def _replay_chunk(prefixes: Sequence[tuple[str, ...]]) -> list[tuple[int, ...]]:

    network, init = _WORKER["network"], _WORKER["init"]
    final, index, size = _WORKER["final"], _WORKER["index"], _WORKER["size"]

    markings = []
    for prefix in prefixes:
        log = PetriNet._create_event_log([prefix])
        result = token_replay.apply(log, network, init, final,
                                    parameters=REPLAY_PARAMETERS)[0]
        vector = [0] * size
        for place, tokens in result["reached_marking"].items():
            vector[index[place]] = tokens
        markings.append(tuple(vector))
    return markings

#: Recursion limit for discovery alone, and the stack of the thread hosting it.
#: The pm4py inductive miner is recursive -- it cuts the log, recurses on the
#: pieces, and ends with a ``deepcopy`` of the tree, which recurses in turn. On
#: BPIC15 under protocol A the tree is deep enough to break the default limit
#: (1000) and the process dies with ``RecursionError`` before writing a row.
#: Under B it did not happen, because there the knowledge comes from the test
#: partition, a quarter of the traces.
#:
#: The sizes are tried in order and the first accepted one is kept: the maximum
#: is not the same everywhere -- Linux takes 256 MB, Windows refuses it with
#: ``ValueError``. The trailing zero is the system default, that is no increase:
#: if it gets that far the ``RecursionError`` can come back, but at least it is
#: a readable exception and not a thread that fails to start.
_DISCOVERY_RECURSION_LIMIT = 50_000
_DISCOVERY_STACK_SIZES = (256 * 1024 * 1024, 128 * 1024 * 1024,
                          64 * 1024 * 1024, 32 * 1024 * 1024, 0)


# ``discover_petri_net_inductive`` with enough stack to finish
def _discover_inductive(event_log, noise_threshold: float):

    outcome: dict = {}

    def discover() -> None:
        previous_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(_DISCOVERY_RECURSION_LIMIT)
        try:
            outcome["net"] = pm4py.discover_petri_net_inductive(
                event_log, noise_threshold=noise_threshold)
        except BaseException as error:          # re-raised in the caller: without
            outcome["error"] = error            # this the thread would die mute
        finally:
            sys.setrecursionlimit(previous_limit)

    previous_stack = threading.stack_size()
    try:
        for size in _DISCOVERY_STACK_SIZES:
            try:
                threading.stack_size(size)
            except (ValueError, RuntimeError):
                continue
            break
        worker = threading.Thread(target=discover)
        worker.start()
        worker.join()
    finally:
        threading.stack_size(previous_stack)

    if "error" in outcome:
        raise outcome["error"]
    return outcome["net"]


@dataclass(frozen=True)
class PetriNet:

    network: Pm4pyNet
    init_marking: Marking
    final_marking: Marking
    places: tuple[Pm4pyNet.Place, ...]
    transitions: tuple[Pm4pyNet.Transition, ...]
    #: Memoization of ``prefix_marking``. It is a pure function of the prefix --
    #: the net does not change -- and prefixes repeat a great deal: 4x on BPIC12,
    #: 119x on BPIC20. Without the cache the same marking is replayed once for
    #: ``with_markings`` and once for ``with_marking_sequences``, which ask for
    #: exactly the same keys. This is not an approximation: the values come out
    #: identical, only the number of calls to pm4py changes.
    _marking_cache: dict[tuple[str, ...], tuple[int, ...]] = field(
        default_factory=dict, compare=False, repr=False
    )

    @classmethod
    def from_traces(cls, traces: Mapping[str, Sequence[str]], noise_threshold: float =0.2) -> PetriNet:

        event_log = cls._create_event_log(traces.values())
        network, init_marking, final_marking = _discover_inductive(event_log, noise_threshold)
        places = tuple(sorted(network.places, key=lambda p: p.name))
        transitions = tuple(sorted(network.transitions, key=lambda t: (t.label is None, t.label or "", str(t))))

        return cls(network, init_marking, final_marking, places, transitions)

    def prefix_marking(self, prefix: Sequence[str]) -> tuple[int, ...]:

        key = tuple(prefix)
        cached = self._marking_cache.get(key)
        if cached is not None:
            return cached

        prefix_log = self._create_event_log([prefix])
        result = token_replay.apply(prefix_log, self.network, self.init_marking, self.final_marking, parameters=REPLAY_PARAMETERS)[0]

        index = self.place_index
        vector = [0] * len(self.places)
        for place, tokens in result["reached_marking"].items():
            vector[index[place]] = tokens

        marking = tuple(vector)
        self._marking_cache[key] = marking
        return marking

    # Controparte parallela di :meth:`prefix_marking`, su molti prefissi
    def prefix_markings(self, prefixes: Iterable[Sequence[str]],
                        workers: int | None = None,
                        chunk_size: int = CHUNK_SIZE) -> list[tuple[int, ...]]:

        keys = [tuple(prefix) for prefix in prefixes]
        pending, seen = [], set()
        for key in keys:
            if key not in self._marking_cache and key not in seen:
                seen.add(key)
                pending.append(key)

        if pending:
            processes = replay_workers(workers)
            if processes > 1 and len(pending) >= PARALLEL_THRESHOLD:
                payload = (self.network, self.init_marking, self.final_marking,
                           self.places)
                chunks = [pending[start:start + chunk_size]
                          for start in range(0, len(pending), chunk_size)]
                with ProcessPoolExecutor(max_workers=processes,
                                         initializer=_init_replay_worker,
                                         initargs=(payload,)) as pool:
                    for chunk, markings in zip(chunks, pool.map(_replay_chunk, chunks)):
                        self._marking_cache.update(zip(chunk, markings))
            else:
                for key in pending:
                    self.prefix_marking(key)

        return [self._marking_cache[key] for key in keys]

    # Token-based-replay fitness of whole traces against this net
    def trace_fitness(self, traces: Iterable[Sequence[str]]) -> list[float]:

        traces = list(traces)
        if not traces:
            return []
        diagnostics = pm4py.conformance_diagnostics_token_based_replay(
            self._create_event_log(traces), self.network,
            self.init_marking, self.final_marking,
            # The same replay configuration that produces the markings: it
            # crosses invisible transitions and does not stop at the first
            # mismatch. A different configuration here would measure a different
            # net from the one the rest of the thesis describes.
            opt_parameters=dict(REPLAY_PARAMETERS),
        )
        return [float(entry["trace_fitness"]) for entry in diagnostics]

    def marking_sequence(self, prefix: Sequence[str]) -> tuple[tuple[int, ...], ...]:
        mark_seq = []
        for i in range(1, len(prefix) + 1):
             mark_seq.append(self.prefix_marking(prefix[:i]))
        return tuple(mark_seq)

    @staticmethod
    def _create_event_log(traces: Iterable[Sequence[str]]) -> EventLog:
        event_log = EventLog()
        for trace in traces:
            pm_trace = Trace()
            for activity in trace:
                pm_trace.append(Event({ACTIVITY: activity}))
            event_log.append(pm_trace)
        return event_log

    @property
    def place_index(self) -> dict[Pm4pyNet.Place, int]:
        return {place: index for index, place in enumerate(self.places)}

    @property
    def transition_index(self) -> dict[Pm4pyNet.Transition, int]:
        return {transition: index for index, transition in enumerate(self.transitions)}

    @property
    def adjacency_matrices(self) -> tuple[AdjacencyMatrix, AdjacencyMatrix]:

        place_idx = self.place_index
        trans_idx = self.transition_index

        a_pt = ([[0] * len(trans_idx) for _ in range(len(place_idx))])
        a_tp = ([[0] * len(place_idx) for _ in range(len(trans_idx))])

        for arc in self.network.arcs:
            if isinstance(arc.source, Pm4pyNet.Place):
                a_pt[place_idx[arc.source]][trans_idx[arc.target]] = 1
            else:
                a_tp[trans_idx[arc.source]][place_idx[arc.target]] = 1

        return tuple(tuple(row) for row in a_pt), tuple(tuple(row) for row in a_tp)