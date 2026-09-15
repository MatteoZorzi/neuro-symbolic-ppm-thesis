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

#: Sotto questa soglia il pool costa piu' di quanto rende: su Windows ogni
#: worker parte con ``spawn`` e rifa' l'import di pm4py, che sono secondi.
PARALLEL_THRESHOLD = 2000

#: Prefissi per task. Serve solo ad ammortizzare il costo di IPC: dentro il
#: task i prefissi vengono comunque rigiocati UNO PER UNO (vedi
#: ``_replay_chunk``), quindi la dimensione non cambia i valori. Misurata
#: irrilevante fra 4 e 64; 64 tiene basso il numero di round-trip senza
#: sbilanciare il carico fra i worker.
CHUNK_SIZE = 64


def replay_workers(workers: int | None = None) -> int:
    """Quanti processi usare per il replay.

    Default: i core FISICI, cioe' meta' di quelli logici su una CPU con SMT.
    Non e' prudenza, e' misura: su un 7700X (8 core, 16 thread) e su 11k
    prefissi di BPIC15, 8 worker fanno 2.7x contro il seriale e 15 ne fanno
    1.8x. I sibling SMT si contendono le stesse unita' di esecuzione e il
    replay non ha abbastanza stalli di memoria da coprire lo scambio, per cui
    riempire i thread logici fa perdere tempo invece che guadagnarne.

    ``NSPM_REPLAY_WORKERS`` forza il valore; 1 disattiva il parallelismo e
    riporta il comportamento a quello seriale.
    """

    if workers is not None:
        return max(1, workers)
    override = os.environ.get("NSPM_REPLAY_WORKERS")
    if override:
        return max(1, int(override))
    return max(1, (os.cpu_count() or 2) // 2)


#: Stato per-worker. La rete arriva una volta sola all'avvio del processo
#: (50 KB, un millisecondo) invece che a ogni task.
_WORKER: dict = {}


def _init_replay_worker(payload) -> None:
    network, init_marking, final_marking, places = payload
    _WORKER["network"] = network
    _WORKER["init"] = init_marking
    _WORKER["final"] = final_marking
    # I Place sopravvivono al pickle come gli stessi oggetti che stanno dentro
    # la rete, perche' vengono serializzati insieme a lei: l'indice resta valido.
    _WORKER["index"] = {place: position for position, place in enumerate(places)}
    _WORKER["size"] = len(places)


def _replay_chunk(prefixes: Sequence[tuple[str, ...]]) -> list[tuple[int, ...]]:
    """Rigioca un blocco di prefissi, UNO PER CHIAMATA.

    Il blocco non diventa un unico ``EventLog``: pm4py condivide delle cache fra
    le tracce di una stessa ``apply``, quindi rigiocare n prefissi insieme da'
    marking diversi dal rigiocarli separatamente. Il blocco serve solo a non
    pagare un round-trip di IPC per prefisso.
    """

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

#: Limite di ricorsione per la sola scoperta, e stack del thread che la ospita.
#: L'inductive miner di pm4py e' ricorsivo -- taglia il log, ricorre sui pezzi,
#: e alla fine fa un ``deepcopy`` dell'albero, che ricorre a sua volta. Su BPIC15
#: sotto il protocollo A l'albero e' profondo abbastanza da sfondare il limite di
#: default (1000) e il processo muore con ``RecursionError`` prima di scrivere
#: una riga. Sotto B non capitava perche' li' la knowledge viene dal test, che e'
#: un quarto delle tracce.
#:
#: Le taglie si provano in ordine e si tiene la prima accettata: il massimo non
#: e' lo stesso ovunque -- Linux prende 256 MB, Windows li rifiuta con
#: ``ValueError``. Lo zero finale e' il default del sistema, cioe' nessun
#: aumento: se si arriva li' il ``RecursionError`` puo' tornare, ma almeno e'
#: un'eccezione leggibile e non un thread che non parte.
_DISCOVERY_RECURSION_LIMIT = 50_000
_DISCOVERY_STACK_SIZES = (256 * 1024 * 1024, 128 * 1024 * 1024,
                          64 * 1024 * 1024, 32 * 1024 * 1024, 0)


def _discover_inductive(event_log, noise_threshold: float):
    """``discover_petri_net_inductive`` con abbastanza stack per finire.

    Alzare ``sys.setrecursionlimit`` da solo non basta e anzi peggiora: il limite
    e' un conto di frame Python, ma lo stack vero e' quello del sistema, e
    superarlo non da' un'eccezione, da' un segfault. Per questo la scoperta gira
    in un thread creato con uno stack esplicito.

    Non e' un'approssimazione: stessa chiamata, stessi parametri, stessa rete.
    Cambia solo quanto spazio ha per arrivare in fondo.
    """

    outcome: dict = {}

    def discover() -> None:
        previous_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(_DISCOVERY_RECURSION_LIMIT)
        try:
            outcome["net"] = pm4py.discover_petri_net_inductive(
                event_log, noise_threshold=noise_threshold)
        except BaseException as error:          # rilanciata nel chiamante: senza
            outcome["error"] = error            # questo il thread morirebbe muto
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
    #: Memoizzazione di ``prefix_marking``. E' una funzione pura del prefisso --
    #: la rete non cambia -- e i prefissi si ripetono parecchio: 4x su BPIC12,
    #: 119x su BPIC20. Senza cache lo stesso marking viene rigiocato una volta
    #: per ``with_markings`` e una per ``with_marking_sequences``, che chiedono
    #: esattamente le stesse chiavi. Non e' un'approssimazione: i valori escono
    #: identici, cambia solo quante volte si chiama pm4py.
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

    def prefix_markings(self, prefixes: Iterable[Sequence[str]],
                        workers: int | None = None,
                        chunk_size: int = CHUNK_SIZE) -> list[tuple[int, ...]]:
        """Controparte parallela di :meth:`prefix_marking`, su molti prefissi.

        Stessa semantica, un processo per core invece che uno solo. Il replay e'
        Python puro, quindi i thread non servirebbero a niente: il GIL li
        serializzerebbe. I prefissi sono pero' indipendenti fra loro --
        ``prefix_marking`` e' una funzione pura e ogni chiamata a pm4py e'
        isolata -- per cui distribuirli su piu' processi da' gli stessi valori
        per costruzione.

        Riempie la stessa cache di ``prefix_marking``, quindi chiamarla prima di
        una passata seriale rende quella passata una sequenza di hit. Restituisce
        i marking nell'ordine dei prefissi ricevuti, duplicati compresi.
        """

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

    def trace_fitness(self, traces: Iterable[Sequence[str]]) -> list[float]:
        """Token-based-replay fitness of whole traces against this net.

        Diverso da :meth:`prefix_marking` per intento: li' il replay produce una
        *feature* e ogni prefisso va rigiocato isolato, qui produce una *metrica*
        su tracce complete, quindi una sola chiamata su tutto il log va bene ed
        e' la forma prevista dall'API pm4py. Il valore e' la ``trace_fitness``
        del replay: 1.0 se la traccia si rigioca senza token mancanti ne'
        residui, 0.0 per una traccia vuota o del tutto fuori modello.
        """

        traces = list(traces)
        if not traces:
            return []
        diagnostics = pm4py.conformance_diagnostics_token_based_replay(
            self._create_event_log(traces), self.network,
            self.init_marking, self.final_marking,
            # Stessa configurazione del replay che produce i marking: attraversa
            # le transizioni invisibili e non si ferma al primo disallineamento.
            # Se la fitness usasse un'altra configurazione misurerebbe una rete
            # diversa da quella che il resto della tesi descrive.
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