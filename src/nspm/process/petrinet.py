from __future__ import annotations
from dataclasses import dataclass
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

@dataclass(frozen=True)
class PetriNet:

    network: Pm4pyNet
    init_marking: Marking
    final_marking: Marking
    places: tuple[Pm4pyNet.Place, ...]
    transitions: tuple[Pm4pyNet.Transition, ...]

    @classmethod
    def from_traces(cls, traces: Mapping[str, Sequence[str]], noise_threshold: float =0.2) -> PetriNet:

        event_log = cls._create_event_log(traces.values())
        network, init_marking, final_marking = pm4py.discover_petri_net_inductive(event_log, noise_threshold=noise_threshold)
        places = tuple(sorted(network.places, key=lambda p: p.name))
        transitions = tuple(sorted(network.transitions, key=lambda t: (t.label is None, t.label or "", str(t))))

        return cls(network, init_marking, final_marking, places, transitions)

    def prefix_marking(self, prefix: Sequence[str]) -> tuple[int, ...]:

        prefix_log = self._create_event_log([prefix])
        result = token_replay.apply(prefix_log, self.network, self.init_marking, self.final_marking, parameters=REPLAY_PARAMETERS)[0]

        index = self.place_index
        vector = [0] * len(self.places)
        for place, tokens in result["reached_marking"].items():
            vector[index[place]] = tokens

        return tuple(vector)

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