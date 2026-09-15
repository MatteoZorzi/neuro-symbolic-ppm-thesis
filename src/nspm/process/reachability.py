# The reachability graph of a Petri net, as a deterministic automaton.
# A Python port of ProM's DpnReachabilityWoGuards.java on top of pm4py nets.

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .automaton import END, START, ProcessDFA


#: A marking, canonicalised as a sorted ``(place name, token count)`` tuple so
#: that it is hashable and comparable (``pm4py.Marking`` is a ``Counter``).
MarkingKey = tuple[tuple[str, int], ...]

#: A compiled transition: ``(label or None, preset, postset)`` with presets and
#: postsets as ``{place name: arc weight}``.
CompiledTransition = tuple[str | None, Mapping[str, int], Mapping[str, int]]


# Raised when the reachability graph exceeds ``max_states``
class UnboundedNetError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Petri-net primitives
# --------------------------------------------------------------------------- #


# Canonicalise a ``pm4py`` marking (or plain mapping) into a key
def _marking_key(marking) -> MarkingKey:

    items = []
    for place, count in marking.items():
        count = int(count)
        if count:
            items.append((getattr(place, "name", place), count))
    return tuple(sorted(items))


# Freeze the net into ``(label, preset, postset)`` triples
def _compile_transitions(net) -> tuple[CompiledTransition, ...]:

    compiled: list[CompiledTransition] = []
    for transition in sorted(
        net.transitions, key=lambda t: (t.label is None, t.label or "", str(t.name))
    ):
        preset: dict[str, int] = {}
        for arc in transition.in_arcs:
            preset[arc.source.name] = preset.get(arc.source.name, 0) + int(arc.weight)
        postset: dict[str, int] = {}
        for arc in transition.out_arcs:
            postset[arc.target.name] = postset.get(arc.target.name, 0) + int(arc.weight)
        compiled.append((transition.label, preset, postset))
    return tuple(compiled)


def _is_enabled(marking: MarkingKey, preset: Mapping[str, int]) -> bool:
    tokens = dict(marking)
    return all(tokens.get(place, 0) >= weight for place, weight in preset.items())


# Consume the preset and produce the postset (transition must be enabled)
def _fire(
    marking: MarkingKey, preset: Mapping[str, int], postset: Mapping[str, int]
) -> MarkingKey:

    tokens = dict(marking)
    for place, weight in preset.items():
        tokens[place] -= weight
    for place, weight in postset.items():
        tokens[place] = tokens.get(place, 0) + weight
    return tuple(sorted((place, n) for place, n in tokens.items() if n))


# All markings reachable from ``markings`` by zero or more silent firings
def _tau_closure(
    markings: Iterable[MarkingKey],
    silent: Sequence[CompiledTransition],
    max_states: int,
) -> frozenset[MarkingKey]:

    closure = set(markings)
    if not silent:
        return frozenset(closure)
    queue = deque(closure)
    while queue:
        marking = queue.popleft()
        for _, preset, postset in silent:
            if not _is_enabled(marking, preset):
                continue
            successor = _fire(marking, preset, postset)
            if successor not in closure:
                if len(closure) >= max_states:
                    raise UnboundedNetError(
                        f"τ-closure exceeded max_states={max_states}; "
                        "the net is likely unbounded."
                    )
                closure.add(successor)
                queue.append(successor)
    return frozenset(closure)


# --------------------------------------------------------------------------- #
# The automaton
# --------------------------------------------------------------------------- #


# A complete, minimal DFA accepting exactly the net's firing language
@dataclass(frozen=True)
class ReachabilityAutomaton:

    activities: tuple[str, ...]
    init_state: int
    transitions: Mapping[int, Mapping[str, int]]
    accepting: frozenset[int]
    trap_state: int | None
    #: Provenance: the markings each state stands for. After minimization a
    #: state may merge several marking sets, so this is the union over the
    #: merged states. Empty for the trap state.
    state_markings: Mapping[int, frozenset[MarkingKey]]

    # -- construction ------------------------------------------------------- #

    # Build the automaton from a ``pm4py`` net by marking traversal
    @classmethod
    def from_petri_net(
        cls,
        net,
        initial_marking,
        final_markings,
        *,
        alphabet: Sequence[str] | None = None,
        minimize: bool = True,
        max_states: int = 100_000,
    ) -> "ReachabilityAutomaton":

        compiled = _compile_transitions(net)
        visible = tuple(t for t in compiled if t[0] is not None)
        silent = tuple(t for t in compiled if t[0] is None)

        net_activities = sorted({label for label, _, _ in visible})
        foreign = sorted(set(alphabet or ()) - set(net_activities))
        activities = tuple(sorted(set(net_activities) | set(foreign)))

        finals = cls._final_marking_keys(final_markings)
        init_key = _marking_key(initial_marking)

        state_sets, edges = _explore(init_key, visible, silent, max_states)
        accepting = frozenset(
            state
            for state, markings in enumerate(state_sets)
            if markings & finals
        )

        # Completion: every unhandled activity of the net's own alphabet goes to
        # the trap; foreign activities self-loop; the trap absorbs everything.
        trap_state = len(state_sets)
        table: dict[int, dict[str, int]] = {}
        for state in range(len(state_sets)):
            row = dict(edges[state])
            for activity in net_activities:
                row.setdefault(activity, trap_state)
            for activity in foreign:
                row[activity] = state
            table[state] = row
        table[trap_state] = {activity: trap_state for activity in activities}

        markings_by_state: dict[int, frozenset[MarkingKey]] = {
            state: markings for state, markings in enumerate(state_sets)
        }
        markings_by_state[trap_state] = frozenset()

        automaton = cls(
            activities=activities,
            init_state=0,
            transitions=table,
            accepting=accepting,
            trap_state=trap_state,
            state_markings=markings_by_state,
        )
        if minimize:
            automaton = automaton.minimized()
        return automaton

    # Build from this project's :class:`~.petrinet.PetriNet` wrapper
    @classmethod
    def from_process_net(
        cls, petri_net, **kwargs
    ) -> "ReachabilityAutomaton":

        return cls.from_petri_net(
            petri_net.network,
            petri_net.init_marking,
            petri_net.final_marking,
            **kwargs,
        )

    @staticmethod
    def _final_marking_keys(final_markings) -> frozenset[MarkingKey]:
        if final_markings is None:
            return frozenset()
        if hasattr(final_markings, "items"):  # A single pm4py Marking (Counter).
            return frozenset({_marking_key(final_markings)})
        return frozenset(_marking_key(marking) for marking in final_markings)

    # -- inspection --------------------------------------------------------- #

    @property
    def states(self) -> tuple[int, ...]:
        return tuple(sorted(self.transitions))

    @property
    def state_count(self) -> int:
        return len(self.transitions)

    # Number of edges that do not lead to the trap state
    @property
    def transition_count(self) -> int:

        return sum(
            1
            for state, row in self.transitions.items()
            if state != self.trap_state
            for target in row.values()
            if target != self.trap_state
        )

    # Activities that do not send ``state`` to the trap
    def allowed_activities(self, state: int) -> frozenset[str]:

        row = self.transitions.get(state, {})
        return frozenset(
            activity
            for activity, target in row.items()
            if target != self.trap_state
        )

    # Follow one edge; ``None`` for an activity outside the alphabet
    def step(self, state: int, activity: str) -> int | None:

        return self.transitions.get(state, {}).get(activity)

    # Replay a prefix from the initial state
    def state_after(self, prefix: Sequence[str]) -> int | None:

        state = self.init_state
        for activity in prefix:
            nxt = self.step(state, activity)
            if nxt is None:
                return None
            state = nxt
        return state

    # True iff the trace is a complete execution reaching a final marking
    def accepts(self, trace: Sequence[str]) -> bool:

        state = self.state_after(trace)
        return state is not None and state in self.accepting

    # True iff the prefix can still be extended into an accepted trace
    def is_conformant_prefix(self, prefix: Sequence[str]) -> bool:

        state = self.state_after(prefix)
        return state is not None and state != self.trap_state

    # -- transformations ---------------------------------------------------- #

    # Moore partition refinement, then a BFS renumbering from the init state
    def minimized(self) -> "ReachabilityAutomaton":

        states = self.states
        block_of = {state: int(state in self.accepting) for state in states}
        while True:
            signatures: dict[tuple, int] = {}
            refined: dict[int, int] = {}
            for state in states:
                row = self.transitions[state]
                signature = (
                    block_of[state],
                    tuple(block_of[row[activity]] for activity in self.activities),
                )
                refined[state] = signatures.setdefault(signature, len(signatures))
            if len(signatures) == len(set(block_of.values())):
                break
            block_of = refined

        return self._renumber(block_of)

    # Collapse states by block and renumber them breadth-first from init
    def _renumber(self, block_of: Mapping[int, int]) -> "ReachabilityAutomaton":

        members: dict[int, list[int]] = {}
        for state, block in block_of.items():
            members.setdefault(block, []).append(state)

        new_id: dict[int, int] = {}
        queue = deque([block_of[self.init_state]])
        new_id[block_of[self.init_state]] = 0
        order = [block_of[self.init_state]]
        while queue:
            block = queue.popleft()
            representative = members[block][0]
            row = self.transitions[representative]
            for activity in self.activities:
                target = block_of[row[activity]]
                if target not in new_id:
                    new_id[target] = len(new_id)
                    order.append(target)
                    queue.append(target)

        table: dict[int, dict[str, int]] = {}
        markings: dict[int, frozenset[MarkingKey]] = {}
        accepting: set[int] = set()
        trap: int | None = None
        for block in order:
            state = new_id[block]
            representative = members[block][0]
            table[state] = {
                activity: new_id[block_of[self.transitions[representative][activity]]]
                for activity in self.activities
            }
            markings[state] = frozenset().union(
                *(self.state_markings.get(member, frozenset()) for member in members[block])
            )
            if representative in self.accepting:
                accepting.add(state)
            if self.trap_state is not None and self.trap_state in members[block]:
                trap = state

        return ReachabilityAutomaton(
            activities=self.activities,
            init_state=0,
            transitions=table,
            accepting=frozenset(accepting),
            trap_state=trap,
            state_markings=markings,
        )

    # Project onto the directly-follows form the checker loss consumes
    def to_process_dfa(self) -> ProcessDFA:

        # For every activity, collect the automaton states an ``a``-edge lands
        # in (the trap contributes nothing: it forbids everything).
        landings: dict[str, set[int]] = {}
        for state, row in self.transitions.items():
            if state == self.trap_state:
                continue
            for activity, target in row.items():
                if target == self.trap_state:
                    continue
                landings.setdefault(activity, set()).add(target)

        allowed_next: dict[str, set[str]] = {}
        allowed_next[START] = set(self.allowed_activities(self.init_state))
        if self.init_state in self.accepting:
            allowed_next[START].add(END)

        end_actions: set[str] = set()
        for activity, targets in landings.items():
            successors: set[str] = set()
            for target in targets:
                successors |= self.allowed_activities(target)
                if target in self.accepting:
                    successors.add(END)
                    end_actions.add(activity)
            allowed_next[activity] = successors

        if not allowed_next[START]:
            raise ValueError(
                "The net's initial marking enables no visible activity; the "
                "resulting automaton has no valid starting action."
            )

        return ProcessDFA(
            allowed_next={
                state: frozenset(successors)
                for state, successors in allowed_next.items()
            },
            activities=tuple(self.activities),
            start_actions=frozenset(allowed_next[START] - {END}),
            end_actions=frozenset(end_actions),
        )

    # -- serialization and export ------------------------------------------- #

    def to_dict(self) -> dict[str, object]:
        return {
            "activities": list(self.activities),
            "init_state": self.init_state,
            "trap_state": self.trap_state,
            "accepting": sorted(self.accepting),
            "transitions": {
                str(state): dict(sorted(row.items()))
                for state, row in sorted(self.transitions.items())
            },
            "state_markings": {
                str(state): [
                    [list(place) for place in marking]
                    for marking in sorted(markings)
                ]
                for state, markings in sorted(self.state_markings.items())
            },
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ReachabilityAutomaton":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            activities=tuple(data["activities"]),
            init_state=int(data["init_state"]),
            transitions={
                int(state): dict(row) for state, row in data["transitions"].items()
            },
            accepting=frozenset(int(state) for state in data["accepting"]),
            trap_state=None if data["trap_state"] is None else int(data["trap_state"]),
            state_markings={
                int(state): frozenset(
                    tuple((place, int(count)) for place, count in marking)
                    for marking in markings
                )
                for state, markings in data["state_markings"].items()
            },
        )

    # DOT string, mirroring ``createAutomatonVisualizationString``
    def to_dot(self) -> str:

        lines = ['digraph "" {', '  init [shape=none, label=""];', '  rankdir = "LR";']
        for state in self.states:
            if state == self.trap_state:
                continue
            shape = "doublecircle" if state in self.accepting else "circle"
            lines.append(f"  s{state} [shape={shape}];")

        for state in self.states:
            if state == self.trap_state:
                continue
            merged: dict[int, list[str]] = {}
            for activity, target in sorted(self.transitions[state].items()):
                if target == self.trap_state:
                    continue
                merged.setdefault(target, []).append(activity)
            for target, labels in sorted(merged.items()):
                label = "\\n".join(labels).replace('"', '\\"')
                lines.append(f'  s{state} -> s{target} [label="{label}"];')

        lines.append(f"  init -> s{self.init_state};")
        lines.append("}")
        return "\n".join(lines)

    # Create a NetworkX graph lazily, keeping it an optional dependency
    def to_networkx(self):

        import networkx as nx

        graph = nx.MultiDiGraph()
        for state in self.states:
            graph.add_node(
                state,
                accepting=state in self.accepting,
                trap=state == self.trap_state,
            )
        for state, row in self.transitions.items():
            for activity, target in row.items():
                graph.add_edge(state, target, label=activity)
        return graph


# --------------------------------------------------------------------------- #
# The traversal itself
# --------------------------------------------------------------------------- #


# Traverse every reachable marking, building the automaton on the way
def _explore(
    init_key: MarkingKey,
    visible: Sequence[CompiledTransition],
    silent: Sequence[CompiledTransition],
    max_states: int,
) -> tuple[list[frozenset[MarkingKey]], list[dict[str, int]]]:

    initial = _tau_closure([init_key], silent, max_states)
    state_of: dict[frozenset[MarkingKey], int] = {initial: 0}
    state_sets: list[frozenset[MarkingKey]] = [initial]
    edges: list[dict[str, int]] = [{}]

    queue = deque([0])
    while queue:
        state = queue.popleft()
        successors: dict[str, set[MarkingKey]] = {}
        for marking in state_sets[state]:
            for label, preset, postset in visible:
                if _is_enabled(marking, preset):
                    successors.setdefault(label, set()).add(
                        _fire(marking, preset, postset)
                    )

        for label in sorted(successors):
            target_set = _tau_closure(successors[label], silent, max_states)
            target = state_of.get(target_set)
            if target is None:
                if len(state_sets) >= max_states:
                    raise UnboundedNetError(
                        f"Reachability graph exceeded max_states={max_states}; "
                        "the net is likely unbounded."
                    )
                target = len(state_sets)
                state_of[target_set] = target
                state_sets.append(target_set)
                edges.append({})
                queue.append(target)
            edges[state][label] = target

    return state_sets, edges
