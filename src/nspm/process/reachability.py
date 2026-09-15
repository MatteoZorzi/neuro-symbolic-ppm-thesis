"""Petri-net reachability graph turned into a deterministic automaton.

This is a Python port of ``temp/DpnReachabilityWoGuards.java`` (ProM's
``DataPetriNets`` + ``LTL2Automaton`` plug-ins) on top of :mod:`pm4py` nets.
The Java code traverses every reachable marking of the (data) Petri net and
builds an automaton during that traversal; the same logic is reproduced here:

1. start from the initial marking and explore every reachable marking by firing
   enabled transitions (:func:`_explore`);
2. one automaton state per marking, one automaton edge per fired transition,
   labelled with the transition's activity label;
3. markings that match a final marking become **accepting** states;
4. a **trap** (fail) state is added, every activity that a state does not handle
   leads to it, and the trap self-loops on everything -- the automaton is
   *complete*, so a rejected trace is one that ends in the trap;
5. activities outside the net's own alphabet **self-loop** on every state (the
   Java ``addNegativePropositionsTransition`` call): a net says nothing about
   activities it does not model, so those events must not reject a trace;
6. the result is determinized, renumbered and minimized -- the Python
   equivalent of ``op.determinize().op.complete().op.renumber().op.minimize()``.

Two deliberate deviations from the Java source:

* **Silent transitions.** The Java version carries a ``//TODO: Silent
  transitions`` and simply skips invisible transitions. That is unusable for
  pm4py nets: the inductive miner emits τ-transitions for every skip and loop,
  and dropping them disconnects the reachability graph. Here τ-transitions are
  handled by **τ-closure**: a state is the set of markings reachable by zero or
  more silent firings, exactly as in the standard NFA-with-ε construction.
* **Determinization is fused into the traversal.** Because τ-closure (and
  duplicate activity labels, which the inductive miner also produces) makes the
  marking graph nondeterministic, states are *sets* of markings built by subset
  construction, instead of building an NFA first and determinizing it after.
  The accepted language is the same.

The automaton produced here is a genuine state machine over markings, so it is
strictly more precise than the directly-follows :class:`~.automaton.ProcessDFA`
learned from traces. To feed it to the existing checker loss, which is keyed on
the *last activity* only, use :meth:`ReachabilityAutomaton.to_process_dfa` --
see the note on that method about the precision that projection gives up.
"""

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


class UnboundedNetError(RuntimeError):
    """Raised when the reachability graph exceeds ``max_states``.

    Sound workflow nets (what the inductive miner returns) are bounded, so this
    normally means the net is unbounded or pathologically large.
    """


# --------------------------------------------------------------------------- #
# Petri-net primitives
# --------------------------------------------------------------------------- #


def _marking_key(marking) -> MarkingKey:
    """Canonicalise a ``pm4py`` marking (or plain mapping) into a key."""

    items = []
    for place, count in marking.items():
        count = int(count)
        if count:
            items.append((getattr(place, "name", place), count))
    return tuple(sorted(items))


def _compile_transitions(net) -> tuple[CompiledTransition, ...]:
    """Freeze the net into ``(label, preset, postset)`` triples.

    Sorting is deterministic (visible before silent, then by label and name) so
    that state numbering is reproducible across runs -- the same reason
    :class:`~.petrinet.PetriNet` sorts its places and transitions.
    """

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


def _fire(
    marking: MarkingKey, preset: Mapping[str, int], postset: Mapping[str, int]
) -> MarkingKey:
    """Consume the preset and produce the postset (transition must be enabled)."""

    tokens = dict(marking)
    for place, weight in preset.items():
        tokens[place] -= weight
    for place, weight in postset.items():
        tokens[place] = tokens.get(place, 0) + weight
    return tuple(sorted((place, n) for place, n in tokens.items() if n))


def _tau_closure(
    markings: Iterable[MarkingKey],
    silent: Sequence[CompiledTransition],
    max_states: int,
) -> frozenset[MarkingKey]:
    """All markings reachable from ``markings`` by zero or more silent firings."""

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


@dataclass(frozen=True)
class ReachabilityAutomaton:
    """A complete, minimal DFA accepting exactly the net's firing language.

    ``transitions[state][activity]`` is total over :attr:`activities`: an
    activity a state cannot fire leads to :attr:`trap_state`, which absorbs
    everything. A trace is accepted iff replaying it ends in an accepting state,
    which -- because final markings are the accepting ones -- means the trace is
    a complete, sound execution of the net, not merely a prefix of one.
    """

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
        """Build the automaton from a ``pm4py`` net by marking traversal.

        Args:
            net: a ``pm4py`` :class:`~pm4py.objects.petri_net.obj.PetriNet`.
            initial_marking: the net's initial marking.
            final_markings: one marking, or an iterable of markings, that count
                as accepting (``getFinalMarkings()`` in the Java version).
            alphabet: optional wider activity alphabet, e.g. the log vocabulary.
                Activities in it that the net does not model self-loop on every
                state instead of rejecting -- the Java behaviour for
                propositions outside the net's alphabet.
            minimize: run Moore partition refinement on the completed DFA.
            max_states: safety bound; see :class:`UnboundedNetError`.
        """

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

    @classmethod
    def from_process_net(
        cls, petri_net, **kwargs
    ) -> "ReachabilityAutomaton":
        """Build from this project's :class:`~.petrinet.PetriNet` wrapper."""

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

    @property
    def transition_count(self) -> int:
        """Number of edges that do not lead to the trap state."""

        return sum(
            1
            for state, row in self.transitions.items()
            if state != self.trap_state
            for target in row.values()
            if target != self.trap_state
        )

    def allowed_activities(self, state: int) -> frozenset[str]:
        """Activities that do not send ``state`` to the trap."""

        row = self.transitions.get(state, {})
        return frozenset(
            activity
            for activity, target in row.items()
            if target != self.trap_state
        )

    def step(self, state: int, activity: str) -> int | None:
        """Follow one edge; ``None`` for an activity outside the alphabet."""

        return self.transitions.get(state, {}).get(activity)

    def state_after(self, prefix: Sequence[str]) -> int | None:
        """Replay a prefix from the initial state.

        Returns the reached state, or ``None`` if the prefix uses an activity
        the automaton has no edge for at all. Reaching the trap is *not*
        ``None``: it is a well-defined "this prefix is non-conformant" answer.
        """

        state = self.init_state
        for activity in prefix:
            nxt = self.step(state, activity)
            if nxt is None:
                return None
            state = nxt
        return state

    def accepts(self, trace: Sequence[str]) -> bool:
        """True iff the trace is a complete execution reaching a final marking."""

        state = self.state_after(trace)
        return state is not None and state in self.accepting

    def is_conformant_prefix(self, prefix: Sequence[str]) -> bool:
        """True iff the prefix can still be extended into an accepted trace."""

        state = self.state_after(prefix)
        return state is not None and state != self.trap_state

    # -- transformations ---------------------------------------------------- #

    def minimized(self) -> "ReachabilityAutomaton":
        """Moore partition refinement, then a BFS renumbering from the init state.

        The DFA is complete by construction, so refining by
        ``(block, blocks of successors)`` until the partition is stable yields
        the unique minimal DFA for the language.
        """

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

    def _renumber(self, block_of: Mapping[int, int]) -> "ReachabilityAutomaton":
        """Collapse states by block and renumber them breadth-first from init."""

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

    def to_process_dfa(self) -> ProcessDFA:
        """Project onto the directly-follows form the checker loss consumes.

        :func:`~..learning.logic.build_allowed_mask` keys the allowed-next mask
        on the **last activity token** alone, so a marking-aware automaton has
        to be flattened: the successors permitted after activity ``a`` become
        the union of what is permitted in *every* automaton state reachable by
        an ``a``-edge.

        That union is a sound over-approximation -- it never forbids something
        the net allows, so the checker penalty stays free of false positives --
        but it does lose the net's memory: when ``a`` occurs in two different
        contexts the projection permits the union of both continuations. Use
        :meth:`step`/:meth:`is_conformant_prefix` directly if you need the
        state-aware answer.
        """

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

    def to_dot(self) -> str:
        """DOT string, mirroring ``createAutomatonVisualizationString``.

        The trap state and every edge into it are omitted, and parallel edges
        between the same pair of states are merged into one multi-line label.
        """

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

    def to_networkx(self):
        """Create a NetworkX graph lazily, keeping it an optional dependency."""

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


def _explore(
    init_key: MarkingKey,
    visible: Sequence[CompiledTransition],
    silent: Sequence[CompiledTransition],
    max_states: int,
) -> tuple[list[frozenset[MarkingKey]], list[dict[str, int]]]:
    """Traverse every reachable marking, building the automaton on the way.

    This is ``visitNextState`` from the Java source, rewritten as an explicit
    worklist (no recursion depth limit) over τ-closed *sets* of markings. Each
    set is one automaton state; each activity label enabled anywhere in the set
    gives one outgoing edge to the τ-closure of the markings it produces.
    """

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
