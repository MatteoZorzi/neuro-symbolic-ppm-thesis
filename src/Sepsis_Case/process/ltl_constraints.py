"""LTLf temporal constraints and their DFAs for the learned-embedder branch.

This module supplies the *symbolic temporal knowledge* that the original
``ProcessDFA`` (a directly-follows automaton) does not capture. Where the
directly-follows automaton only encodes one-step ``next`` relations, here we
mine genuine LTLf **precedence** constraints from the training traces and build,
for each one, a small deterministic automaton whose edges carry propositional
guards over the activity alphabet.

A precedence constraint "``a`` precedes ``b``" is written in finite LTL as::

    (¬b U a) ∨ G(¬b)

i.e. ``b`` may not occur until ``a`` has occurred (and a trace with no ``b`` at
all trivially satisfies it). This is exactly the kind of ordering constraint the
T-LEAF paper expresses in LTLf and compiles to a DFA via LTLfKit/spot. Because
neither ``spot`` nor ``ltlf2dfa`` is available in this environment, the (tiny,
three-state) DFA for each precedence constraint is constructed directly instead
of being compiled from the formula string. The formula string is still produced
by :meth:`PrecedenceConstraint.to_ltlf` for inspection and provenance.

No general LTLf compiler is used, so only the precedence template is supported;
conjunctions of constraints are handled at the loss level (one DFA per relevant
constraint) rather than by building a product automaton -- see
``learning/logic.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Mapping, Sequence


# Node-type labels, matching the convention used by the original T-LEAF
# embedder (``src/Synthetic/models/Graph.py``): the initial state, ordinary
# (non-accepting) states, and accepting/final states.
INIT = 0
COMMON = 1
FINAL = 2


@dataclass(frozen=True)
class Guard:
    """A propositional edge guard over the one-hot activity alphabet.

    Exactly one activity is true at each step of a process trace, so a guard is
    fully described by the activities it requires (``positives``), the
    activities it forbids (``negatives``), or the catch-all ``is_true`` guard.
    """

    positives: tuple[str, ...] = ()
    negatives: tuple[str, ...] = ()
    is_true: bool = False

    @classmethod
    def activity(cls, name: str) -> "Guard":
        return cls(positives=(name,))

    @classmethod
    def none_of(cls, names: Iterable[str]) -> "Guard":
        return cls(negatives=tuple(sorted(set(names))))

    @classmethod
    def true(cls) -> "Guard":
        return cls(is_true=True)

    def accepts(self, activity: str) -> bool:
        if self.is_true:
            return True
        if self.positives and activity not in self.positives:
            return False
        if activity in self.negatives:
            return False
        return bool(self.positives) or bool(self.negatives)


@dataclass(frozen=True)
class ConstraintDFA:
    """A small deterministic automaton with propositional edge guards.

    ``accepting`` is stored explicitly rather than derived from ``node_types``:
    the initial state can be both the start state (node type ``INIT``, used for
    feature typing and random-walk seeding) *and* an accepting state (an empty
    trace satisfies a precedence rule), so the two notions must be independent.
    """

    init_state: int
    node_types: Mapping[int, int]
    edges: tuple[tuple[int, int, Guard], ...]
    accepting: frozenset[int]

    @property
    def states(self) -> tuple[int, ...]:
        return tuple(sorted(self.node_types))

    def accepts(self, trace: Sequence[str]) -> bool:
        """Run the (deterministic) automaton and test for acceptance."""

        state = self.init_state
        for activity in trace:
            nxt = None
            for src, dst, guard in self.edges:
                if src == state and guard.accepts(activity):
                    nxt = dst
                    break
            if nxt is None:  # No outgoing guard matched: reject.
                return False
            state = nxt
        return state in self.accepting


@dataclass(frozen=True)
class PrecedenceConstraint:
    """The LTLf constraint "``earlier`` precedes ``later``"."""

    earlier: str
    later: str
    support: int
    confidence: float

    def to_ltlf(self) -> str:
        """Return the finite-LTL string ``(¬b U a) ∨ G(¬b)``."""

        return f"(!'{self.later}' U '{self.earlier}') | G(!'{self.later}')"

    def is_satisfied(self, trace: Sequence[str]) -> bool:
        """True iff ``later`` never appears before the first ``earlier``."""

        for activity in trace:
            if activity == self.later:
                return False
            if activity == self.earlier:
                return True
        return True  # ``later`` never occurred.

    def to_dfa(self) -> ConstraintDFA:
        """Build the three-state DFA accepting traces that satisfy this rule."""

        # State 0: initial and accepting (no ``later`` seen yet).
        # State 1: ``earlier`` has been seen -> ``later`` now permitted (accept).
        # State 2: ``later`` seen before ``earlier`` -> violated (sink, reject).
        node_types = {0: INIT, 1: FINAL, 2: COMMON}
        edges = (
            (0, 1, Guard.activity(self.earlier)),
            (0, 2, Guard.activity(self.later)),
            (0, 0, Guard.none_of((self.earlier, self.later))),
            (1, 1, Guard.true()),
            (2, 2, Guard.true()),
        )
        # State 0 (init, no ``later`` yet) and state 1 (``earlier`` seen) both
        # accept; state 2 (``later`` before ``earlier``) is the rejecting sink.
        return ConstraintDFA(
            init_state=0,
            node_types=node_types,
            edges=edges,
            accepting=frozenset({0, 1}),
        )


def mine_precedence_constraints(
    traces: Iterable[Sequence[str]],
    *,
    min_support: int = 20,
    min_confidence: float = 0.98,
    max_constraints: int = 40,
) -> list[PrecedenceConstraint]:
    """Mine high-confidence precedence rules from (training) traces.

    A candidate "``a`` precedes ``b``" is scored over the traces that contain
    ``b`` (its support). Confidence is the fraction of those traces in which
    ``a`` occurs strictly before the first ``b``. Rules meeting both thresholds
    are returned, strongest first, capped at ``max_constraints``.
    """

    traces = [tuple(trace) for trace in traces if trace]
    activities = sorted({activity for trace in traces for activity in trace})

    # later_support[b] = traces containing b; pair_before[(a, b)] = traces in
    # which a appears strictly before the first b.
    later_support: dict[str, int] = {activity: 0 for activity in activities}
    pair_before: dict[tuple[str, str], int] = {}
    for trace in traces:
        first_index: dict[str, int] = {}
        for position, activity in enumerate(trace):
            first_index.setdefault(activity, position)
        present = set(trace)
        for later in present:
            later_support[later] += 1
            b_first = first_index[later]
            for earlier in present:
                if earlier == later:
                    continue
                if first_index[earlier] < b_first:
                    key = (earlier, later)
                    pair_before[key] = pair_before.get(key, 0) + 1

    candidates: list[PrecedenceConstraint] = []
    for (earlier, later), before in pair_before.items():
        support = later_support[later]
        if support < min_support:
            continue
        confidence = before / support
        if confidence < min_confidence:
            continue
        candidates.append(
            PrecedenceConstraint(earlier, later, support=support, confidence=confidence)
        )

    candidates.sort(key=lambda c: (c.confidence, c.support), reverse=True)
    return candidates[:max_constraints]


def relevant_constraints(
    constraints: Sequence[PrecedenceConstraint],
    trace_activities: Iterable[str],
) -> list[PrecedenceConstraint]:
    """Constraints whose consequent ``later`` activity appears in the trace.

    These are the only constraints a trace can actually violate, mirroring the
    paper's per-sample selection of clauses whose propositions are present.
    """

    present = set(trace_activities)
    return [c for c in constraints if c.later in present]


def sample_satisfying_trace(
    constraint: PrecedenceConstraint,
    alphabet: Sequence[str],
    *,
    length: int,
    rng: random.Random,
) -> tuple[str, ...]:
    """Synthesize a trace that satisfies the precedence constraint.

    With probability 1/2 the ``later`` activity is omitted entirely; otherwise
    ``earlier`` is placed strictly before the first ``later``.
    """

    others = [a for a in alphabet if a not in (constraint.earlier, constraint.later)]
    length = max(length, 3)
    if not others:
        return tuple([constraint.earlier] * (length - 1) + [constraint.later])

    if rng.random() < 0.5:  # ``later`` never appears -> trivially satisfied.
        return tuple(rng.choice(others) for _ in range(length))

    earlier_pos = rng.randrange(0, length - 1)
    later_pos = rng.randrange(earlier_pos + 1, length)
    trace = [rng.choice(others) for _ in range(length)]
    trace[earlier_pos] = constraint.earlier
    trace[later_pos] = constraint.later
    return tuple(trace)


def sample_unsatisfying_trace(
    constraint: PrecedenceConstraint,
    alphabet: Sequence[str],
    *,
    length: int,
    rng: random.Random,
) -> tuple[str, ...]:
    """Synthesize a trace that violates the constraint (``later`` before ``earlier``)."""

    others = [a for a in alphabet if a not in (constraint.earlier, constraint.later)]
    length = max(length, 3)
    trace = [rng.choice(others) if others else constraint.later for _ in range(length)]

    later_pos = rng.randrange(0, length - 1)
    trace[later_pos] = constraint.later
    # Place ``earlier`` strictly after the first ``later`` (or leave it absent).
    if rng.random() < 0.7:
        earlier_pos = rng.randrange(later_pos + 1, length)
        trace[earlier_pos] = constraint.earlier
    return tuple(trace)
