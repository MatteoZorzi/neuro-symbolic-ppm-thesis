# Empirical deterministic automaton derived from training traces

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


START = "<START>"
END = "<END>"


# A directly-follows automaton used as a next-action constraint
@dataclass(frozen=True)
class ProcessDFA:

    allowed_next: Mapping[str, frozenset[str]]
    activities: tuple[str, ...]
    start_actions: frozenset[str]
    end_actions: frozenset[str]

    @classmethod
    def from_traces(cls, traces: Iterable[Sequence[str]]) -> "ProcessDFA":
        transitions: dict[str, set[str]] = {}
        activities: set[str] = set()
        start_actions: set[str] = set()
        end_actions: set[str] = set()
        trace_count = 0

        for trace in traces:
            trace = list(trace)
            if not trace:
                continue
            trace_count += 1
            activities.update(trace)
            start_actions.add(trace[0])
            end_actions.add(trace[-1])
            previous = START
            for activity in trace:
                transitions.setdefault(previous, set()).add(activity)
                previous = activity
            transitions.setdefault(previous, set()).add(END)

        if trace_count == 0:
            raise ValueError("Cannot build a DFA from an empty trace collection.")

        frozen = {state: frozenset(next_states) for state, next_states in transitions.items()}
        return cls(
            allowed_next=frozen,
            activities=tuple(sorted(activities)),
            start_actions=frozenset(start_actions),
            end_actions=frozenset(end_actions),
        )

    @property
    def states(self) -> tuple[str, ...]:
        return (START, *self.activities, END)

    @property
    def transition_count(self) -> int:
        return sum(len(next_states) for next_states in self.allowed_next.values())

    def is_allowed(self, previous: str, following: str) -> bool:
        return following in self.allowed_next.get(previous, frozenset())

    # Real-activity successors observed after ``previous`` (excludes START/END)
    def allowed_activities(self, previous: str) -> frozenset[str]:

        return frozenset(
            action
            for action in self.allowed_next.get(previous, frozenset())
            if action not in (START, END)
        )

    # Return whether every transition, including start/end, was observed
    def accepts(self, trace: Sequence[str]) -> bool:

        if not trace:
            return False
        previous = START
        for activity in trace:
            if not self.is_allowed(previous, activity):
                return False
            previous = activity
        return self.is_allowed(previous, END)

    # Fraction of evaluated transitions represented by this automaton
    def transition_coverage(self, traces: Iterable[Sequence[str]]) -> float:

        covered = 0
        total = 0
        for trace in traces:
            previous = START
            for activity in trace:
                total += 1
                covered += int(self.is_allowed(previous, activity))
                previous = activity
        return covered / total if total else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "activities": list(self.activities),
            "start_actions": sorted(self.start_actions),
            "end_actions": sorted(self.end_actions),
            "allowed_next": {
                state: sorted(next_states)
                for state, next_states in sorted(self.allowed_next.items())
            },
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ProcessDFA":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            allowed_next={
                state: frozenset(next_states)
                for state, next_states in data["allowed_next"].items()
            },
            activities=tuple(data["activities"]),
            start_actions=frozenset(data["start_actions"]),
            end_actions=frozenset(data["end_actions"]),
        )

    # Create a NetworkX graph lazily, keeping it an optional dependency
    def to_networkx(self):

        import networkx as nx

        graph = nx.DiGraph()
        graph.add_nodes_from(self.states)
        for source, targets in self.allowed_next.items():
            graph.add_edges_from((source, target) for target in targets)
        return graph
