"""Deterministic StateKey, transition-table and loop semantics (stage 5)."""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..perception.projection_graph_types import GraphSnapshot

STATE_SCHEMA_VERSION = "r2-state.v1"


def _canonical(value: Any) -> Any:
    if isinstance(value, float):
        return format(value, ".12g")
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, set):
        return sorted((_canonical(item) for item in value), key=repr)
    return value


def _digest(payload: Any) -> str:
    encoded = json.dumps(_canonical(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_state_key(
    graph: GraphSnapshot,
    level: int,
    terminal: bool,
    recent_action_history: Sequence[str],
    dynamic_context: Mapping[str, Any],
    *,
    history_window: int = 8,
) -> str:
    """Hash graph topology/features plus bounded action/context state."""
    graph_payload = {
        "nodes": sorted((node.node_id, node.object_id, tuple(node.features)) for node in graph.nodes),
        "edges": sorted((edge.src, edge.dst, edge.type, edge.weight, tuple(edge.evidence)) for edge in graph.edges),
    }
    payload = {
        "graph": graph_payload,
        "level": int(level),
        "terminal": bool(terminal),
        "recent_action_history": list(recent_action_history[-history_window:]),
        "dynamic_context": _canonical(dynamic_context),
    }
    return _digest(payload)


@dataclass(frozen=True)
class TransitionRecord:
    state_before: str
    action: str
    state_after: str
    tick: int = 0
    progressed: bool = False


class TransitionTable:
    """FIFO-bounded transitions with an index that is evicted atomically."""

    def __init__(self, max_entries: int = 1024) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._records: OrderedDict[str, TransitionRecord] = OrderedDict()
        self.transition_index: dict[str, set[str]] = {}

    def add(self, record: TransitionRecord) -> None:
        if record.state_before in self._records:
            self._remove(record.state_before)
        self._records[record.state_before] = record
        self.transition_index.setdefault(record.state_before, set()).add(record.action)
        while len(self._records) > self.max_entries:
            oldest, _ = self._records.popitem(last=False)
            self.transition_index.pop(oldest, None)

    def _remove(self, state_key: str) -> None:
        self._records.pop(state_key, None)
        self.transition_index.pop(state_key, None)

    def get(self, state_key: str) -> TransitionRecord | None:
        return self._records.get(state_key)

    def values(self) -> list[TransitionRecord]:
        return list(self._records.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_entries": self.max_entries,
            "records": [record.__dict__ for record in self._records.values()],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TransitionTable":
        table = cls(int(value.get("max_entries", 1024)))
        for raw in value.get("records", ()):
            table.add(TransitionRecord(**dict(raw)))
        return table


@dataclass
class StateTracker:
    loop_window: int = 8
    visited: deque[str] = field(default_factory=deque)
    visited_set: set[str] = field(default_factory=set)
    frontier: list[str] = field(default_factory=list)
    dynamic_context: dict[str, Any] = field(default_factory=dict)
    transitions: TransitionTable = field(default_factory=TransitionTable)

    def __post_init__(self) -> None:
        if self.loop_window < 1:
            raise ValueError("loop_window must be positive")

    def observe(self, state_key: str) -> bool:
        """Record ordered recent visitation and return whether a short loop exists."""
        self.visited.append(state_key)
        self.visited_set.add(state_key)
        while len(self.visited) > self.loop_window:
            self.visited.popleft()
        window = list(self.visited)
        return len(window) >= 2 and len(window) != len(set(window))

    def record_transition(self, state_before: str, action: str, state_after: str, *, tick: int = 0, progressed: bool = False) -> None:
        self.transitions.add(TransitionRecord(state_before, action, state_after, tick, progressed))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "loop_window": self.loop_window,
            "visited": list(self.visited),
            "visited_set": sorted(self.visited_set),
            "frontier": list(self.frontier),
            "dynamic_context": _canonical(self.dynamic_context),
            "transitions": self.transitions.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StateTracker":
        if str(value.get("schema_version", STATE_SCHEMA_VERSION)) != STATE_SCHEMA_VERSION:
            raise ValueError("unsupported state schema version")
        tracker = cls(
            loop_window=int(value.get("loop_window", 8)),
            visited=deque(str(item) for item in value.get("visited", ())),
            visited_set=set(str(item) for item in value.get("visited_set", ())),
            frontier=[str(item) for item in value.get("frontier", ())],
            dynamic_context=dict(value.get("dynamic_context", {})),
            transitions=TransitionTable.from_dict(value.get("transitions", {})),
        )
        if set(tracker.visited) - tracker.visited_set:
            raise ValueError("visited_set does not cover visited window")
        return tracker


class PersistentStateStore:
    """Atomic JSON persistence with schema and payload hash verification."""

    def save(self, path: str | os.PathLike[str], tracker: StateTracker) -> None:
        payload = tracker.to_dict()
        envelope = {"schema_version": STATE_SCHEMA_VERSION, "payload": payload, "payload_hash": _digest(payload)}
        destination = Path(path)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, destination)

    def load(self, path: str | os.PathLike[str]) -> StateTracker:
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        if envelope.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ValueError("unsupported persisted state schema version")
        payload = envelope.get("payload", {})
        if envelope.get("payload_hash") != _digest(payload):
            raise ValueError("persisted state hash mismatch")
        return StateTracker.from_dict(payload)
