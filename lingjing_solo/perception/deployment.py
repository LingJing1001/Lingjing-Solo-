"""Stage 7 offline replay, shadow-mode and fail-closed canary controls."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from .dpl_adapter import GateReport
from .world_model import ProjectionWorldModel

REPLAY_SCHEMA_VERSION = "r2-replay.v1"
REQUIRED_LOG_FIELDS = (
    "run_id", "episode_id", "tick", "frame_hash", "graph_hash", "action",
    "delta_count", "level", "terminal", "visibility", "dpl_availability",
    "object_changes", "transition_result", "errors", "latency_ms", "memory_kb",
)


def _hash_frame(frame: Any) -> str:
    return hashlib.sha256(memoryview(frame).tobytes()).hexdigest()


@dataclass(frozen=True)
class ReplayRecord:
    run_id: str
    episode_id: str
    tick: int
    frame_hash: str
    graph_hash: str
    action: str | None
    delta_count: int
    level: int
    terminal: bool
    visibility: str
    dpl_availability: Mapping[str, bool]
    object_changes: tuple[str, ...]
    transition_result: Mapping[str, Any]
    errors: tuple[str, ...] = ()
    latency_ms: float = 0.0
    memory_kb: int = 0
    schema_version: str = REPLAY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["object_changes"] = list(self.object_changes)
        value["errors"] = list(self.errors)
        return value

    def validate(self) -> None:
        missing = [name for name in REQUIRED_LOG_FIELDS if name not in self.to_dict()]
        if missing:
            raise ValueError(f"replay record missing fields: {missing}")
        if self.schema_version != REPLAY_SCHEMA_VERSION:
            raise ValueError("unsupported replay schema")
        if self.visibility not in {"full", "partial"}:
            raise ValueError("visibility must be full or partial")
        if self.visibility == "partial" and not self.errors:
            raise ValueError("partial visibility requires an explicit error/fail-closed reason")


class ReplayLogger:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ReplayRecord) -> None:
        record.validate()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


@dataclass(frozen=True)
class ReplayFrame:
    frame: Any
    action: str | None = None
    episode_id: str = "episode-0"
    level: int = 0
    terminal: bool = False
    visibility: str = "full"
    delta_count: int = 0


@dataclass
class OfflineReplayRunner:
    """Build snapshots and logs without calling an environment action API."""

    model: ProjectionWorldModel = field(default_factory=ProjectionWorldModel)
    run_id: str = "offline-run"

    def run(self, frames: Iterable[ReplayFrame], log_path: str | os.PathLike[str]) -> list[ReplayRecord]:
        logger = ReplayLogger(log_path)
        records: list[ReplayRecord] = []
        for frame in frames:
            started = time.perf_counter()
            errors: list[str] = []
            snapshot = None
            report: GateReport | None = None
            try:
                snapshot, report = self.model.update(frame.frame, frame.action or "NOOP", source_backend="offline_replay", visibility=frame.visibility)
                if frame.visibility == "partial":
                    errors.append("partial_visibility:planning_stopped")
                if not report.passed:
                    errors.extend(report.violations)
            except Exception as exc:
                errors.append(f"replay_error:{type(exc).__name__}:{exc}")
            record = ReplayRecord(
                run_id=self.run_id,
                episode_id=frame.episode_id,
                tick=self.model.tick - 1,
                frame_hash=_hash_frame(frame.frame),
                graph_hash=snapshot.graph.global_hash if snapshot is not None else "",
                action=None if errors and frame.visibility == "partial" else frame.action,
                delta_count=frame.delta_count,
                level=frame.level,
                terminal=frame.terminal,
                visibility=frame.visibility,
                dpl_availability=dict(report.dpl.validity) if report is not None else {},
                object_changes=snapshot.object_changes if snapshot is not None else (),
                transition_result={"executed": False, "offline": True, "fail_closed": bool(errors)},
                errors=tuple(errors),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                memory_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            )
            logger.append(record)
            records.append(record)
        return records


@dataclass(frozen=True)
class DeploymentPolicy:
    """Feature-flag policy for shadow/canary; default is non-invasive."""

    mode: str = "offline"
    enabled_levels: frozenset[int] = frozenset()
    enabled_episodes: frozenset[str] = frozenset()
    require_full_visibility: bool = True

    def allows(self, level: int, episode_id: str, visibility: str, *, graph_valid: bool) -> bool:
        if self.mode == "offline" or not graph_valid:
            return False
        if self.require_full_visibility and visibility != "full":
            return False
        if self.enabled_levels and level not in self.enabled_levels:
            return False
        if self.enabled_episodes and episode_id not in self.enabled_episodes:
            return False
        return self.mode in {"tier_b", "rendered", "shadow", "canary"}


@dataclass(frozen=True)
class ShadowComparison:
    old_action: str | None
    new_action: str | None
    graph_hash: str
    mismatch: bool
    fail_closed: bool


def compare_shadow(old_action: str | None, new_action: str | None, graph_hash: str, *, graph_valid: bool, visibility: str) -> ShadowComparison:
    """Compare actions without allowing the new planner to execute an action."""
    fail_closed = not graph_valid or visibility != "full" or not graph_hash
    return ShadowComparison(old_action, new_action if not fail_closed else None, graph_hash, old_action != new_action, fail_closed)
