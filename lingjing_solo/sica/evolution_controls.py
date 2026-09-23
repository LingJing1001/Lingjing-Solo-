"""SICA phase-3 controls: R7, R8 and the remaining R4 limits."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class VersionSnapshot:
    snapshot_id: str
    path: str
    state_hash: str
    metadata: dict[str, Any]


class SnapshotStore:
    """Atomic, content-addressed JSON snapshots with an append-only manifest."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.jsonl"

    def create(self, snapshot_id: str, state: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None) -> VersionSnapshot:
        if not snapshot_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_ ." for c in snapshot_id):
            raise ValueError("snapshot_id contains unsafe characters")
        payload = {"state": dict(state), "metadata": dict(metadata or {})}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        path = self.root / f"{snapshot_id}-{digest[:16]}.json"
        if not path.exists():
            fd, tmp_name = tempfile.mkstemp(prefix=".snapshot-", dir=self.root)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_name, path)
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
        item = VersionSnapshot(snapshot_id, str(path), digest, dict(metadata or {}))
        with self.manifest_path.open("a", encoding="utf-8") as manifest:
            manifest.write(json.dumps(asdict(item), sort_keys=True, ensure_ascii=False) + "\n")
            manifest.flush()
            os.fsync(manifest.fileno())
        return item

    def restore(self, snapshot: VersionSnapshot | str) -> dict[str, Any]:
        path = Path(snapshot.path if isinstance(snapshot, VersionSnapshot) else snapshot)
        resolved = path.resolve(strict=False)
        if resolved.parent != self.root.resolve() or resolved.suffix != ".json":
            raise PermissionError("snapshot path is outside the snapshot store")
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("state"), dict):
            raise ValueError("invalid snapshot payload")
        return payload["state"]


@dataclass(frozen=True)
class PerformanceDecision:
    degraded: bool
    baseline: float
    candidate: float
    delta: float
    relative_delta: float
    reason: str


class PerformanceMonitor:
    def __init__(self, *, max_absolute_drop: float = 0.0, max_relative_drop: float = 0.0) -> None:
        if max_absolute_drop < 0 or max_relative_drop < 0:
            raise ValueError("performance tolerances must be non-negative")
        self.max_absolute_drop = float(max_absolute_drop)
        self.max_relative_drop = float(max_relative_drop)

    def evaluate(self, baseline: float, candidate: float) -> PerformanceDecision:
        baseline = float(baseline)
        candidate = float(candidate)
        delta = candidate - baseline
        relative = delta / abs(baseline) if baseline else (0.0 if delta >= 0 else float("-inf"))
        degraded = delta < -self.max_absolute_drop and relative < -self.max_relative_drop
        return PerformanceDecision(degraded, baseline, candidate, delta, relative,
                                   "performance regression" if degraded else "performance acceptable")


class RollbackManager:
    def __init__(self, store: SnapshotStore, monitor: PerformanceMonitor) -> None:
        self.store = store
        self.monitor = monitor
        self.last_decision: PerformanceDecision | None = None
        self.last_restored_snapshot: str | None = None
        self.rollback_count = 0

    def evaluate_and_rollback(self, snapshot: VersionSnapshot, baseline: float, candidate: float,
                              *, restore: Callable[[dict[str, Any]], None]) -> PerformanceDecision:
        decision = self.monitor.evaluate(baseline, candidate)
        self.last_decision = decision
        if decision.degraded:
            state = self.store.restore(snapshot)
            restore(state)
            self.last_restored_snapshot = snapshot.snapshot_id
            self.rollback_count += 1
        return decision


@dataclass
class EvolutionController:
    cooldown_ticks: int = 0
    exploration_quota: int | None = None
    quota_window: int = 1
    max_meta_depth: int = 2
    _last_accept_tick: int | None = None
    _window_start: int = 0
    _exploration_used: int = 0

    def __post_init__(self) -> None:
        if self.cooldown_ticks < 0 or self.quota_window < 1 or self.max_meta_depth < 0:
            raise ValueError("invalid phase-3 control limit")
        if self.exploration_quota is not None and self.exploration_quota < 0:
            raise ValueError("exploration_quota must be non-negative")

    def _roll_window(self, tick: int) -> None:
        if tick - self._window_start >= self.quota_window:
            self._window_start = tick
            self._exploration_used = 0

    def admit(self, *, tick: int, recursion_depth: int, is_exploration: bool) -> None:
        if recursion_depth > self.max_meta_depth:
            raise PermissionError(f"meta-modification depth {recursion_depth} exceeds {self.max_meta_depth}")
        if self._last_accept_tick is not None and tick - self._last_accept_tick < self.cooldown_ticks:
            raise PermissionError("SICA cooldown period is active")
        self._roll_window(tick)
        if is_exploration and self.exploration_quota is not None and self._exploration_used >= self.exploration_quota:
            raise PermissionError("exploration quota exhausted")

    def commit(self, *, tick: int, is_exploration: bool) -> None:
        self._last_accept_tick = tick
        if is_exploration:
            self._exploration_used += 1

    @property
    def exploration_remaining(self) -> int | None:
        if self.exploration_quota is None:
            return None
        return max(0, self.exploration_quota - self._exploration_used)
