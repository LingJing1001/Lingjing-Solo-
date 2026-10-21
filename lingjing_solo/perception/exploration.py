"""ARC-AGI-3 exploration helpers for stage 6.

The functions are deliberately policy-only: they produce masked inputs and
ranked candidates, but never execute an environment action.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Mapping, Sequence

import numpy as np


class Priority(IntEnum):
    P0 = 0  # UI
    P1 = 1  # high salience
    P2 = 2
    P3 = 3
    P4 = 4  # background/noise


@dataclass(frozen=True)
class GameMask:
    mask: np.ndarray
    top_rows: int
    bottom_rows: int
    source: str = "fixed_ui_band"

    def apply(self, frame: Any) -> np.ndarray:
        array = np.asarray(frame).copy()
        if array.shape[:2] != self.mask.shape:
            raise ValueError("game mask and frame dimensions differ")
        if array.ndim == 3:
            array[~self.mask, :] = 0
        else:
            array[~self.mask] = 0
        return array


def detect_ui_mask(
    frames: Sequence[Any] | Any,
    *,
    top_ratio: float = 0.08,
    bottom_ratio: float = 0.08,
    min_band: int = 1,
) -> GameMask:
    """Detect conservative fixed top/bottom UI bands from one or more frames."""
    if not 0 <= top_ratio < 0.5 or not 0 <= bottom_ratio < 0.5:
        raise ValueError("UI band ratios must be in [0, 0.5)")
    sequence = list(frames) if isinstance(frames, (list, tuple)) else [frames]
    if not sequence:
        raise ValueError("at least one frame is required")
    first = np.asarray(sequence[0])
    if first.ndim not in (2, 3) or first.shape[0] < 2 or first.shape[1] < 2:
        raise ValueError("frames must be 2D or HWC 3D arrays")
    height, width = first.shape[:2]
    top = max(min_band, int(height * top_ratio)) if top_ratio else 0
    bottom = max(min_band, int(height * bottom_ratio)) if bottom_ratio else 0
    top, bottom = min(top, height), min(bottom, height - top)
    mask = np.ones((height, width), dtype=bool)
    if top:
        mask[:top, :] = False
    if bottom:
        mask[height - bottom :, :] = False
    return GameMask(mask, top, bottom)


def boundary_recheck(objects: Sequence[Any], frame_shape: tuple[int, int], *, edge_tolerance: int = 0) -> tuple[str, ...]:
    """Return object IDs needing a full-frame recheck after game-mask cropping."""
    height, width = frame_shape[:2]
    ids: list[str] = []
    for obj in objects:
        x0, y0, x1, y1 = obj.bbox
        if x0 <= edge_tolerance or y0 <= edge_tolerance or x1 >= width - edge_tolerance or y1 >= height - edge_tolerance:
            ids.append(obj.object_id)
    return tuple(sorted(ids))


@dataclass(frozen=True)
class PriorityWeights:
    visual_salience: float = 0.5
    dpl_geometry_score: float = 0.3
    motion_significance: float = 0.2

    def __post_init__(self) -> None:
        if min(self.visual_salience, self.dpl_geometry_score, self.motion_significance) < 0:
            raise ValueError("priority weights must be non-negative")
        if self.visual_salience + self.dpl_geometry_score + self.motion_significance <= 0:
            raise ValueError("at least one priority weight must be positive")


@dataclass(frozen=True)
class ActionCandidate:
    action: str
    priority: Priority = Priority.P3
    visual_salience: float = 0.0
    dpl_geometry_score: float = 0.0
    motion_significance: float = 0.0
    tested: bool = False
    reachable: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def priority_score(self) -> float:
        return self.visual_salience * 0.5 + self.dpl_geometry_score * 0.3 + self.motion_significance * 0.2


def prioritize_actions(candidates: Sequence[ActionCandidate], weights: PriorityWeights = PriorityWeights()) -> list[ActionCandidate]:
    """Sort P0-P4 candidates; weights are configuration, not claimed optimum."""
    def score(candidate: ActionCandidate) -> float:
        return (
            weights.visual_salience * candidate.visual_salience
            + weights.dpl_geometry_score * candidate.dpl_geometry_score
            + weights.motion_significance * candidate.motion_significance
        )
    return sorted(candidates, key=lambda item: (item.tested, int(item.priority), -score(item), item.action))


class FrontierManager:
    """Maintain multi-source BFS distance to states with untested actions."""

    def __init__(self) -> None:
        self.transitions: dict[str, set[str]] = {}
        self.untested: dict[str, set[str]] = {}
        self.distance: dict[str, int] = {}

    def update_state(self, state: str, action: str, next_state: str, *, action_tested: bool = True) -> None:
        self.transitions.setdefault(state, set()).add(next_state)
        self.untested.setdefault(state, set())
        if action_tested:
            self.untested[state].discard(action)
        else:
            self.untested[state].add(action)

    def mark_untested(self, state: str, actions: Sequence[str]) -> None:
        self.untested.setdefault(state, set()).update(actions)

    def recompute_distances(self, current_states: Sequence[str] | str) -> dict[str, int]:
        sources = [current_states] if isinstance(current_states, str) else list(current_states)
        frontier = [state for state in self.untested if self.untested[state]]
        self.distance = {state: 0 for state in frontier}
        queue = deque(frontier)
        reverse: dict[str, set[str]] = {}
        for source, targets in self.transitions.items():
            for target in targets:
                reverse.setdefault(target, set()).add(source)
        while queue:
            target = queue.popleft()
            for predecessor in reverse.get(target, set()):
                if predecessor not in self.distance:
                    self.distance[predecessor] = self.distance[target] + 1
                    queue.append(predecessor)
        return {state: self.distance.get(state, -1) for state in sources}

    def nearest_frontier(self, current_state: str) -> str | None:
        self.recompute_distances(current_state)
        reachable = [(distance, state) for state, distance in self.distance.items() if distance >= 0 and state in self.untested and self.untested[state]]
        return min(reachable)[1] if reachable else None


def select_action(
    current_state: str,
    candidates: Sequence[ActionCandidate],
    frontier: FrontierManager,
    *,
    priority_threshold: Priority = Priority.P1,
    dpl_gate_passed: bool = True,
) -> ActionCandidate | None:
    """Three-layer selection: local, navigate to frontier, then broaden threshold."""
    if not dpl_gate_passed:
        return None
    ranked = prioritize_actions(candidates)
    local = [item for item in ranked if not item.tested and item.reachable and item.priority <= priority_threshold]
    if local:
        return local[0]
    target = frontier.nearest_frontier(current_state)
    if target and target != current_state:
        return next((item for item in ranked if item.reachable and not item.tested), None)
    broadened = [item for item in ranked if not item.tested and item.reachable]
    return broadened[0] if broadened else None
