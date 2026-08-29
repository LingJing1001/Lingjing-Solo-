"""受控单动作差分分析（R4 / Layer 2）。

该模块只记录观测证据，不把颜色或网格变化直接解释为胜利。
真实动作语义必须由 harness 提供 action 标签，并通过多条样本的一致性确认。
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class ActionObservation:
    """同一动作前后的观测对。"""

    action: str
    before: np.ndarray
    after: np.ndarray
    state_after: Optional[str] = None
    levels_completed_after: Optional[int] = None


@dataclass(frozen=True)
class ActionDelta:
    """一条单动作样本的可审计差分结果。"""

    action: str
    changed_pixels: int
    player_before: Optional[tuple[float, float]]
    player_after: Optional[tuple[float, float]]
    displacement: Optional[tuple[float, float]]
    state_after: Optional[str]
    levels_completed_after: Optional[int]

    @property
    def triggered_level_change(self) -> bool:
        return self.levels_completed_after is not None and self.levels_completed_after > 0


@dataclass(frozen=True)
class ActionSummary:
    """同一动作多次观测的聚合结果。"""

    action: str
    samples: int
    moved_samples: int
    displacements: tuple[tuple[float, float], ...]
    consistent_displacement: Optional[tuple[float, float]]
    confidence: float
    level_trigger_samples: int


def _normalize_grid(
    grid: np.ndarray, *, frame_channel: Optional[int] = None
) -> np.ndarray:
    array = np.asarray(grid)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        if array.shape[0] == 1:
            return array[0]
        if frame_channel is not None:
            if not 0 <= frame_channel < array.shape[0]:
                raise ValueError(
                    f"frame_channel out of range: {frame_channel} for shape={array.shape}"
                )
            return array[frame_channel]
    raise ValueError(f"expected HxW or 1xHxW grid, got shape={array.shape}")


def _centroid(grid: np.ndarray, color: int) -> Optional[tuple[float, float]]:
    ys, xs = np.where(grid == color)
    if len(xs) == 0:
        return None
    return (round(float(xs.mean()), 3), round(float(ys.mean()), 3))


def analyze_observation(sample: ActionObservation, *, player_color: int = 1) -> ActionDelta:
    """计算单次动作的观测差分；不推断方向名称或胜利条件。"""
    if not isinstance(sample.action, str) or not sample.action.strip():
        raise ValueError("action must be a non-empty string")
    before = _normalize_grid(sample.before)
    after = _normalize_grid(sample.after)
    if before.shape != after.shape:
        raise ValueError(f"before/after shape mismatch: {before.shape} != {after.shape}")
    player_before = _centroid(before, player_color)
    player_after = _centroid(after, player_color)
    displacement = None
    if player_before is not None and player_after is not None:
        displacement = (
            round(player_after[0] - player_before[0], 3),
            round(player_after[1] - player_before[1], 3),
        )
    return ActionDelta(
        action=sample.action,
        changed_pixels=int(np.count_nonzero(before != after)),
        player_before=player_before,
        player_after=player_after,
        displacement=displacement,
        state_after=sample.state_after,
        levels_completed_after=sample.levels_completed_after,
    )


def summarize_actions(
    observations: Iterable[ActionObservation], *, player_color: int = 1
) -> dict[str, ActionSummary]:
    """聚合同动作样本，只有完全一致的位移才给出确定性位移。"""
    grouped: dict[str, list[ActionDelta]] = defaultdict(list)
    for observation in observations:
        delta = analyze_observation(observation, player_color=player_color)
        grouped[delta.action].append(delta)

    summaries: dict[str, ActionSummary] = {}
    for action, deltas in grouped.items():
        displacements = tuple(
            delta.displacement for delta in deltas if delta.displacement is not None
        )
        unique = set(displacements)
        consistent = next(iter(unique)) if len(unique) == 1 and displacements else None
        moved = sum(delta.displacement not in (None, (0.0, 0.0)) for delta in deltas)
        confidence = 0.0
        if deltas:
            confidence = round((moved / len(deltas)) * (1.0 if consistent else 0.5), 3)
        summaries[action] = ActionSummary(
            action=action,
            samples=len(deltas),
            moved_samples=moved,
            displacements=displacements,
            consistent_displacement=consistent,
            confidence=confidence,
            level_trigger_samples=sum(delta.triggered_level_change for delta in deltas),
        )
    return summaries


def analyze_recording(
    path: str | Path, *, player_color: int = 1, frame_channel: Optional[int] = None
) -> tuple[ActionDelta, ...]:
    """从 ARC recording JSONL 生成有序的多动作差分结果。

    recording 的首个带 frame 记录作为 baseline；每个后续记录必须包含
    ``data.requested_action``，并将该动作关联到 baseline→当前帧的转移。
    缺少动作标签时显式失败，避免把未知动作误记为可用证据。
    """
    previous: Optional[tuple[np.ndarray, Optional[str], Optional[int]]] = None
    deltas: list[ActionDelta] = []
    with Path(path).open(encoding="utf-8") as recording:
        for line_number, line in enumerate(recording, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}") from exc
            data = record.get("data", record)
            if "frame" not in data:
                continue
            action = data.get("requested_action")
            if isinstance(action, dict):
                action = action.get("name") or action.get("id")
            if previous is None:
                previous = (
                    _normalize_grid(
                        np.asarray(data["frame"]), frame_channel=frame_channel
                    ),
                    data.get("state"),
                    data.get("levels_completed"),
                )
                continue
            if action is None or not str(action).strip():
                raise ValueError(f"requested_action missing at line {line_number}")
            current = _normalize_grid(
                np.asarray(data["frame"]), frame_channel=frame_channel
            )
            before, _, _ = previous
            deltas.append(
                analyze_observation(
                    ActionObservation(
                        action=str(action),
                        before=before,
                        after=current,
                        state_after=data.get("state"),
                        levels_completed_after=data.get("levels_completed"),
                    ),
                    player_color=player_color,
                )
            )
            previous = (current, data.get("state"), data.get("levels_completed"))
    return tuple(deltas)
