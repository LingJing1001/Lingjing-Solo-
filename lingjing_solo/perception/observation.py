"""R4 normalized observation contract.

The detector consumes this boundary object instead of parsing harness payloads.
Frames use row-major ``(y, x)`` coordinates and are normalized to a 2-D numeric
array. ``frame_id`` is evidence metadata only; it is never used as an answer
or level lookup.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class ActionRecord:
    action: str
    x: Optional[int] = None
    y: Optional[int] = None
    outcome: Optional[str] = None


@dataclass(frozen=True)
class HypothesisContext:
    """Field/Learner context passed in without exposing level constants."""

    hypothesis_space_type: Optional[str] = None
    relevance_by_source: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.hypothesis_space_type is not None and (
            not isinstance(self.hypothesis_space_type, str)
            or not self.hypothesis_space_type
        ):
            raise ValueError("hypothesis_space_type must be a non-empty string or None")
        for source, relevance in self.relevance_by_source:
            if not isinstance(source, str) or not source:
                raise ValueError("hypothesis relevance source must be a non-empty string")
            if not 0 <= relevance <= 1:
                raise ValueError("hypothesis relevance must be in [0, 1]")


@dataclass(frozen=True)
class NormalizedObservation:
    frame: np.ndarray
    width: int
    height: int
    frame_id: str
    previous_frame: Optional[np.ndarray] = None
    action_history: tuple[ActionRecord, ...] = ()
    legal_actions: Optional[tuple[str, ...]] = None
    game_family: str = "click"
    remaining_budget: Optional[int] = None
    current_hypothesis: Optional[HypothesisContext] = None
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        frame = np.asarray(self.frame)
        if frame.ndim != 2 or frame.size == 0:
            raise ValueError("frame must be a non-empty 2-D array")
        if frame.shape != (self.height, self.width):
            raise ValueError(
                f"frame dimensions do not match shape={frame.shape}, "
                f"height={self.height}, width={self.width}"
            )
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError("frame_id must be a non-empty evidence reference")
        if self.game_family not in {"click", "keyboard_click", "keyboard"}:
            raise ValueError(f"unsupported game_family: {self.game_family}")
        if self.remaining_budget is not None and self.remaining_budget < 0:
            raise ValueError("remaining_budget must be non-negative")


def _as_grid(frame: Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        return array.copy()
    if array.ndim == 3:
        if array.shape[-1] in (1, 3, 4):
            channels = array[..., :3]
        elif array.shape[0] in (1, 3, 4):
            channels = np.moveaxis(array[:3], 0, -1)
        else:
            raise ValueError(f"unsupported frame shape: {array.shape}")
        if channels.shape[-1] == 1:
            return channels[..., 0].copy()
        weights = np.asarray((0.299, 0.587, 0.114), dtype=np.float64)
        return np.rint(np.asarray(channels, dtype=np.float64) @ weights).astype(np.int32)
    raise ValueError(f"expected a 2-D grid or RGB frame, got shape={array.shape}")


def normalize_observation(
    frame: Any,
    *,
    frame_id: str,
    previous_frame: Any = None,
    action_history: tuple[ActionRecord, ...] = (),
    legal_actions: Optional[tuple[str, ...]] = None,
    game_family: str = "click",
    remaining_budget: Optional[int] = None,
    current_hypothesis: Optional[HypothesisContext] = None,
    evidence_refs: tuple[str, ...] = (),
) -> NormalizedObservation:
    """Convert a harness frame into the R4 contract with fail-closed inputs."""
    normalized = _as_grid(frame)
    previous = None if previous_frame is None else _as_grid(previous_frame)
    if previous is not None and previous.shape != normalized.shape:
        raise ValueError(
            f"previous_frame shape mismatch: {previous.shape} != {normalized.shape}"
        )
    if not np.isfinite(normalized).all():
        raise ValueError("frame contains NaN or infinite values")
    return NormalizedObservation(
        frame=normalized,
        width=int(normalized.shape[1]),
        height=int(normalized.shape[0]),
        frame_id=frame_id,
        previous_frame=previous,
        action_history=tuple(action_history),
        legal_actions=None if legal_actions is None else tuple(legal_actions),
        game_family=game_family,
        remaining_budget=remaining_budget,
        current_hypothesis=current_hypothesis,
        evidence_refs=tuple(evidence_refs),
    )


def observation_from_mapping(payload: Mapping[str, Any], **kwargs: Any) -> NormalizedObservation:
    """Build an observation from a recording-like mapping without reading level data."""
    if "frame" not in payload:
        raise ValueError("observation payload is missing frame")
    frame_id = kwargs.pop("frame_id", payload.get("frame_id"))
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("observation payload is missing frame_id")
    return normalize_observation(payload["frame"], frame_id=frame_id, **kwargs)
