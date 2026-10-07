"""Convert one-action recording rows into AOP training labels.

The converter is deliberately stdlib-only and fail-closed. A pixel delta is
reported as an observation fact, never promoted to action success or progress.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable
import hashlib
import json


class LabelError(ValueError):
    """Raised when a recording row cannot produce a trustworthy label."""


@dataclass(frozen=True)
class AOPLabel:
    episode_id: str
    step_id: int
    requested_action: dict[str, Any]
    observation_before: dict[str, Any]
    observation_after: dict[str, Any]
    state_delta: dict[str, Any]
    reward: float
    level_completed: int
    legal_actions: list[Any]
    temporal: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _action(row: dict[str, Any]) -> dict[str, Any]:
    action = row.get("requested_action")
    if not isinstance(action, dict) or not action.get("name") or action.get("name") == "RESET":
        raise LabelError("row has no non-RESET requested_action")
    return {"name": str(action["name"]), "id": action.get("id")}


def _obs(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    if not isinstance(value, dict):
        raise LabelError(f"row missing object field: {key}")
    return value


def _number(obs: dict[str, Any], *paths: tuple[str, ...], default: float = 0.0) -> float:
    for path in paths:
        cur: Any = obs
        for part in path:
            if not isinstance(cur, dict) or part not in cur:
                cur = None
                break
            cur = cur[part]
        if isinstance(cur, (int, float)) and not isinstance(cur, bool):
            return float(cur)
    return default


def row_to_label(row: dict[str, Any], *, episode_id: str | None = None, step_id: int | None = None) -> AOPLabel:
    if not isinstance(row, dict):
        raise LabelError("recording row must be an object")
    before = _obs(row, "observation")
    after = _obs(row, "data")
    action = _action(row)
    episode = episode_id or str(row.get("episode_id") or after.get("episode_id") or "")
    if not episode:
        raise LabelError("missing episode_id")
    step = step_id if step_id is not None else row.get("tick")
    if not isinstance(step, int) or step < 0:
        raise LabelError("step_id/tick must be a non-negative integer")

    before_state = before.get("state")
    after_state = after.get("state")
    before_levels = _number(before, ("levels_completed",), ("game_specific", "levels_completed"))
    after_levels = _number(after, ("levels_completed",), ("game_specific", "levels_completed"))
    before_score = _number(before, ("score",), ("game_specific", "score"))
    after_score = _number(after, ("score",), ("game_specific", "score"))
    before_hash = before.get("state_hash") or _hash_json(before)
    after_hash = after.get("state_hash") or _hash_json(after)
    frame_before = before.get("frame") or before.get("grid")
    frame_after = after.get("frame") or after.get("grid")
    visual_changed = frame_before is not None and frame_after is not None and frame_before != frame_after
    progressed = (after_levels > before_levels) or (after_score > before_score) or after_state == "WIN"

    return AOPLabel(
        episode_id=episode,
        step_id=step,
        requested_action=action,
        observation_before=before,
        observation_after=after,
        state_delta={
            "before_hash": before_hash,
            "after_hash": after_hash,
            "state_changed": before_hash != after_hash,
            "visual_changed": visual_changed,
            "action_effective": before_hash != after_hash or before_state != after_state,
            "progressed": progressed,
            "state_before": before_state,
            "state_after": after_state,
            "levels_delta": int(after_levels - before_levels),
            "score_delta": after_score - before_score,
        },
        reward=1.0 if progressed else 0.0,
        level_completed=int(after_levels),
        legal_actions=list(after.get("legal_actions") or before.get("legal_actions") or []),
        temporal={"tick_before": before.get("tick"), "tick_after": after.get("tick", step), "dt": 1},
    )


def convert_rows(rows: Iterable[dict[str, Any]]) -> list[AOPLabel]:
    labels: list[AOPLabel] = []
    for row in rows:
        labels.append(row_to_label(row))
    return labels
