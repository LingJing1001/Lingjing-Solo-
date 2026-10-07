"""Fail-closed validation for ARC/CEAX recording JSONL before label generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import json


class RecordingValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    episodes: int
    resets: int
    actions: int


def _obj(row: dict[str, Any], key: str, line_no: int) -> dict[str, Any]:
    value = row.get(key)
    if not isinstance(value, dict):
        raise RecordingValidationError(f"line {line_no}: {key} must be an object")
    return value


def validate_rows(rows: Iterable[dict[str, Any]]) -> ValidationReport:
    total = episodes = resets = actions = 0
    seen: set[str] = set()
    last_tick: dict[str, int] = {}
    for line_no, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise RecordingValidationError(f"line {line_no}: row must be an object")
        total += 1
        episode = row.get("episode_id")
        if not isinstance(episode, str) or not episode:
            raise RecordingValidationError(f"line {line_no}: missing episode_id")
        if episode not in seen:
            seen.add(episode)
            episodes += 1
        before = _obj(row, "observation", line_no)
        after = _obj(row, "data", line_no)
        before_tick = before.get("tick")
        after_tick = after.get("tick")
        if not isinstance(before_tick, int) or not isinstance(after_tick, int) or after_tick <= before_tick:
            raise RecordingValidationError(f"line {line_no}: after must be a settled later tick")
        if episode in last_tick and before_tick < last_tick[episode]:
            raise RecordingValidationError(f"line {line_no}: episode ticks are not monotonic")
        last_tick[episode] = after_tick
        legal = before.get("legal_actions") or before.get("available_actions")
        if not isinstance(legal, list):
            raise RecordingValidationError(f"line {line_no}: missing before legal_actions")
        action = row.get("requested_action")
        if not isinstance(action, dict) or not isinstance(action.get("name"), str):
            raise RecordingValidationError(f"line {line_no}: missing requested_action")
        if action["name"] == "RESET":
            resets += 1
            last_tick.pop(episode, None)
            continue
        actions += 1
        if legal and action["name"] not in {str(x.get("name")) if isinstance(x, dict) else str(x) for x in legal}:
            raise RecordingValidationError(f"line {line_no}: requested action is not legal before the step")
        if not any(k in before for k in ("state", "levels_completed", "score", "game_specific")):
            raise RecordingValidationError(f"line {line_no}: no authoritative before-state/progress field")
    if total == 0:
        raise RecordingValidationError("recording is empty")
    if actions == 0:
        raise RecordingValidationError("recording contains no non-RESET action")
    return ValidationReport(total, episodes, resets, actions)


def validate_jsonl(path: str) -> ValidationReport:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RecordingValidationError(f"line {line_no}: invalid JSON") from exc
    return validate_rows(rows)
