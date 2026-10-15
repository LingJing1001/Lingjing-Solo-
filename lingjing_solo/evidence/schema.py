"""Versioned, fail-closed evidence contracts for replayed sandbox episodes."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Iterable


class EventValidationError(ValueError):
    pass


_REQUIRED = ("schema_version", "episode_id", "environment_id", "environment_version", "tick", "action", "transition", "provenance", "authority_status")
_ALLOWED = {"shadow", "quarantined", "rejected"}


def _finite(value: Any, path: str = "value") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise EventValidationError(f"non-finite numeric value at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite(item, f"{path}[{index}]")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise EventValidationError(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True)
class EpisodeEvent:
    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EpisodeEvent":
        if not isinstance(payload, dict):
            raise EventValidationError("event must be an object")
        missing = [field for field in _REQUIRED if field not in payload]
        if missing:
            raise EventValidationError("missing required fields: " + ", ".join(missing))
        if payload["schema_version"] != "episode_event.v1":
            raise EventValidationError("unsupported schema_version")
        for field in ("episode_id", "environment_id", "environment_version", "observation_ref", "replay_hash"):
            if field in payload:
                _text(payload[field], field)
        if not isinstance(payload["tick"], int) or isinstance(payload["tick"], bool) or payload["tick"] < 0:
            raise EventValidationError("tick must be a non-negative integer")
        if payload["authority_status"] not in _ALLOWED:
            raise EventValidationError("authority_status must remain shadow/quarantined/rejected")
        if not isinstance(payload["action"], dict) or not isinstance(payload["action"].get("name"), str) or not payload["action"].get("requested", False):
            raise EventValidationError("action must preserve a requested action name")
        if not isinstance(payload["transition"], dict) or not payload["transition"].get("settled", False):
            raise EventValidationError("transition must identify a settled frame")
        if not isinstance(payload["provenance"], dict):
            raise EventValidationError("provenance must be an object")
        for field in ("source", "source_commit", "input_hash"):
            _text(payload["provenance"].get(field), f"provenance.{field}")
        _finite(payload)
        return cls(dict(payload))

    @property
    def authority_status(self) -> str:
        return self.payload["authority_status"]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.payload, sort_keys=True, ensure_ascii=False))


def validate_event_rows(rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    last: dict[tuple[str, str], int] = {}
    for raw in rows:
        event = EpisodeEvent.from_dict(raw)
        key = (event.payload["episode_id"], event.payload.get("segment", "0"))
        tick = event.payload["tick"]
        if event.payload["action"]["name"] == "RESET":
            last.pop(key, None)
        elif key in last and tick <= last[key]:
            raise EventValidationError("episode ticks are not monotonic")
        last[key] = tick
        count += 1
    if not count:
        raise EventValidationError("event stream is empty")
    return count
