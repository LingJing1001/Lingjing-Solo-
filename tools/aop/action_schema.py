"""Canonical ACTIONn normalization at the AOP data/runtime boundary."""
from __future__ import annotations

from typing import Any


def normalize_action_name(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError(f"invalid action value: {value!r}")
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        number = int(value)
        if number < 1:
            raise ValueError(f"invalid action id: {number}")
        return f"ACTION{number}"
    if isinstance(value, str) and value.startswith("ACTION") and value[6:].isdigit():
        return value
    raise ValueError(f"invalid action value: {value!r}")


def normalize_label(label: dict[str, Any]) -> dict[str, Any]:
    row = dict(label)
    action = row.get("requested_action")
    if not isinstance(action, dict):
        raise ValueError("label missing requested_action")
    row["requested_action"] = {**action, "name": normalize_action_name(action.get("name", action.get("id")))}
    legal = row.get("legal_actions")
    if legal is None:
        before = row.get("observation_before")
        legal = before.get("legal_actions") if isinstance(before, dict) else []
    if not isinstance(legal, list):
        raise ValueError("legal_actions must be a list")
    row["legal_actions"] = [normalize_action_name(value) for value in legal]
    return row


def action_names_from_labels(labels: list[dict[str, Any]]) -> list[str]:
    names = set()
    for label in labels:
        row = normalize_label(label)
        names.add(row["requested_action"]["name"])
        names.update(row["legal_actions"])
    return sorted(names, key=lambda name: int(name[6:]))
