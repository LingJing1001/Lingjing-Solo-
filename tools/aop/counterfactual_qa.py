"""Validation helpers for same-state counterfactual candidate groups."""
from __future__ import annotations
from collections import defaultdict
from typing import Any

from .action_schema import normalize_action_name


def validate_groups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors: list[str] = []
    for index, row in enumerate(rows):
        gid = row.get("counterfactual_group_id")
        if not isinstance(gid, str) or not gid:
            errors.append(f"row {index}: missing counterfactual_group_id")
            continue
        groups[gid].append(row)
        legal = row.get("legal_actions") or []
        legal_names = {normalize_action_name(value) for value in legal}
        action = normalize_action_name((row.get("requested_action") or {}).get("name"))
        if action not in legal_names:
            errors.append(f"row {index}: candidate {action!r} is not legal")
        if row.get("replay_verified") is not True:
            errors.append(f"row {index}: replay_verified is not true")
    comparable = 0
    informative = 0
    for gid, members in groups.items():
        hashes = {row.get("replay_state_hash") for row in members}
        candidates = [(row.get("requested_action") or {}).get("name") for row in members]
        if len(members) < 2:
            errors.append(f"group {gid}: fewer than two candidates")
        if len(hashes) != 1:
            errors.append(f"group {gid}: pre-state hashes differ")
        if len(set(candidates)) != len(candidates):
            errors.append(f"group {gid}: duplicate candidate")
        if len(members) >= 2 and len(hashes) == 1:
            comparable += 1
        rewards = {(bool(row.get("state_delta", {}).get("progressed")), bool(row.get("state_delta", {}).get("action_effective"))) for row in members}
        informative += int(len(rewards) > 1)
    return {
        "rows": len(rows), "groups": len(groups), "comparable_groups": comparable,
        "informative_groups": informative, "errors": errors,
        "pass": not errors and comparable == len(groups),
    }
