"""Leakage-safe dataset splitting utilities."""
from __future__ import annotations

from typing import Any, Iterable


def _split_key(label: dict[str, Any]) -> tuple[str, str]:
    """Group counterfactual branches before splitting; otherwise use episode."""
    group = label.get("counterfactual_group_id")
    if isinstance(group, str) and group:
        return ("counterfactual", group)
    episode = label.get("episode_id")
    if not isinstance(episode, str) or not episode:
        raise ValueError("label missing episode_id")
    return ("episode", episode)


def split_labels(labels: Iterable[dict[str, Any]], validation_fraction: float = 0.2) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for label in labels:
        by_group.setdefault(_split_key(label), []).append(label)
    if len(by_group) < 2:
        raise ValueError("need at least two episodes/groups for a leakage-safe split")
    groups = sorted(by_group)
    valid_count = max(1, round(len(groups) * validation_fraction))
    valid_ids = {
        groups[min(len(groups) - 1, (index * len(groups)) // valid_count)]
        for index in range(valid_count)
    }
    train = [x for group in groups if group not in valid_ids for x in by_group[group]]
    valid = [x for group in groups if group in valid_ids for x in by_group[group]]
    return train, valid
