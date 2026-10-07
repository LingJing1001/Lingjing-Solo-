"""Episode-disjoint dataset utilities."""
from __future__ import annotations

from typing import Any, Iterable


def split_labels(labels: Iterable[dict[str, Any]], validation_fraction: float = 0.2) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for label in labels:
        episode = label.get("episode_id")
        if not isinstance(episode, str) or not episode:
            raise ValueError("label missing episode_id")
        by_episode.setdefault(episode, []).append(label)
    if len(by_episode) < 2:
        raise ValueError("need at least two episodes for a leakage-safe split")
    episodes = sorted(by_episode)
    valid_count = max(1, round(len(episodes) * validation_fraction))
    valid_ids = set(episodes[-valid_count:])
    train = [x for ep in episodes if ep not in valid_ids for x in by_episode[ep]]
    valid = [x for ep in episodes if ep in valid_ids for x in by_episode[ep]]
    return train, valid
