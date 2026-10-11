"""Offline data-quality report for normalized AOP labels."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.aop.action_schema import normalize_label  # noqa: E402


def _read(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"line {line_no}: label must be an object")
                rows.append(normalize_label(row))
    return rows


def build_report(path: str | Path) -> dict[str, Any]:
    rows = _read(Path(path))
    episodes: dict[str, int] = Counter()
    games: dict[str, int] = Counter()
    actions: Counter[str] = Counter()
    legal_actions: Counter[str] = Counter()
    per_episode: dict[str, dict[str, Any]] = defaultdict(lambda: {"rows": 0, "actions": Counter()})
    missing: Counter[str] = Counter()
    duplicate_keys: Counter[tuple[str, Any]] = Counter()
    legal_rows = 0
    reset_rows = 0
    for row in rows:
        episode = row.get("episode_id")
        action = row.get("requested_action")
        action_name = action.get("name") if isinstance(action, dict) else None
        legal = row.get("legal_actions")
        if not isinstance(episode, str) or not episode:
            missing["episode_id"] += 1
            continue
        episodes[episode] += 1
        game = episode.split("_", 1)[0]
        games[game] += 1
        per_episode[episode]["rows"] += 1
        if isinstance(action_name, str):
            actions[action_name] += 1
            per_episode[episode]["actions"][action_name] += 1
        else:
            missing["requested_action.name"] += 1
        if not isinstance(legal, list):
            missing["legal_actions"] += 1
        elif isinstance(action_name, str):
            legal_actions.update(legal)
            legal_rows += int(action_name in legal)
        before = row.get("observation_before")
        state = before.get("state") if isinstance(before, dict) else None
        reset_rows += int(str(state).upper() == "RESET")
        transition_key = row.get("step_id", row.get("tick"))
        if transition_key is not None:
            duplicate_keys[(episode, transition_key)] += 1
    duplicate_rows = sum(count - 1 for count in duplicate_keys.values() if count > 1)
    return {
        "schema": "aop.data.qa.v1",
        "source": str(Path(path)),
        "rows": len(rows),
        "episodes": len(episodes),
        "games": len(games),
        "game_rows": dict(sorted(games.items())),
        "action_counts": dict(sorted(actions.items())),
        "legal_action_counts": dict(sorted(legal_actions.items())),
        "requested_actions_missing_from_legal": sorted(set(actions) - set(legal_actions)),
        "per_episode": {
            key: {"rows": value["rows"], "actions": dict(sorted(value["actions"].items()))}
            for key, value in sorted(per_episode.items())
        },
        "missing_fields": dict(sorted(missing.items())),
        "duplicate_transition_rows": duplicate_rows,
        "reset_rows": reset_rows,
        "requested_action_legal_rate": (legal_rows / len(rows)) if rows else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"QA rows={report['rows']} episodes={report['episodes']} games={report['games']} legal_rate={report['requested_action_legal_rate']:.4f} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
