#!/usr/bin/env python3
"""Normalize ARC-AGI-3 Recorder JSONL into AOP transition JSONL.

The ARC recorder stores one event per observed frame. The vendored recorder
patch attaches requested_action to the settled frame returned by that action.
This tool pairs each action-bearing frame with the immediately preceding frame;
it never guesses missing actions and fails closed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _event_data(event: dict, line_no: int) -> dict:
    data = event.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"line {line_no}: recorder event data must be an object")
    return data


def normalize(input_path: Path, output_path: Path, episode_id: str | None = None) -> int:
    events: list[dict] = []
    direct: list[dict] = []
    with input_path.open(encoding="utf-8") as src:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no}: invalid JSON") from exc
            if not isinstance(event, dict):
                raise ValueError(f"line {line_no}: event must be an object")
            data = event.get("data")
            if not isinstance(data, dict):
                raise ValueError(f"line {line_no}: recorder event data must be an object")
            # The official remote recorder uses action_input, while the local
            # AOP recorder emits requested_action. Convert the former without
            # inferring anything beyond the recorded action id.
            if "requested_action" not in data:
                action_input = data.get("action_input")
                if isinstance(action_input, dict) and isinstance(action_input.get("id"), str):
                    action_name = action_input["id"]
                    action_id = None
                    if action_name.startswith("ACTION") and action_name[6:].isdigit():
                        action_id = int(action_name[6:])
                    data = {**data, "requested_action": {"name": action_name, "id": action_id}}
            observation = event.get("observation")
            if isinstance(observation, dict) and data.get("requested_action") is not None:
                direct.append({"episode_id": episode_id or data.get("episode_id"),
                               "tick": data.get("tick"), "requested_action": data.get("requested_action"),
                               "observation": observation, "data": data})
            elif "frame" in data or "grid" in data:
                events.append(data)

    rows: list[dict] = list(direct)
    if events:
        if len(events) < 2:
            raise ValueError("recording must contain an initial frame and one action result")
        first = events[0]
        resolved_episode = episode_id or str(first.get("guid") or first.get("game_id") or "")
        if not resolved_episode:
            raise ValueError("cannot derive episode_id; pass --episode-id")
        for tick, (before, after) in enumerate(zip(events, events[1:])):
            action = after.get("requested_action")
            if action is None:
                continue
            if not isinstance(action, dict) or not isinstance(action.get("name"), str):
                raise ValueError(f"transition {tick}: requested_action must contain name")
            rows.append({"episode_id": resolved_episode, "tick": tick,
                         "requested_action": action,
                         "observation": {**before, "tick": tick,
                                         "legal_actions": before.get("available_actions", before.get("legal_actions", []))},
                         "data": {**after, "tick": tick + 1,
                                  "legal_actions": after.get("available_actions", after.get("legal_actions", []))}})

    if not rows:
        raise ValueError("recording contains no action-bearing settled frame")
    for index, row in enumerate(rows):
        if not row.get("episode_id"):
            raise ValueError(f"transition {index}: cannot derive episode_id")
        if not isinstance(row.get("tick"), int):
            raise ValueError(f"transition {index}: tick must be an integer")
        if not isinstance(row.get("requested_action"), dict) or not isinstance(row["requested_action"].get("name"), str):
            raise ValueError(f"transition {index}: requested_action must contain name")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as dst:
        for row in rows:
            dst.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episode-id")
    args = parser.parse_args()
    try:
        written = normalize(args.input, args.output, args.episode_id)
    except (OSError, ValueError) as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 1
    print(f"NORMALIZED transitions={written} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
