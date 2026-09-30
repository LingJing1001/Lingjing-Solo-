"""Run one R11L click probe and emit only observable evidence.

Usage:
  uv run python tools/r11l_single_action_probe.py --x 32 --y 32

The API key is read by the ARC SDK from the process environment/.env. This
probe creates a fresh R11L scorecard, resets once, sends one ACTION6 click,
and prints JSON evidence for the two observed frames.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

import numpy as np
from arc_agi import Arcade  # type: ignore[import-not-found]
from arcengine import GameAction  # type: ignore[import-not-found]

GAME_ID = "r11l-495a7899"


def plane(observation: Any) -> np.ndarray:
    frame = np.asarray(getattr(observation, "frame"), dtype=np.int16)
    return frame[0] if frame.ndim == 3 else frame


def frame_hash(grid: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(grid).tobytes()).hexdigest()


def evidence(before: Any, after: Any, x: int, y: int) -> dict[str, Any]:
    before_grid = plane(before)
    after_grid = plane(after)
    changed = np.argwhere(before_grid != after_grid)
    bbox = None
    if len(changed):
        bbox = {
            "row_min": int(changed[:, 0].min()),
            "row_max": int(changed[:, 0].max()),
            "col_min": int(changed[:, 1].min()),
            "col_max": int(changed[:, 1].max()),
        }
    server_action_input = getattr(after, "action_input", None)
    server_action_input = json.loads(
        json.dumps(
            server_action_input,
            default=lambda value: value.model_dump() if hasattr(value, "model_dump") else str(value),
        )
    )
    return {
        "game_id": GAME_ID,
        "action": {"name": "ACTION6", "id": GameAction.ACTION6.value, "x": x, "y": y},
        "before": {
            "frame_shape": list(before_grid.shape),
            "frame_hash": frame_hash(before_grid),
            "state": str(getattr(before, "state", None)),
            "levels_completed": getattr(before, "levels_completed", None),
            "available_actions": [getattr(a, "value", a) for a in before.available_actions],
        },
        "after": {
            "frame_shape": list(after_grid.shape),
            "frame_hash": frame_hash(after_grid),
            "state": str(getattr(after, "state", None)),
            "levels_completed": getattr(after, "levels_completed", None),
            "available_actions": [getattr(a, "value", a) for a in after.available_actions],
            "server_action_input": server_action_input,
        },
        "changed_cells": int(len(changed)),
        "changed_bbox": bbox,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, required=True)
    parser.add_argument("--y", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.x < 64 or not 0 <= args.y < 64:
        parser.error("--x and --y must be in the inclusive [0, 63] frame bounds")

    arcade = Arcade()
    info = next((item for item in arcade.get_environments() if item.game_id == GAME_ID), None)
    if info is None:
        print(json.dumps({"error": "R11L environment not returned", "game_id": GAME_ID}))
        return 1
    env = arcade.make(GAME_ID, save_recording=True)
    before = env.reset()
    if before is None:
        print(json.dumps({"error": "reset returned None", "game_id": GAME_ID}))
        return 1
    legal_values = {getattr(action, "value", action) for action in before.available_actions}
    if GameAction.ACTION6.value not in legal_values:
        print(json.dumps({"error": "ACTION6 not legal after reset", "available_actions": sorted(legal_values)}))
        return 1
    action = GameAction.ACTION6
    action.set_data({"x": args.x, "y": args.y, "game_id": GAME_ID})
    after = env.step(action, data=action.action_data.model_dump())
    if after is None:
        print(json.dumps({"error": "click returned None", "game_id": GAME_ID}))
        return 1
    print(json.dumps(evidence(before, after, args.x, args.y), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
