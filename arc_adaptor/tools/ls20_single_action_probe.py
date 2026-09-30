"""Probe one LS20 action and report observable frame differences.

Usage:
  uv run python tools/ls20_single_action_probe.py ACTION1

The ARC API key must come from the process environment/.env; it is never
embedded in this file. Each invocation creates a fresh game and performs one
non-reset action after reset.
"""
from __future__ import annotations

import sys
from collections import Counter

import numpy as np
from arc_agi import Arcade  # type: ignore[import-not-found]
from arcengine import GameAction  # type: ignore[import-not-found]


def plane(obs: object) -> np.ndarray:
    frame = np.asarray(getattr(obs, "frame"))
    return frame[0] if frame.ndim == 3 else frame


def positions(grid: np.ndarray, color: int) -> list[tuple[int, int]]:
    return [(int(p[0]), int(p[1])) for p in np.argwhere(grid == color)]


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1].upper() not in {
        "ACTION1", "ACTION2", "ACTION3", "ACTION4"
    }:
        print("usage: ls20_single_action_probe.py ACTION1|ACTION2|ACTION3|ACTION4")
        return 2

    action_name = sys.argv[1].upper()
    action = GameAction[action_name]
    arcade = Arcade()
    info = next((e for e in arcade.get_environments() if e.game_id.startswith("ls20")), None)
    if info is None:
        print("ERROR no ls20 environment returned")
        return 1

    env = arcade.make(info.game_id)
    before = env.reset()
    if before is None:
        print("ERROR reset returned None")
        return 1
    after = env.step(action)
    if after is None:
        print("ERROR action returned None")
        return 1

    before_grid = plane(before)
    after_grid = plane(after)
    changed = np.argwhere(before_grid != after_grid)
    before_player = positions(before_grid, 1)
    after_player = positions(after_grid, 1)
    if len(changed):
        changed_rows = (int(changed[:, 0].min()), int(changed[:, 0].max()))
        changed_cols = (int(changed[:, 1].min()), int(changed[:, 1].max()))
    else:
        changed_rows = changed_cols = None
    print(f"game_id={info.game_id}")
    print(f"action={action_name} action_id={action.value}")
    print(f"before_state={getattr(before, 'state', None)} after_state={getattr(after, 'state', None)}")
    print(f"levels={getattr(before, 'levels_completed', None)}->{getattr(after, 'levels_completed', None)}")
    print(f"before_color1_count={len(before_player)} after_color1_count={len(after_player)}")
    print(f"before_color1_bbox={before_player[:32]}")
    print(f"after_color1_bbox={after_player[:32]}")
    print(f"changed_cells={len(changed)}")
    print(f"changed_bbox_rows={changed_rows} changed_bbox_cols={changed_cols}")
    print(f"changed_color_pairs={Counter((int(before_grid[tuple(p)]), int(after_grid[tuple(p)])) for p in changed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
