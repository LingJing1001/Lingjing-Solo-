"""ls20 level-1 path probe with grid BFS."""
from __future__ import annotations

import heapq
import importlib.util
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(ROOT))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, GameState


def grid_of(frame) -> np.ndarray:
    raw = frame.frame or []
    layer = raw[-1] if raw and isinstance(raw[0][0], list) else raw
    return np.array(layer, dtype=np.int8)


# ls20 level 1 landmarks (from game source)
PLAYER0 = (34, 45)
GOAL0 = (34, 10)
ROT_PAD0 = (19, 30)
WALL_COLOR = 14  # ihdgageizm walls common color - verify from grid

DIRS = {
    "ACTION1": (0, -1),
    "ACTION2": (0, 1),
    "ACTION3": (-1, 0),
    "ACTION4": (1, 0),
}
INV = {"ACTION1": "ACTION2", "ACTION2": "ACTION1", "ACTION3": "ACTION4", "ACTION4": "ACTION3"}


def find_player(grid: np.ndarray) -> tuple[int, int]:
    # bright / distinct agent blob near start
    bg = Counter = __import__("collections").Counter
    c = Counter(grid.flatten())
    bgc = c.most_common(1)[0][0]
    # player often a distinct color near start
    for y in range(40, 50):
        for x in range(28, 40):
            if grid[y, x] != bgc and grid[y, x] != 0:
                return x, y
    return PLAYER0


def walkable(grid: np.ndarray, x: int, y: int, wall_colors: set[int]) -> bool:
    H, W = grid.shape
    if not (0 <= x < W and 0 <= y < H):
        return False
    return int(grid[y, x]) not in wall_colors


def bfs(grid: np.ndarray, start, goal, wall_colors):
    q = deque([(start, [])])
    seen = {start}
    while q:
        (x, y), path = q.popleft()
        if (x, y) == goal:
            return path
        for act, (dx, dy) in DIRS.items():
            nx, ny = x + dx, y + dy
            if (nx, ny) in seen:
                continue
            if not walkable(grid, nx, ny, wall_colors):
                continue
            seen.add((nx, ny))
            q.append(((nx, ny), path + [act]))
    return None


class QueueAgent:
    def __init__(self, mod, env, queue: list[str]):
        self.inner = mod.MyAgent(
            card_id="q", game_id="ls20", agent_name="q",
            ROOT_URL="x", record=False, arc_env=env, tags=[],
        )
        self.queue = deque(queue)
        self.inner.MAX_ACTIONS = len(queue) + 50

    def run(self):
        self.inner.timer = __import__("time").time()
        while (
            not self.inner.is_done(self.inner.frames, self.inner.frames[-1])
            and self.inner.action_counter <= self.inner.MAX_ACTIONS
        ):
            if self.queue:
                name = self.queue.popleft()
            else:
                name = "ACTION1"
            action = getattr(GameAction, name)
            action.reasoning = "bfs"
            frame = self.inner.take_action(action)
            if frame:
                self.inner.append_frame(frame)
                lv = frame.levels_completed
                print(
                    f"{self.inner.action_counter:3d} {name} L={lv} state={frame.state}"
                )
                if lv > 0 or frame.state == GameState.WIN:
                    print("SUCCESS")
                    return True
            self.inner.action_counter += 1
        return False


def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("ls20")
    spec = importlib.util.spec_from_file_location("ma", ROOT / "agent" / "my_agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # bootstrap one reset
    boot = mod.MyAgent(
        card_id="b", game_id="ls20", agent_name="b",
        ROOT_URL="x", record=False, arc_env=env, tags=[],
    )
    boot.MAX_ACTIONS = 0
    boot.main()

    g0 = grid_of(boot.frames[-1])
    # wall colors: very frequent dark border color
    from collections import Counter
    hist = Counter(g0.flatten())
    wall_colors = {14, 4}  # from level layout
    for col, _ in hist.most_common(3):
        wall_colors.add(int(col))

    px, py = find_player(g0)
    print(f"player~ ({px},{py}) goal={GOAL0} rot={ROT_PAD0}")

    path1 = bfs(g0, (px, py), ROT_PAD0, wall_colors)
    path2 = bfs(g0, ROT_PAD0, GOAL0, wall_colors)
    print(f"path to rot: {len(path1 or [])} steps, to goal: {len(path2 or [])} steps")

    queue = list(path1 or [])
    # wiggle on rotation pad (step on rhsxkxzdjz triggers rotate)
    queue += ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
    queue += list(path2 or [])

    # fresh env
    env2 = arc.make("ls20")
    ok = QueueAgent(mod, env2, queue).run()
    print("done ok=", ok)


if __name__ == "__main__":
    main()
