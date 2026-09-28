"""Probe sb26 arrange + tu93 keyboard mechanics for CEAX hardening."""
from __future__ import annotations

import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np
from arcengine import GameAction, GameState

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

import arc_agi
from arc_agi import OperationMode
from agents.agent import Agent
from lingjing_solo.core import SoloConfig, extract_grid
from lingjing_solo.perception import PerceptionEncoder

enc = PerceptionEncoder(SoloConfig())


def objs(grid):
    return enc.segment(grid) if grid is not None else []


def centroid(o):
    ys = [p[0] for p in o.pixels]
    xs = [p[1] for p in o.pixels]
    return int(round(sum(xs) / len(xs))), int(round(sum(ys) / len(ys)))


class Sb26ArrangeProbe(Agent):
    """Select bottom (y>=48) then place mid/top; submit periodically."""

    MAX_ACTIONS = 120

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.phase = 0  # 0 select, 1 place
        self.selected = None
        self.pairs = 0
        self.prev = None

    def is_done(self, frames, latest):
        if latest.state is GameState.WIN:
            return True
        return self.action_counter >= self.MAX_ACTIONS

    def choose_action(self, frames, latest):
        if latest.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.phase = 0
            self.selected = None
            self.pairs = 0
            self.prev = None
            a = GameAction.RESET
            a.reasoning = "reset"
            return a

        grid = extract_grid(latest)
        levels = int(latest.levels_completed or 0)
        delta = 0
        if self.prev is not None and grid is not None:
            delta = int((self.prev != grid).sum())
        if self.action_counter <= 2 or self.action_counter % 10 == 0 or levels:
            print(
                f"[sb26] step={self.action_counter} L={levels} phase={self.phase} "
                f"pairs={self.pairs} delta={delta} hist={Counter(grid.flatten()).most_common(5) if grid is not None else None}",
                flush=True,
            )

        # After a place attempt, try submit every 2 pairs or every 8 steps late
        if self.pairs >= 1 and (
            (self.pairs >= 2 and self.phase == 0)
            or (self.action_counter > 0 and self.action_counter % 18 == 0)
        ):
            if grid is not None:
                self.prev = grid.copy()
            a = GameAction.ACTION5
            a.reasoning = f"submit pairs={self.pairs}"
            print(f"[sb26] SUBMIT pairs={self.pairs}", flush=True)
            return a

        items = objs(grid)
        bottom = [o for o in items if centroid(o)[1] >= 48 and 4 <= len(o.pixels) <= 80]
        top = [o for o in items if centroid(o)[1] < 48 and 4 <= len(o.pixels) <= 120]
        bottom.sort(key=lambda o: len(o.pixels))
        top.sort(key=lambda o: len(o.pixels))

        if self.phase == 0:
            pool = bottom or [o for o in items if 4 <= len(o.pixels) <= 80]
            if not pool:
                a = GameAction.ACTION5
                a.reasoning = "submit_empty"
                return a
            o = pool[self.action_counter % len(pool)]
            x, y = centroid(o)
            self.selected = (x, y)
            self.phase = 1
            if grid is not None:
                self.prev = grid.copy()
            a = GameAction.ACTION6
            a.set_data({"x": x, "y": y})
            a.reasoning = f"select {x},{y} c={o.color}"
            return a

        # place
        pool = [o for o in top if centroid(o) != self.selected] or top or items
        o = pool[(self.action_counter // 2) % max(1, len(pool))]
        x, y = centroid(o)
        self.phase = 0
        self.pairs += 1
        if grid is not None:
            self.prev = grid.copy()
        a = GameAction.ACTION6
        a.set_data({"x": x, "y": y})
        a.reasoning = f"place {x},{y} c={o.color}"
        return a


class Tu93NavProbe(Agent):
    """BFS on rare-ish walkable colors toward sparse goals."""

    MAX_ACTIONS = 200

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.prev = None
        self.player = None
        self.path = []

    def is_done(self, frames, latest):
        if latest.state is GameState.WIN:
            return True
        return self.action_counter >= self.MAX_ACTIONS

    def choose_action(self, frames, latest):
        if latest.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.prev = None
            self.player = None
            self.path = []
            a = GameAction.RESET
            a.reasoning = "reset"
            return a

        grid = extract_grid(latest)
        levels = int(latest.levels_completed or 0)
        if grid is None:
            a = GameAction.ACTION1
            a.reasoning = "noop"
            return a

        if self.prev is not None:
            ys, xs = np.where(self.prev != grid)
            if len(xs):
                self.player = (int(xs.mean()), int(ys.mean()))

        hist = Counter(grid.flatten())
        bg = hist.most_common(1)[0][0]
        # walkable candidates: mid-frequency non-bg
        walk_colors = [c for c, n in hist.most_common() if c != bg and 30 < n < 800][:3]
        if not walk_colors:
            walk_colors = [c for c, n in hist.most_common() if c != bg][:2]

        H, W = grid.shape
        if self.player is None:
            # bootstrap: try ACTION1-4
            a = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4][
                self.action_counter % 4
            ]
            self.prev = grid.copy()
            a.reasoning = "bootstrap"
            return a

        # goals = small non-walk non-bg blobs far from player
        goals = []
        seen = set()
        for y in range(H):
            for x in range(W):
                if (x, y) in seen:
                    continue
                c = int(grid[y, x])
                if c == bg or c in walk_colors:
                    continue
                stack = [(x, y)]
                pts = []
                while stack:
                    cx, cy = stack.pop()
                    if (cx, cy) in seen or not (0 <= cx < W and 0 <= cy < H):
                        continue
                    if int(grid[cy, cx]) != c:
                        continue
                    seen.add((cx, cy))
                    pts.append((cx, cy))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        stack.append((cx + dx, cy + dy))
                if 2 <= len(pts) <= 40:
                    gx = sum(p[0] for p in pts) / len(pts)
                    gy = sum(p[1] for p in pts) / len(pts)
                    goals.append((gx, gy, c, len(pts)))

        px, py = self.player
        walk = set()
        for y in range(H):
            for x in range(W):
                if int(grid[y, x]) in walk_colors:
                    walk.add((x, y))
        # also allow near player
        walk.add((int(px), int(py)))

        goal = None
        if goals:
            goal = max(goals, key=lambda g: (g[0] - px) ** 2 + (g[1] - py) ** 2)

        action = GameAction.ACTION1
        if goal:
            gx, gy = int(goal[0]), int(goal[1])
            # BFS on walk union goal neighborhood
            q = deque([(int(px), int(py))])
            parent = {(int(px), int(py)): None}
            found = None
            while q:
                cx, cy = q.popleft()
                if abs(cx - gx) + abs(cy - gy) <= 2:
                    found = (cx, cy)
                    break
                for dx, dy, act in (
                    (0, -1, GameAction.ACTION1),
                    (0, 1, GameAction.ACTION2),
                    (-1, 0, GameAction.ACTION3),
                    (1, 0, GameAction.ACTION4),
                ):
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in parent:
                        continue
                    if (nx, ny) in walk or abs(nx - gx) + abs(ny - gy) <= 1:
                        parent[(nx, ny)] = ((cx, cy), act)
                        q.append((nx, ny))
            if found and parent[found] is not None:
                cur = found
                first = None
                while parent[cur] is not None:
                    prevc, act = parent[cur]
                    first = act
                    cur = prevc
                if first:
                    action = first

        if self.action_counter < 5 or self.action_counter % 25 == 0 or levels:
            print(
                f"[tu93] step={self.action_counter} L={levels} player={self.player} "
                f"walk={walk_colors} goals={len(goals)} act={action.name}",
                flush=True,
            )
        self.prev = grid.copy()
        action.reasoning = f"nav p={self.player}"
        return action


def run(game_id: str, cls):
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make(game_id)
    agent = cls(
        card_id="probe",
        game_id=game_id,
        agent_name=f"probe.{game_id}",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=env,
        tags=["probe"],
    )
    agent.main()
    final = agent.frames[-1]
    print(
        f"=== {game_id} L={final.levels_completed} state={final.state} "
        f"actions={agent.action_counter}"
    )
    return int(final.levels_completed or 0)


if __name__ == "__main__":
    g = sys.argv[1] if len(sys.argv) > 1 else "both"
    if g in ("sb26", "both"):
        run("sb26", Sb26ArrangeProbe)
    if g in ("tu93", "both"):
        run("tu93", Tu93NavProbe)
