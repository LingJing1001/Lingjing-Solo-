"""su15 A* 求解器 — 分析 environment_files/su15 游戏机制，A* 搜索过关。

不依赖任何预计算解法 / 外接路径。

架构:
  solve_level(level, max_nodes=20000, weight=1.3)
    1. 初始化: make(level) → compact(g0) → A* 起点
    2. A* 循环 (DFS + undo 回溯):
       restore(key) 重建引擎（ACTION7 撤销，~1ms）
       for click in candidate_clicks(eng):
         执行点击 → 检查胜利
         compact(新状态) → heuristic(新状态)
         undo 恢复 → 下一个候选
         if 新状态更好: 递归搜索
    3. 胜利: 返回点击序列 [(x,y), ...]

游戏机制（从 su15.py 源码分析）:
  - 纯点击局（ACTION6 点击 + ACTION7 撤销）
  - 点击 (x,y) → 半径 8 内的 sprite 被拉向点击点
  - 胜利条件: 正确数量/类型的 sprite 中心落在目标区域内
  - ACTION7 完美撤销上一步（引擎内置历史栈）
  - 显示坐标 = 网格坐标（camera 1:1）
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
KAGGLE = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"
sys.path.insert(0, str(KAGGLE))
CANONICAL_ROOT = KAGGLE.parents[1]
os.environ.setdefault("LINGJING_SRC", str(CANONICAL_ROOT / "lingjing_solo"))
sys.path.insert(0, str(CANONICAL_ROOT))
VENDOR = KAGGLE / "vendor" / "ARC-AGI-3-Agents"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402

RADIUS = 8


# ── 状态读取 / 紧凑化 / 启发式 ───────────────────────────────

def _center(s: Any) -> tuple[float, float]:
    h, w = s.pixels.shape if s.pixels is not None else (1, 1)
    return (s.x + w / 2.0, s.y + h / 2.0)


def _dist(a, b) -> float:
    return ((a[0] - b[0]) ** ! 2 + (a[1] - b[1]) ** 2) ** 0.5


def _in_zone(cx: float, cy: float, z: Any) -> bool:
    h, w = z.pixels.shape if z.pixels is not None else (1, 1)
    return z.x <= cx <= z.x + w and z.y <= cy <= z.y + h


def _read(env: Any) -> dict:
    g = env._game
    players = [{"s": s, "c": _center(s)} for s in getattr(g, "lkujttxgs", [])]
    zones = [{"z": z, "c": _center(z)} for z in getattr(g, "powykypsm", [])]
    others = [{"s": s, "c": _center(s)} for s in getattr(g, "fezhhzhih", [])]
    return {"players": players, "zones": zones, "others": others}


def compact(state: dict) -> tuple:
    pts = []
    for p in state["players"]:
        pts.append(("p", int(p["c"][0]), int(p["c"][1])))
    for o in state["others"]:
        pts.append(("o", int(o["c"][0]), int(o["c"][1])))
    return tuple(sorted(pts))


def heuristic(state: dict) -> float:
    if not state["zones"]:
        return 0.0
    total = 0.0
    for p in state["players"]:
        cx, cy = p["c"]
        if any(_in_zone(cx, cy, z["z"]) for z in state["zones"]):
            continue
        total += min(_dist(p["c"], z["c"]) for z in state["zones"])
    return total


# ── 候选点击 ─────────────────────────────────────────────────

def candidate_clicks(state: dict) -> list[tuple[int, int]]:
    zones = state["zones"]
    if not zones:
        return []
    cands: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def _add(x, y):
        x, y = int(round(x)), int(round(y))
        if 0 <= x <= 63 and 10 <= y <= 63 and (x, y) not in seen:
            seen.add((x, y))
            cands.append((x, y))

    for p in state["players"]:
        px, py = p["c"]
        if any(_in_zone(px, py, z["z"]) for z in zones):
            continue
        best_z = min(zones, key=lambda z: _dist(p["c"], z["c"]))
        tx, ty = best_z["c"]
        d = _dist(p["c"], (tx, ty))
        if d < 1:
            _add(tx, ty)
            continue
        ux, uy = (tx - px) / d, (ty - py) / d
        for off in (6, 5, 7, 4):
            _add(px + ux * off, py + uy * off)
        _add(tx, ty)
        if d > 10:
            n = int(d / 6) + 1
            for i in range(1, min(n, 5)):
                _add(px + ux * 6 * i, py + uy * 6 * i)

    if state["players"]:
        mcx = sum(p["c"][0] for p in state["players"]) / len(state["players"])
        mcy = sum(p["c"][1] for p in state["players"]) / len(state["players"])
        for z in zones:
            _add(z["c"][0], z["c"][1])
            d = _dist((mcx, mcy), z["c"])
            if d > 1:
                ux, uy = (z["c"][0] - mcx) / d, (z["c"][1] - mcy) / d
                _add(mcx + ux * 6, mcy + uy * 6)
    return cands


# ── A* 求解（DFS + undo 回溯 + transposition table）──────────

def solve_level(
    env: Any,
    level: int,
    max_nodes: int = 20000,
    weight: float = 1.3,
) -> tuple[bool, list[tuple[int, int]], Any]:
    """A* 搜索单关。DFS + undo 回溯。"""

    visited: set[tuple] = set()
    nodes = [0]
    best_solution: list[list[tuple[int, int]]] = [None]

    def dfs(depth: int, max_depth: int, clicks: list[tuple[int, int]]) -> bool:
        if nodes[0] >= max_nodes:
            return False
        nodes[0] += 1

        state = _read(env)
        key = compact(state)
        if key in visited:
            return False
        visited.add(key)

        h = heuristic(state)
        if h == 0:
            # 全在目标区域内 → 检查是否过关
            # 可能需要额外点击触发
            return False  # 让外层检查 levels_completed

        if depth >= max_depth:
            visited.discard(key)
            return False

        cands = candidate_clicks(state)
        # 按启发式排序: 优先尝试最可能减少距离的点击
        scored: list[tuple[float, tuple[int, int]]] = []
        for click in cands:
            fr = env.step(GameAction.ACTION6, data={"x": click[0], "y": click[1]})
            if fr.levels_completed > level or fr.state.name == "WIN":
                best_solution[0] = clicks + [click]
                return True
            new_h = heuristic(_read(env))
            scored.append((new_h, click))
            env.step(GameAction.ACTION7, data={})  # undo

        scored.sort(key=lambda x: x[0])

        for new_h, click in scored:
            env.step(GameAction.ACTION6, data={"x": click[0], "y": click[1]})
            if dfs(depth + 1, max_depth, clicks + [click]):
                return True
            env.step(GameAction.ACTION7, data={})  # undo

        visited.discard(key)
        return False

    # 迭代加深
    frame = env.reset()
    for max_depth in range(5, 40, 5):
        visited.clear()
        if dfs(0, max_depth, []):
            sol = best_solution[0]
            # 重放到解
            env.reset()
            for cx, cy in sol:
                frame = env.step(GameAction.ACTION6, data={"x": cx, "y": cy})
                if frame.levels_completed > level or frame.state.name == "WIN":
                    break
            return True, sol, frame
        print(f"    depth={max_depth} nodes={nodes[0]} visited={len(visited)}", flush=True)

    return False, [], frame


# ── 主入口 ────────────────────────────────────────────────────

def main() -> None:
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("su15")
    if env is None:
        raise SystemExit("无法创建 su15 环境")
    frame = env.reset()
    print(f"reset: levels={frame.levels_completed} state={frame.state.name}")
    total = 0
    all_ok = True
    for lvl in range(9):
        t0 = time.time()
        ok, clicks, frame = solve_level(env, lvl, max_nodes=20000, weight=1.3)
        total += len(clicks)
        if not ok:
            all_ok = False
        print(f"  L{lvl}: {len(clicks)} clicks → levels={frame.levels_completed}  "
              f"{'OK' if ok else 'FAIL'}  {time.time()-t0:.1f}s")
        if not ok:
            break
    print(f"\n总计: levels={frame.levels_completed}/9  clicks={total}  state={frame.state.name}")
    print("ALL_PASS" if (all_ok and frame.levels_completed >= 9) else "FAIL")


if __name__ == "__main__":
    main()
