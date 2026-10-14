"""m0r0 专用求解器 — 镜像合并位置模型（读源码 + 看渲染帧确认）。

机制（来自 environment_files/m0r0/492f87ba/m0r0.py step()）：
- 关卡重置时 pyhtlpzlmnr=True（镜像模式默认开启）。
- ACTION1-4 仅在镜像模式下对 4 个 pikgci 做镜像联动：
      toljda-leklkn : (dx, dy)
      toljda-rivmdg : (-dx, dy)
      boweok-leklkn : (dx, -dy)
      boweok-rivmdg : (-dx, -dy)
  其中 ACTION1/2 → dy=-1/+1，ACTION3/4 → dx=-1/+1。
- 点击 sys_click 且 name=="mosdlc" → 退出镜像（单精灵移动模式）；
  点击其他 sys_click（pikgci）→ 保持镜像。
- 胜利：所有 pikgci 两两同坐标 → interaction=INTANGIBLE（合并），
  当未合并/未移除的 pikgci 数 == 0 时 next_level()。
- 复位陷阱：pikgci 落到 spswjz 触发动画，6 步后回到 sys_click 初始位。

求解策略：用真引擎 game 做 deepcopy BFS，状态 = 镜像标志 + 4 pikgci 位置/交互 +
合并集合 + 动画计数/动画中精灵；逐关求解后拼接成完整多关计划，
并返回规范格式 (action_name, x, y) 供 EngineSolver 逐帧回放。
输入 game 在调用前后保持不变（所有探索都在副本上进行）。
"""
from __future__ import annotations

import copy
import time
from collections import deque
from typing import Any, List, Optional, Tuple

try:
    from arcengine import ActionInput, GameAction, InteractionMode
except Exception:  # pragma: no cover - 依赖离线引擎
    ActionInput = None
    GameAction = None
    InteractionMode = None

# 4 个 pikgci 的固定顺序（用于稳定状态哈希）
_PIK = [
    "pikgci-toljda-leklkn",
    "pikgci-toljda-rivmdg",
    "pikgci-boweok-leklkn",
    "pikgci-boweok-rivmdg",
]

# 点击坐标用 display 空间（grid*scale+offset），引擎 step 再 display_to_grid 回来。
_CLICK_SCALE = 5
_CLICK_OFFSET = 5


def _grid_to_display(gx: int, gy: int) -> Tuple[int, int]:
    return (int(gx) * _CLICK_SCALE + _CLICK_OFFSET, int(gy) * _CLICK_OFFSET + _CLICK_OFFSET)


def _is_win(g: Any, level: int) -> bool:
    try:
        if int(getattr(g, "_current_level_index", 0)) > level:
            return True
    except Exception:
        pass
    return str(getattr(g, "_state", "")).upper() == "WIN"


def _pik_state(cur):
    """返回单个 pikgci 的 (x, y, done_flag)，done=已合并/移除。"""
    sp = cur.current_level.get_sprites_by_name(_PIK[0]) if False else None
    return None


def _state_key(g: Any) -> Tuple:
    key = []
    key.append(1 if getattr(g, "pyhtlpzlmnr", False) else 0)
    sel = getattr(g, "jnyxgyktgcv", None)
    key.append(sel.name if sel else "")
    for name in _PIK:
        sp = g.current_level.get_sprites_by_name(name)
        if sp:
            s = sp[0]
            done = False
            if InteractionMode is not None:
                done = s.interaction in (InteractionMode.INTANGIBLE, InteractionMode.REMOVED)
            else:
                done = str(getattr(s, "interaction", "")).upper() in ("INTANGIBLE", "REMOVED")
            key.append((int(s.x), int(s.y), 1 if done else 0))
        else:
            key.append(None)
    ok = getattr(g, "okpvcjupabr", set())
    key.append(tuple(sorted(ok)))
    key.append(int(getattr(g, "ukempikfmtm", -1)))
    key.append(tuple(sorted(s.name for s in getattr(g, "anfcrclwoac", []))))
    key.append(int(getattr(g, "_current_level_index", 0)))
    return tuple(key)


def _actions(g: Any) -> List[Any]:
    acts: List[Any] = [1, 2, 3, 4]
    for s in g.current_level.get_sprites():
        if getattr(s, "is_visible", True) and hasattr(s, "tags") and "sys_click" in (s.tags or []):
            dx, dy = _grid_to_display(s.x, s.y)
            acts.append(("click", dx, dy))
    return acts


def _apply(g: Any, a: Any) -> None:
    if ActionInput is None or GameAction is None:
        raise RuntimeError("arcengine 不可用，无法驱动 m0r0 求解器")
    if isinstance(a, int):
        g.perform_action(ActionInput(id=GameAction.from_id(a), data={}), raw=True)
    else:
        _, x, y = a
        g.perform_action(
            ActionInput(id=GameAction.ACTION6, data={"x": int(x), "y": int(y)}),
            raw=True,
        )


def _bfs_level(g: Any, level: int, t_limit: float, max_nodes: int) -> Optional[List[Any]]:
    if _is_win(g, level):
        return []
    t0 = time.time()
    start = copy.deepcopy(g)
    sk = _state_key(start)
    parent = {sk: (None, None)}
    queue = deque([(start, sk, 0)])
    nodes = 0
    while queue and nodes < max_nodes and (time.time() - t0) < t_limit:
        cur, ck, depth = queue.popleft()
        nodes += 1
        if depth >= 40:
            continue
        for a in _actions(cur):
            ng = copy.deepcopy(cur)
            try:
                _apply(ng, a)
            except Exception:
                continue
            if _is_win(ng, level):
                path = [a]
                k = ck
                while parent[k][0] is not None:
                    pk, pa = parent[k]
                    path.append(pa)
                    k = pk
                path.reverse()
                return path
            nk = _state_key(ng)
            if nk not in parent:
                parent[nk] = (ck, a)
                queue.append((ng, nk, depth + 1))
    return None


def _canon(a: Any) -> Tuple[str, int, int]:
    if isinstance(a, int):
        return (f"ACTION{a}", 0, 0)
    _, x, y = a
    return ("ACTION6", int(x), int(y))


def solve(game: Any, t_limit: float = 300.0, max_nodes: int = 2000000) -> Optional[List[Tuple[str, int, int]]]:
    """求解 m0r0，返回完整/部分多关计划 [(action_name, x, y), ...]。

    采用贪心前缀策略：逐关求解，解出就追加到计划，遇到第一关在预算内无解就
    停止并返回已解出的前缀。这样保证返回的计划恰好覆盖若干完整关卡、不会与
    引擎实际进程错位（余下关卡交回 CEAX 兜底），而不是"全有或全无"。

    输入 game 在调用前后保持不变；若首关即无解返回 None。
    首解较慢（受限于镜像迷宫的绕行搜索），结果会被 EngineSolver 缓存，后续即时。
    """
    if ActionInput is None or GameAction is None:
        return None
    n_levels = len(getattr(game, "_levels", []) or [])
    if n_levels == 0:
        return None
    full: List[Tuple[str, int, int]] = []
    t0 = time.time()
    per_nodes = max(50000, max_nodes // max(1, n_levels))
    for lvl in range(n_levels):
        per_tl = min(60.0, max(5.0, (t_limit - (time.time() - t0)) / max(1, n_levels)))
        g = copy.deepcopy(game)
        g._current_level_index = lvl
        g.level_reset()
        plan = _bfs_level(g, lvl, per_tl, per_nodes)
        if plan is None:
            # 该关在预算内无解：停止，返回已解出的前缀（安全，无错位）
            print(
                f"[m0r0_solver] L{lvl} 在预算内无解，返回已解前缀（{len(full)} 步，覆盖 "
                f"{lvl} 关）",
                flush=True,
            )
            break
        full.extend(_canon(a) for a in plan)
    return full if full else None


def _self_test() -> None:
    """离线自测：构造 m0r0 → solve → 在副本上回放验证全关通关。"""
    import os
    import sys
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(REPO, "integrations/ARC-AGI-3-Kaggle-Starter/vendor/ARC-AGI-3-Agents"))
    import arc_agi
    from arc_agi import Arcade, OperationMode

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=os.path.join(REPO, "environment_files"),
        recordings_dir=os.path.join(REPO, "state", "m0r0_solver_test"),
    )
    arc.create_scorecard()  # 触发环境扫描，否则 available_environments 为空
    t = [e for e in arc.available_environments if e.game_id.startswith("m0r0")][0]
    if not getattr(t, "class_name", None):
        t.class_name = "M0r0"
    env = arc._create_wrapper(t, arc.create_scorecard(), False, True, seed=0)
    env.reset()
    g = env._game

    plan = solve(g, t_limit=360.0)
    print(f"[m0r0_solver] plan len={len(plan) if plan else None}")
    if not plan:
        print("[m0r0_solver] 无解")
        return
    # 在副本上回放
    gg = copy.deepcopy(g)
    gg._current_level_index = 0
    gg.level_reset()
    for name, x, y in plan:
        if name == "ACTION6":
            gg.perform_action(ActionInput(id=GameAction.ACTION6, data={"x": x, "y": y}), raw=True)
        else:
            aid = getattr(GameAction, name)
            gg.perform_action(ActionInput(id=aid, data={}), raw=True)
    print(f"[m0r0_solver] replay -> level_index={gg._current_level_index} state={gg._state}")


if __name__ == "__main__":
    _self_test()
