"""通用孪生世界模型引擎（ARC-AGI-3，跨游戏共享）。

与 m0r0_solver（仅为单一游戏手写）不同，本模块是**游戏无关**的：

- 转移模型 = 直接在活引擎对象的 deepcopy 副本上调用 ``perform_action``，
  即 Twin 论文的 "twin world model"：一份可离线推演的环境副本。
- 状态去重 = ``twin_state_key``（棋盘快签名：当前关卡的精灵 name/x/y/interaction/visible
  + 非易变标量标志，O(精灵数) ~0.1ms）。早期版本用整份 `__dict__` 递归编码会触发
  `_action_count` 等易变字段污染 key、去重失效、BFS 爆炸；改棋盘签名后既快又正确。
- 动作空间 = 引擎报告的键盘动作（ACTION1-5）+ ACTION6 点击每个可点击精灵的
  display 坐标（坐标经 ``camera.grid_to_display`` 换算，回退 5x+5）。
- 胜利信号 = 当前关卡序号递增（每个环境通关都调 ``next_level()``，这是跨 25 个
  游戏验证过的通用信号），并兜底 ``_state`` 命中 win 类字符串 / ``won`` 等属性。
- 逐关 BFS，greedy-prefix：解出就追加，遇首关在预算内无解即停并返回已解前缀
  （保证计划恰好覆盖若干完整关卡、不与引擎实际进程错位）。

输入 game 在调用前后保持不变（所有探索都在副本上进行）。
返回 EngineSolver 规范格式 ``[(action_name, x, y), ...]``，便于逐帧回放。
"""
from __future__ import annotations

import copy
import hashlib
import time
from collections import deque
from typing import Any, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy 一般随引擎装好
    np = None  # type: ignore


# ---------------------------------------------------------------------------
# 易变字段表：twin_state_key 跳过这些 name，避免把"同一棋盘、不同步数"判成不同态
# （例如 _action_count 每步 +1、_seed/_rng、_camera、动画计数等都不应进签名）。
# ---------------------------------------------------------------------------
_DENY_EXACT = {
    "_action_count", "_action", "_action_complete", "_seed", "_rng", "_camera",
    "_next_level", "_env", "_scorecard", "_dirty", "_pending", "_frame", "_clock",
    "_timer", "_counter", "_iteration", "anfcrclwoac", "ukempikfmtm",
    "_clean_levels",
}
_DENY_SUB = ("action_count", "action_complete", "rng", "uuid",
             "framebuffer", "_anim", "counter", "iteration")


def _is_volatile(name: str) -> bool:
    if name in _DENY_EXACT:
        return True
    nl = name.lower()
    return any(s in nl for s in _DENY_SUB)


def _scalar_repr(v: Any, _depth: int = 0) -> Optional[str]:
    """仅当容器**全为标量叶子**时返回 repr，否则 None。

    这是把 dict/list/set 这类非标量状态（网格、清单、计数器……）安全纳入状态签名的
    关键：一旦遇到非标量叶子（Sprite/Level 等）就整体返回 None，既保证 identity-free
    （不含 "object at 0x" 内存地址），又天然有界（不会把整份对象图序列化进来）。
    """
    if v is None or isinstance(v, (bool, int, float, str)):
        return repr(v)
    if np is not None and isinstance(v, np.ndarray):
        # numpy 网格（很多游戏的棋盘真身就是 bool/int ndarray，如 lf52）。
        # 只编码小型数值网格，超限返回 None（有界 + identity-free）。
        try:
            if v.size <= 8192 and v.dtype.kind in "biuf":
                return "A" + repr(v.shape) + v.dtype.str + v.tobytes().hex()
        except Exception:
            return None
        return None
    if _depth >= 4:
        return None
    if isinstance(v, dict):
        try:
            items = sorted(v.items(), key=lambda kv: repr(kv[0]))
        except Exception:
            return None
        out = []
        for k, x in items:
            kr, xr = _scalar_repr(k, _depth + 1), _scalar_repr(x, _depth + 1)
            if kr is None or xr is None:
                return None
            out.append(f"{kr}:{xr}")
        return "{" + ",".join(out) + "}"
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            xr = _scalar_repr(x, _depth + 1)
            if xr is None:
                return None
            out.append(xr)
            if len(out) > 2000:  # 有界，防病态巨型容器
                break
        return "[" + ",".join(out) + "]"
    if isinstance(v, (set, frozenset)):
        rs = []
        for x in v:
            xr = _scalar_repr(x, _depth + 1)
            if xr is None:
                return None
            rs.append(xr)
            if len(rs) > 2000:
                break
        return "{" + ",".join(sorted(rs)) + "}"
    return None


def twin_state_key(game: Any) -> Optional[Tuple[str, int]]:
    """(md5 指纹, 关卡序号)。对「棋盘」做快签名，而非递归整份对象图。

    ARC-AGI-3 的转移是「棋盘 + 动作」的纯函数，故状态真身 = 当前关卡的精灵
    （name/x/y/interaction/visible）+ 非易变标量标志 + 非易变的**非标量容器**
    （网格/清单/计数器，很多游戏的棋盘真身就在这里，漏掉它会让不同态判同、BFS 乱剪枝）。
    整体 O(精灵数 + 标量容器大小) ~亚毫秒，既避开整图递归的 ~150ms/节点，又不会像
    早期 `_enc` 深度截断那样把精灵压成类名而丢失调位信息。
    """
    if game is None:
        return None
    lvl_idx = 0
    for nm in ("_current_level_index", "level_index"):
        try:
            lvl_idx = int(getattr(game, nm))
            break
        except Exception:
            continue
    scalar_parts: List[str] = []
    container_parts: List[str] = []
    for k, v in game.__dict__.items():
        if _is_volatile(k):
            continue
        if isinstance(v, (bool, int, float, str)):
            scalar_parts.append(f"{k}={v!r}")
        elif isinstance(v, (dict, list, tuple, set, frozenset)):
            r = _scalar_repr(v)
            if r is not None:
                container_parts.append(f"{k}={r}")
    cur = getattr(game, "current_level", None)
    if cur is not None:
        try:
            sps = []
            for s in cur.get_sprites():
                sps.append((
                    str(getattr(s, "name", "?")),
                    round(float(getattr(s, "x", 0)), 3),
                    round(float(getattr(s, "y", 0)), 3),
                    str(getattr(s, "interaction", "")),
                    str(getattr(s, "visible", "")),
                ))
            sps.sort()
            container_parts.append("SP=" + repr(sps))
        except Exception:
            pass
    scalar_parts.sort()
    container_parts.sort()
    payload = "|".join([f"L={lvl_idx}"] + scalar_parts + container_parts)
    return (hashlib.md5(payload.encode("utf-8", "replace")).hexdigest(), lvl_idx)

try:
    from arcengine import ActionInput, GameAction  # type: ignore
except Exception:  # pragma: no cover - 依赖离线引擎
    ActionInput = None
    GameAction = None

# 点击坐标默认变换（m0r0 脚本确认 scale=5 offset=5；多数游戏同此约定）。
_CLICK_SCALE = 5
_CLICK_OFFSET = 5


def _to_display(cam: Any, gx: int, gy: int) -> Tuple[int, int]:
    """grid 坐标 → display 坐标；优先用相机自身的 grid_to_display。"""
    if cam is not None:
        try:
            out = cam.grid_to_display(gx, gy)
            if out is not None:
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    return int(out[0]), int(out[1])
                if isinstance(out, dict) and "x" in out and "y" in out:
                    return int(out["x"]), int(out["y"])
        except Exception:
            pass
    return (int(gx) * _CLICK_SCALE + _CLICK_OFFSET,
            int(gy) * _CLICK_OFFSET + _CLICK_OFFSET)


def _win_signals(game: Any, level: int) -> bool:
    """通用胜利判定：关卡序号递增（主信号）或 win 类状态/属性。"""
    try:
        if int(getattr(game, "_current_level_index", 0)) > level:
            return True
    except Exception:
        pass
    st = str(getattr(game, "_state", "")).upper()
    if st in ("WIN", "WON", "COMPLETE", "COMPLETED", "SOLVED", "DONE"):
        return True
    for attr in ("won", "completed", "is_solved", "solved", "is_win"):
        v = getattr(game, attr, None)
        if v is True:
            return True
    return False


def _click_targets(game: Any) -> List[Tuple[int, int]]:
    """可点击精灵的 display 坐标列表（去重）。"""
    cam = getattr(game, "camera", None)
    out: List[Tuple[int, int]] = []
    seen = set()
    lvl = getattr(game, "current_level", None)
    if lvl is None:
        return out
    for s in lvl.get_sprites():
        tags = getattr(s, "tags", None) or []
        tag_str = " ".join(tags) if isinstance(tags, (list, tuple)) else str(tags)
        if "click" not in tag_str.lower():
            continue
        gx, gy = int(getattr(s, "x", 0)), int(getattr(s, "y", 0))
        dx, dy = _to_display(cam, gx, gy)
        if (dx, dy) in seen:
            continue
        seen.add((dx, dy))
        out.append((dx, dy))
    return out


def _default_move_actions() -> List[int]:
    """默认键盘动作 ACTION1..5（排除点击 ACTION6）。

    引擎报告 `available_actions` 时以它为准。

    注意：曾一度把默认集扩到 ACTION7（理由：GameAction 枚举里有 ACTION7，漏了
    可能搜不到某些关卡）。但这是**零证据的投机**：没有任何游戏实测需要它，反而
    把 m0r0 从 38 步打到 15 步——因为 BFS 每个动作都要 deepcopy 一份引擎副本
    （运动中状态 ~17.8ms），动作数 6→8 直接让单位时间探到的节点数掉 33%。
    故回退到 [1..5]。若将来真发现某游戏需要 ACTION7，应补针对性验证再加。
    """
    return [1, 2, 3, 4, 5]


def _enum_actions(game: Any) -> List[Any]:
    """动作枚举：键盘动作（int）+ 点击动作 (('click', dx, dy))。"""
    avail = getattr(game, "available_actions", None)
    if avail:
        move_ids = [a for a in avail
                    if isinstance(a, int) and a not in (0, 6)]
    else:
        move_ids = list(_default_move_actions())
    acts: List[Any] = list(move_ids)
    for dx, dy in _click_targets(game):
        acts.append(("click", dx, dy))
    return acts


def _apply(game: Any, a: Any) -> None:
    if ActionInput is None or GameAction is None:
        raise RuntimeError("arcengine 不可用，无法驱动通用孪生引擎")
    if isinstance(a, int):
        game.perform_action(
            ActionInput(id=GameAction.from_id(a), data={}), raw=True
        )
    else:
        _, x, y = a
        game.perform_action(
            ActionInput(id=GameAction.ACTION6, data={"x": int(x), "y": int(y)}),
            raw=True,
        )


def _state_key_of(game: Any) -> Any:
    return twin_state_key(game)


def _bfs_level(game: Any, level: int, t_limit: float, max_nodes: int
               ) -> Optional[List[Any]]:
    if _win_signals(game, level):
        return []
    t0 = time.time()
    start = copy.deepcopy(game)
    sk = _state_key_of(start)
    parent = {sk: (None, None)}
    queue = deque([(start, sk, 0)])
    nodes = 0
    acts = _enum_actions(start)
    while queue and nodes < max_nodes and (time.time() - t0) < t_limit:
        cur, ck, depth = queue.popleft()
        nodes += 1
        if depth >= 40:  # 深度上限（tu93 最长已解关 47 步/3 关，单关未超此限）
            continue
        for a in acts:
            ng = copy.deepcopy(cur)
            try:
                _apply(ng, a)
            except Exception:
                continue
            if _win_signals(ng, level):
                path = [a]
                k = ck
                while parent[k][0] is not None:
                    pk, pa = parent[k]
                    path.append(pa)
                    k = pk
                path.reverse()
                return path
            nk = _state_key_of(ng)
            if nk not in parent:
                parent[nk] = (ck, a)
                queue.append((ng, nk, depth + 1))
    return None


def _canon(a: Any) -> Tuple[str, int, int]:
    if isinstance(a, int):
        return (f"ACTION{a}", 0, 0)
    _, x, y = a
    return ("ACTION6", int(x), int(y))


def solve(game: Any, t_limit: float = 300.0, max_nodes: int = 2000000
          ) -> Optional[List[Tuple[str, int, int]]]:
    """通用孪生求解：返回完整/部分多关计划 [(action_name, x, y), ...]。

    采用贪心前缀策略：逐关求解，解出就追加到计划，遇到第一关在预算内无解就
    停止并返回已解出的前缀（安全、无错位）。输入 game 调用前后不变。
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
        remaining = t_limit - (time.time() - t0)
        if remaining <= 5.0:
            break
        # 剩余预算均分给剩余关卡（greedy-prefix 下 L0 先跑，天然吃满配额）。
        per_tl = max(10.0, min(300.0, remaining / max(1.0, (n_levels - lvl))))
        g = copy.deepcopy(game)
        g._current_level_index = lvl
        g.level_reset()
        if hasattr(g, "on_set_level"):
            try:
                g.on_set_level(g.current_level)
            except Exception:
                pass
        plan = _bfs_level(g, lvl, per_tl, per_nodes)
        if plan is None:
            print(
                f"[twin_engine] L{lvl} 在预算内无解，返回已解前缀"
                f"（{len(full)} 步，覆盖 {lvl} 关）",
                flush=True,
            )
            break
        full.extend(_canon(a) for a in plan)
    return full if full else None


def _self_test() -> None:
    """离线自测：构造 m0r0 → solve → 副本回放验证通关。"""
    import os
    import sys
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(
        0,
        os.path.join(
            REPO, "integrations/ARC-AGI-3-Kaggle-Starter/vendor/ARC-AGI-3-Agents"
        ),
    )
    import arc_agi
    from arc_agi import Arcade, OperationMode

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=os.path.join(REPO, "environment_files"),
        recordings_dir=os.path.join(REPO, "state", "twin_engine_test"),
    )
    arc.create_scorecard()
    t = [e for e in arc.available_environments if e.game_id.startswith("m0r0")][0]
    if not getattr(t, "class_name", None):
        t.class_name = "M0r0"
    env = arc._create_wrapper(t, arc.create_scorecard(), False, True, seed=0)
    env.reset()
    g = env._game

    plan = solve(g, t_limit=360.0)
    print(f"[twin_engine] m0r0 plan len={len(plan) if plan else None}")
    if not plan:
        print("[twin_engine] 无解")
        return
    gg = copy.deepcopy(g)
    gg._current_level_index = 0
    gg.level_reset()
    for name, x, y in plan:
        if name == "ACTION6":
            gg.perform_action(
                ActionInput(id=GameAction.ACTION6, data={"x": x, "y": y}), raw=True
            )
        else:
            gg.perform_action(
                ActionInput(id=getattr(GameAction, name), data={}), raw=True
            )
    print(
        f"[twin_engine] replay -> level_index={gg._current_level_index} "
        f"state={gg._state}"
    )


if __name__ == "__main__":
    _self_test()
