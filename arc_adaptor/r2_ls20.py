"""LS20 的 R2 观察层：把「引擎内部状态」和「渲染帧」统一成同一份 observation。

为什么要单独做这一层（2026-09-19 实测，见 docs/最小可运行闭环 §3.1）：
    旧 R2 `run_ls20_r2r3r4.py: r2_perceive_ls20` 用「整幅画面里所有色 11 的格」当目标，
    实测 LS20 L1 的 84 个色 11 像素**全部**在第 61 行及以下的 HUD 步数条里，玩法区 0 个
    ⇒ 旧 R2 的 target_center 其实是步数条（y≈61），玩家永远追不上，A* 永远失败，
    决策 100% 落到 R4 贪心回退（同一次跑测 12 步全是 prune）。
    而 LS20 的真实机制是「玩家携带三元组 (shape,color,rot) 必须匹配目标格要求」
    （ls20.py:1882-1887：不匹配时踩目标格被当墙拒绝且不扣步数），
    这些条件在帧像素里根本不可见。

所以本模块提供两个后端，输出同一套公共字段，游戏专用信息一律进 `game_specific`：
    observe_state(game, grid, ...)  离线引擎后端：直接读引擎真值（Tier B 用）
    observe_frame(grid, ...)        渲染帧后端：线上/无引擎时用，HUD 已剔除

`state_key(game)` 是给 R3 搜索去重用的联合状态：(玩家格, 三元组, 已过关数)。
只按玩家坐标去重是不够的 —— 同一格不同三元组是两个世界。

内部混淆名来自本地探针实测（arc_adaptor/agents/strategies/probe_ls20_state.py /
probe_ls20_preview_rot.py）；引擎改版时只有 LS20_STATE_FIELDS 这一处需要同步。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Sequence

import numpy as np

from lingjing_solo.planning.ls20_solver import (
    PLAYFIELD_ROWS,
    _find_color_pad,
    _find_goal_markers,
    _find_player,
    _find_shape_pad,
    _rot_pad_cells,
    read_lives,
    read_step_bar,
)

OBSERVATION_SCHEMA = "ls20-r2-v1"

# 混淆名 → 语义。缺任何一个就说明引擎换了，必须显式失败而不是猜。
LS20_STATE_FIELDS: dict[str, str] = {
    "gudziatsk": "玩家精灵 (x,y 为左上角色格，移动步长 5)",
    "plrpelhym": "目标格精灵列表（踩上并要求三元组匹配才过关）",
    "srgbthxut": "目标预览精灵列表（在目标格 +(1,1)，画出要求的形状/颜色）",
    "fwckfzsyc": "当前三元组 shape 索引",
    "hiaauhahz": "当前三元组 color 索引",
    "cklxociuu": "当前三元组 rot 索引 (mod 4)",
    "ldxlnycps": "目标要求 shape 索引列表",
    "yjdexjsoa": "目标要求 color 索引列表",
    "ehwheiwsk": "目标要求 rot 索引列表",
    "_step_counter_ui": "步数条 (current_steps 剩余 / osgviligwp 上限 / efipnixsvl 每动作扣多少)",
    "_current_level_index": "当前关卡下标",
    "aqygnziho": "剩余生命（步数归零 -1，归 0 才 GAME_OVER，见 ls20.py:1983-1987）",
    "ebfuxzbvn": "本关重开的整屏覆盖帧标记（该帧画面检测器全部读空）",
}

REQUIRED_FIELDS = ("gudziatsk", "plrpelhym", "fwckfzsyc", "hiaauhahz", "cklxociuu")


class R2PerceptionError(RuntimeError):
    """R2 读不到必要事实时抛出；调用方不得拿半截观察去搜索。"""


# ─────────────────────────────────────────────────────────────
# 公共工具
# ─────────────────────────────────────────────────────────────

def to_grid(frame_or_array: Any) -> np.ndarray:
    """FrameData / list / ndarray → 64x64 int8 玩法网格。"""
    raw = getattr(frame_or_array, "frame", frame_or_array)
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], np.ndarray):
        raw = raw[0]
    grid = np.asarray(raw, dtype=np.int8)
    if grid.ndim == 1 and grid.size == PLAYFIELD_ROWS * PLAYFIELD_ROWS:
        side = int(np.sqrt(grid.size))
        grid = grid.reshape(side, side)
    elif grid.ndim == 3:
        grid = grid[0]
    if grid.ndim != 2:
        raise R2PerceptionError(f"无法把帧解析成 2D 网格，实际 shape={grid.shape}")
    return grid


def playfield(grid: np.ndarray) -> np.ndarray:
    """砍掉 HUD：第 PLAYFIELD_ROWS 行起是步数条/生命条，不是玩法区。"""
    return grid[:PLAYFIELD_ROWS, :]


def state_hash(grid: np.ndarray) -> str:
    """只哈希玩法区 —— 把 HUD 步数条卷进来会让同一局面被判成不同状态。"""
    payload = playfield(np.asarray(grid, dtype=np.int8)).tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def _xy(sprite: Any) -> list[int]:
    return [int(sprite.x), int(sprite.y)]


def _as_index_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    try:
        return [int(v) for v in value]
    except TypeError:
        raise R2PerceptionError(f"三元组分量无法解析: {value!r}")


def _base_observation(*, grid: np.ndarray, game_id: str, tick: int, state: str,
                      levels_completed: int, legal_actions: Sequence[str],
                      run_id: str, episode_id: str, visibility: str) -> dict[str, Any]:
    """§3.1 的公共字段；每个后端只往 game_specific 里塞自己的差异。"""
    return {
        "schema": OBSERVATION_SCHEMA,
        "observation_id": f"{run_id or 'local'}/{tick}",
        "game_id": game_id,
        "episode_id": episode_id,
        "tick": int(tick),
        "grid_shape": [int(grid.shape[0]), int(grid.shape[1])],
        "state_hash": state_hash(grid),
        "state": str(state),
        "levels_completed": int(levels_completed),
        "legal_actions": [str(a) for a in legal_actions],
        "objects_or_features": {},
        "visibility": visibility,
        "game_specific": {},
    }


def _goal_records(grid: np.ndarray, goal_cells: Iterable[Sequence[int]]) -> list[dict[str, Any]]:
    """每个目标格带上「踩进去时画面长什么样」——三元组不匹配时它是墙色。"""
    field = playfield(grid)
    out: list[dict[str, Any]] = []
    for x, y in goal_cells:
        x, y = int(x), int(y)
        if not (0 <= y < field.shape[0] and 0 <= x < field.shape[1]):
            out.append({"xy": [x, y], "out_of_bounds": True})
            continue
        patch = field[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        out.append({
            "xy": [x, y],
            "cell_color": int(field[y, x]),
            "neighborhood": patch.tolist(),
        })
    return out


# ─────────────────────────────────────────────────────────────
# 后端 A：渲染帧（线上/无引擎可用；能力受限，要明确标注）
# ─────────────────────────────────────────────────────────────

def observe_frame(grid: np.ndarray, *, game_id: str = "ls20", tick: int = 0,
                  state: str = "NOT_FINISHED", levels_completed: int = 0,
                  legal_actions: Sequence[str] = ("ACTION1", "ACTION2", "ACTION3", "ACTION4"),
                  run_id: str = "", episode_id: str = "") -> dict[str, Any]:
    """只靠画面能确定的事实。

    注意这里读不到「当前三元组」：玩家携带块的 shape/color/rot 在渲染里不独立可辨
    （probe_ls20_preview_rot.py 实测：rot 要求不同的关，预览精灵原始像素完全相同）。
    所以本后端的 visibility='partial'，R3 只能做位置寻路，不能规划修饰台。
    """
    grid = np.asarray(grid, dtype=np.int8)
    player = _find_player(grid)
    goals = _find_goal_markers(grid, player)
    pads = {
        "shape": _find_shape_pad(grid, player),
        "rot": _find_goal_rot_pad(grid, player),
        "color": _find_color_pad(grid, player),
    }
    obs = _base_observation(grid=grid, game_id=game_id, tick=tick, state=state,
                            levels_completed=levels_completed, legal_actions=legal_actions,
                            run_id=run_id, episode_id=episode_id, visibility="partial")
    steps = read_step_bar(playfield_source(grid))
    obs["objects_or_features"] = {
        "player": list(player) if player else None,
        "goal_cells": [[int(x), int(y)] for x, y in goals],
        "modifier_pads": {k: list(v) if v else None for k, v in pads.items()},
    }
    obs["game_specific"] = {
        "backend": "frame",
        "step_bar": list(steps) if steps else None,
        "lives": read_lives(grid),
        "refills_in_playfield": _playfield_refill_cells(grid),
        "triple_current": None,
        "triple_required": None,
        "limitations": ["帧后端读不到当前/要求三元组 ⇒ 不能规划旋转台/调色台"],
    }
    return obs


def _find_goal_rot_pad(grid: np.ndarray, start: Any):
    """旋转台：复用 solver 的格集合，取离玩家最近的一个。"""
    cells = _rot_pad_cells(playfield(grid))
    if not cells:
        return None
    if not start:
        return list(cells[0])
    return list(min(cells, key=lambda c: abs(c[0] - start[0]) + abs(c[1] - start[1])))


def playfield_source(grid: np.ndarray) -> np.ndarray:
    """read_step_bar/read_lives 需要含 HUD 的整幅图，这里只做类型收口。"""
    return np.asarray(grid, dtype=np.int8)


def _playfield_refill_cells(grid: np.ndarray) -> list[list[int]]:
    """玩法区里的色 11 = 步数补给（npxgalaybz），不是过关目标。"""
    field = playfield(grid)
    ys, xs = np.nonzero(field == 11)
    return [[int(x), int(y)] for x, y in zip(xs, ys)]


# ─────────────────────────────────────────────────────────────
# 后端 B：引擎内部状态（离线引擎 Tier B；R3 规划要用它）
# ─────────────────────────────────────────────────────────────

def triple_current(game: Any) -> dict[str, int]:
    missing = [name for name in ("fwckfzsyc", "hiaauhahz", "cklxociuu")
               if not hasattr(game, name)]
    if missing:
        raise R2PerceptionError(f"引擎缺少三元组字段 {missing}，LS20_STATE_FIELDS 需同步")
    return {
        "shape": int(game.fwckfzsyc),
        "color": int(game.hiaauhahz),
        "rot": int(game.cklxociuu) % 4,
    }


def triple_required(game: Any) -> dict[str, list[int]]:
    return {
        "shape": _as_index_list(getattr(game, "ldxlnycps", None)),
        "color": _as_index_list(getattr(game, "yjdexjsoa", None)),
        "rot": [r % 4 for r in _as_index_list(getattr(game, "ehwheiwsk", None))],
    }


def state_key(game: Any) -> tuple:
    """R3 搜索的去重键：位置 × 三元组 × 关卡。

    只用玩家坐标去重会漏掉「同一格、不同携带块」这两条完全不同的分支；
    步数不进键，因为它只影响预算、不影响可达局面（进了新状态自然更省步）。
    """
    for name in REQUIRED_FIELDS:
        if not hasattr(game, name):
            raise R2PerceptionError(f"引擎缺少 {name}，无法构造 state_key（见 LS20_STATE_FIELDS）")
    player = game.gudziatsk
    triple = triple_current(game)
    return (
        int(player.x), int(player.y),
        triple["shape"], triple["color"], triple["rot"],
        int(getattr(game, "_current_level_index", 0)),
    )


def observe_state(game: Any, grid: np.ndarray, *, game_id: str = "ls20", tick: int = 0,
                  legal_actions: Sequence[str] = ("ACTION1", "ACTION2", "ACTION3", "ACTION4"),
                  run_id: str = "", episode_id: str = "") -> dict[str, Any]:
    """引擎真值后端：过关判定交给 state/levels_completed，绝不用像素差冒充成功。"""
    for name in REQUIRED_FIELDS:
        if not hasattr(game, name):
            raise R2PerceptionError(
                f"引擎缺少 {LS20_STATE_FIELDS.get(name, name)} 所需字段 {name}")
    grid = np.asarray(grid, dtype=np.int8)
    goal_cells = [_xy(s) for s in game.plrpelhym]
    player = _xy(game.gudziatsk)
    counter = getattr(game, "_step_counter_ui", None)
    obs = _base_observation(
        grid=grid, game_id=game_id, tick=tick,
        state=getattr(game, "_state", "UNKNOWN").name if hasattr(getattr(game, "_state", None), "name")
        else str(getattr(game, "_state", "UNKNOWN")),
        levels_completed=int(getattr(game, "_current_level_index", 0)),
        legal_actions=legal_actions, run_id=run_id, episode_id=episode_id,
        visibility="full",
    )
    obs["objects_or_features"] = {
        "player": player,
        "goal_cells": goal_cells,
        "goals": _goal_records(grid, goal_cells),
        "previews": [_xy(s) for s in getattr(game, "srgbthxut", []) or []],
    }
    obs["game_specific"] = {
        "backend": "engine_state",
        "triple_current": triple_current(game),
        "triple_required": triple_required(game),
        "steps_left": int(counter.current_steps) if counter is not None else None,
        "step_budget": int(getattr(counter, "osgviligwp", 0)) if counter is not None else None,
        # 预算三件套：每动作扣多少 / 还剩几条命 / 这一帧是不是「归零重开」的闪光帧。
        # 后两个是 R3 剪枝和 R2 可信度判定要的：归零不是 GameOver，而是本关重开 +
        # 全屏覆盖帧（ls20.py:1983-2005），该帧所有按 5x5 块工作的检测器都会读空。
        "step_decrement": int(getattr(counter, "efipnixsvl", 1)) if counter is not None else None,
        "lives": (int(game.aqygnziho) if hasattr(game, "aqygnziho") else None),
        "level_restart_flash": bool(getattr(game, "ebfuxzbvn", False)),
        "level_index": int(getattr(game, "_current_level_index", 0)),
        "state_key": list(state_key(game)),
        "moved_step": 5,
    }
    # 帧后端能给的东西这里也顺手给，便于两个后端的输出直接 diff
    obs["game_specific"]["frame_check"] = {
        "player_seen": _find_player(grid) is not None,
        "goals_seen": len(_find_goal_markers(grid, tuple(player))),
    }
    return obs


def is_goal_ready(obs: dict[str, Any]) -> bool:
    """当前携带三元组是否已被目标接受 —— R3 位置 A* 的短路判据。

    triple_required 的三个列表是「全局可接受值集合」（引擎把 ldxlnycps/yjdexjsoa/
    ehwheiwsk 存成集合，不是按目标编号一一对应），所以这里直接做一次合取判断。
    """
    spec = obs.get("game_specific", {})
    cur, req = spec.get("triple_current"), spec.get("triple_required")
    if not cur or not req or not req.get("shape"):
        return False
    return (cur["shape"] in req["shape"] and cur["color"] in req["color"]
            and cur["rot"] in req["rot"])


def moves_left(obs: dict[str, Any]) -> int | None:
    """把步数条换算成「还做得了几个动作」——R3 预算剪枝用的量。

    引擎每动作扣 step_decrement（撞墙/被拒的目标格不扣，见 ls20.py:1510-1512），
    所以这是**上界**：宁可高估剩余，也不要因为少算一步就把可达分支剪掉。
    拿不到步数条（帧后端、闪光帧）时返回 None，让调用方区分「无预算信息」和「没预算」。
    """
    spec = obs.get("game_specific", {})
    steps, dec = spec.get("steps_left"), spec.get("step_decrement")
    if steps is None or not dec or dec <= 0:
        return None
    return int(steps) // int(dec)


def transition(before: dict[str, Any], action: str, after: dict[str, Any]) -> dict[str, Any]:
    """§3.1 要求的 (state_before, action, state_after) 因果记录。"""
    return {
        "before_hash": before["state_hash"],
        "action": {"name": action},
        "after_hash": after["state_hash"],
        "before_key": before["game_specific"].get("state_key"),
        "after_key": after["game_specific"].get("state_key"),
        "levels_delta": after["levels_completed"] - before["levels_completed"],
        "moved": before["state_hash"] != after["state_hash"],
    }


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=list)
