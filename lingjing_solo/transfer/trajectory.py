"""统一轨迹 schema + 引擎内省效果分类（迁移学习管线的数据底座）。

设计动机（docs/热图提议器优化方案.md 的三条教训）：
  1. 屏幕像素 delta 漏掉纯状态变化 → 标签毒化。这里每步的效果标签由
     「屏幕网格 delta + 引擎隐藏状态 + 关卡推进」三方联合判定，引擎侧
     隐藏状态变化但屏幕不动 → ``state_only``，不再是隐形的 no_op。
  2. 数据太薄 → schema 里 ``is_probe`` 支持反事实探针增广：在已访问状态
     上快照/恢复，试别的动作拿引擎真值标签。
  3. 各来源格式不一 → routes JSON / my_agent 计划常量 / state 解文件全部
     编译成同一种 StepRecord，按 ``source`` 加可靠性权重。

约定：
  * 网格统一为 ``grid[y][x]``（行=第一下标），颜色为 ARC 色号 int。
  * ``screen_delta`` 里 ``old``/``new`` 均为色号；空列表 = 屏幕无变化。
  * 本模块不 import 任何引擎包（与 planning 层同一纯洁性边界）。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = 1

# 主效果类（互斥，按优先级取第一个命中）。
OUTCOME_WIN = "win"
OUTCOME_ADVANCE = "advance"
OUTCOME_GAME_OVER = "game_over"
OUTCOME_SCREEN = "screen"
OUTCOME_STATE_ONLY = "state_only"
OUTCOME_NO_OP = "no_op"
OUTCOMES = (
    OUTCOME_NO_OP, OUTCOME_STATE_ONLY, OUTCOME_SCREEN,
    OUTCOME_ADVANCE, OUTCOME_WIN, OUTCOME_GAME_OVER,
)
# 排序/打分用的“有效”效果类（部署指标：把有效动作排在 no_op 前面）。
EFFECTIVE_OUTCOMES = frozenset({OUTCOME_SCREEN, OUTCOME_STATE_ONLY, OUTCOME_ADVANCE, OUTCOME_WIN})

# 屏幕变化的几何细分（次要标签，供机制规则挖掘用）。
KIND_NONE = "none"
KIND_RECOLOR = "recolor"
KIND_MOVE = "move"
KIND_SPAWN = "spawn"
KIND_DESPAWN = "despawn"
KIND_MIXED = "mixed"
EFFECT_KINDS = (KIND_NONE, KIND_RECOLOR, KIND_MOVE, KIND_SPAWN, KIND_DESPAWN, KIND_MIXED)


def grid_of(frame: Any) -> np.ndarray:
    """FrameData.frame → 2D 色号网格（兼容 (1,h,w) 与 (h,w)）。"""
    g = np.asarray(frame, dtype=np.int64)
    if g.ndim == 3:
        g = g[0]
    return g


def grid_digest(grid: Any) -> str:
    import hashlib

    g = np.ascontiguousarray(np.asarray(grid, dtype=np.int64))
    return hashlib.md5(g.tobytes()).hexdigest()[:16]


def screen_delta(
    grid_before: np.ndarray, grid_after: np.ndarray, cap: int = 256
) -> List[Dict[str, int]]:
    """逐格差异 [{x, y, old, new}]，按 (y, x) 排序，超过 cap 截断（尾部省略）。"""
    a = np.asarray(grid_before, dtype=np.int64)
    b = np.asarray(grid_after, dtype=np.int64)
    if a.shape != b.shape:
        return []
    ys, xs = np.nonzero(a != b)
    out: List[Dict[str, int]] = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        if len(out) >= cap:
            break
        out.append({"x": int(x), "y": int(y), "old": int(a[y, x]), "new": int(b[y, x])})
    return out


def background_color(grid: np.ndarray) -> int:
    """众数色号当背景（ARC 惯例：背景占屏主导）。"""
    g = np.asarray(grid)
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def classify_outcome(
    *,
    levels_before: int,
    levels_after: int,
    state_after: str,
    delta_count: int,
    hidden_changed: bool,
    sig_changed: bool,
) -> str:
    """主效果类：advance/win > 屏幕变化 > 纯状态变化 > no_op。"""
    if state_after == "WIN":
        return OUTCOME_WIN
    if levels_after > levels_before:
        return OUTCOME_ADVANCE
    if state_after in ("GAME_OVER", "LOSE"):
        return OUTCOME_GAME_OVER
    if delta_count > 0:
        return OUTCOME_SCREEN
    if hidden_changed or sig_changed:
        return OUTCOME_STATE_ONLY
    return OUTCOME_NO_OP


def classify_effect_kind(
    grid_before: np.ndarray, grid_after: np.ndarray, bg: Optional[int] = None
) -> str:
    """屏幕变化的几何细分。

    recolor  = 原地换色（同格 old/new 都非背景）
    spawn    = 只出现新前景格
    despawn  = 只消失前景格
    move     = 前景格等量增减且逐色面积守恒（位移的保守判据）
    mixed    = 其余
    """
    a = np.asarray(grid_before, dtype=np.int64)
    b = np.asarray(grid_after, dtype=np.int64)
    if a.shape != b.shape or not np.any(a != b):
        return KIND_NONE
    if bg is None:
        bg = background_color(a)
    diff = a != b
    swapped = int(np.sum(diff & (a != bg) & (b != bg)))
    added = int(np.sum(diff & (a == bg) & (b != bg)))
    removed = int(np.sum(diff & (a != bg) & (b == bg)))
    if swapped and not added and not removed:
        return KIND_RECOLOR
    if added and not removed and not swapped:
        return KIND_SPAWN
    if removed and not added and not swapped:
        return KIND_DESPAWN
    if added and removed and not swapped:
        area_kept = True
        for c in np.unique(np.concatenate([a[diff], b[diff]])):
            if int(c) == int(bg):
                continue
            if int(np.sum(a == c)) != int(np.sum(b == c)):
                area_kept = False
                break
        if area_kept:
            return KIND_MOVE
    return KIND_MIXED


@dataclass
class StepRecord:
    """一步 (状态, 动作) → 效果 的引擎真值记录（JSON 一行）。"""

    game_id: str                 # 完整 id，如 "vc33-5430563c"
    gid: str                     # 前缀，如 "vc33"（LOGO 分组键）
    level_idx: int
    step_idx: int                # 本次回放内的全局步序
    action: Dict[str, Any]       # {"id": 6, "x": 45, "y": 33}，x/y 可为 None
    source: str                  # route / agent-plan / state-solution / probe
    reliability: float           # 来源可靠性权重（探针 0.5，真路线 1.0）
    outcome: str                 # OUTCOMES 之一
    effect_kind: str = KIND_NONE
    is_probe: bool = False
    base_sig: str = ""           # 探针所属的基态签名（同组候选可做排序评测）
    grid_hash_before: str = ""
    grid_hash_after: str = ""
    delta_count: int = 0
    screen_delta: List[Dict[str, int]] = field(default_factory=list)
    hidden_before: str = ""      # 引擎隐藏状态摘要（_get_hidden_state）
    hidden_after: str = ""
    state_sig_before: str = ""   # generic_shadow 状态签名摘要
    state_sig_after: str = ""
    levels_before: int = 0
    levels_after: int = 0
    # 采录时算好的可迁移上下文特征（不存整张网格，控制单条体积）。
    context: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StepRecord":
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, line: str) -> "StepRecord":
        return cls.from_dict(json.loads(line))


def read_jsonl(path: str | Path) -> List[StepRecord]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(StepRecord.from_json(line))
    return records


def write_jsonl(records: Iterable[StepRecord], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(r.to_json() + "\n")
            n += 1
    return n


def outcome_distribution(records: Sequence[StepRecord]) -> Dict[str, int]:
    dist = {o: 0 for o in OUTCOMES}
    for r in records:
        dist[r.outcome] = dist.get(r.outcome, 0) + 1
    return dist


# ---------------------------------------------------------------- 探针上下文

def connected_components(
    grid: np.ndarray, bg: Optional[int] = None, max_components: int = 64
) -> List[Dict[str, Any]]:
    """逐色 4-连通域（轻量 BFS，带质心/外接框），供探针选点与对象特征用。

    比 perception.encoder 的完整分割便宜一个量级，且不依赖 SoloConfig；
    同色粘连在一起时算一个域——对“点这个对象会发生什么”已够用。
    """
    g = np.asarray(grid)
    h, w = g.shape
    if bg is None:
        bg = background_color(g)
    comps: List[Dict[str, Any]] = []
    visited = np.zeros_like(g, dtype=bool)
    for color in sorted(int(c) for c in np.unique(g) if int(c) != int(bg)):
        ys, xs = np.nonzero((g == color) & ~visited)
        for y0, x0 in zip(ys.tolist(), xs.tolist()):
            if visited[y0, x0]:
                continue
            stack = [(y0, x0)]
            visited[y0, x0] = True
            cells: List[Tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < h and 0 <= nx < w
                        and not visited[ny, nx] and g[ny, nx] == color
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            cys = [c[0] for c in cells]
            cxs = [c[1] for c in cells]
            comps.append({
                "color": color,
                "area": len(cells),
                "cy": sum(cys) // len(cys),
                "cx": sum(cxs) // len(cxs),
                "y0": min(cys), "y1": max(cys),
                "x0": min(cxs), "x1": max(cxs),
                "cells": cells,
            })
            if len(comps) >= max_components:
                return comps
    return comps


def click_context(
    grid: np.ndarray, x: int, y: int, comps: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """点击动作的可迁移上下文（无绝对坐标，供 affordance 特征化）。

    neighborhood 取点击点 3×3 的色号九宫格（越界补 -1）；
    对象统计取点击色所属连通域：面积/外接框/该色域数/该色总格数。
    """
    g = np.asarray(grid)
    h, w = g.shape
    if comps is None:
        comps = connected_components(g)
    color = int(g[y, x])
    bg = background_color(g)
    mine = [c for c in comps if c["color"] == color]
    target = None
    best = 10**9
    for c in mine:
        if c["y0"] <= y <= c["y1"] and c["x0"] <= x <= c["x1"]:
            d = abs(c["cy"] - y) + abs(c["cx"] - x)
            if d < best:
                best, target = d, c
    hood = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy, xx = y + dy, x + dx
            hood.append(int(g[yy, xx]) if (0 <= yy < h and 0 <= xx < w) else -1)
    n_objects = len(comps)
    ctx: Dict[str, Any] = {
        "kind": "click",
        "click_color": color,
        "bg": bg,
        "hood3x3": hood,
        "n_objects": n_objects,
        "n_colors": int(len(np.unique(g))),
        "grid_h": h,
        "grid_w": w,
        "rel_x": round(x / max(1, w - 1), 3),
        "rel_y": round(y / max(1, h - 1), 3),
        "on_bg": color == bg,
    }
    if target is not None:
        ctx["obj"] = {
            "color": color,
            "area": int(target["area"]),
            "bw": int(target["x1"] - target["x0"] + 1),
            "bh": int(target["y1"] - target["y0"] + 1),
            "n_comps_of_color": len(mine),
            "color_cells_total": int(np.sum(g == color)),
        }
    else:
        ctx["obj"] = None
    return ctx


def keyboard_context(grid: np.ndarray) -> Dict[str, Any]:
    """键盘动作的盘面级上下文（无点击点，特征只剩盘面统计）。"""
    g = np.asarray(grid)
    comps = connected_components(g, max_components=128)
    bg = background_color(g)
    return {
        "kind": "keyboard",
        "bg": bg,
        "n_objects": len(comps),
        "n_colors": int(len(np.unique(g))),
        "grid_h": int(g.shape[0]),
        "grid_w": int(g.shape[1]),
        "fg_cells": int(np.sum(g != bg)),
    }
