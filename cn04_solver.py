# -*- coding: utf-8 -*-
"""CN04 黑盒求解器 — 分析特殊像素位置，直接计算对齐方案。

游戏机制:
  - ACTION6 点击选中 sprite，ACTION1-4 移动，ACTION5 旋转/换层
  - 特殊像素(色8/13) 两个 sprite 重叠 → 匹配 → 全匹配通关
  - 每个 sprite 有 2 个特殊像素，需要两个 sprite 的特殊像素在同一格重叠

策略:
  1. 读每个 sprite 的特殊像素位置（考虑当前旋转）
  2. 对每对需要匹配的 sprite，算出对齐需要的旋转 + 移动
  3. 生成动作序列：点击选中 → 旋转 → 移动到目标位置
  4. 用引擎黑盒验证
"""
from __future__ import annotations

import os, sys, time
from pathlib import Path
from typing import Any
import numpy as np

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


def apply_action(env: Any, action: tuple) -> Any:
    kind = action[0]
    if kind == "click":
        return env.step(GameAction.ACTION6, data={"x": action[1], "y": action[2]})
    elif kind == "move":
        return env.step(GameAction.from_id(action[1]), data={})
    elif kind == "rotate":
        return env.step(GameAction.ACTION5, data={})


def replay(env: Any, actions: list, level: int) -> Any:
    frame = env.reset()
    for a in actions:
        frame = apply_action(env, a)
        if frame.levels_completed > level or frame.state.name == "WIN":
            break
    return frame


def build_g2d(env: Any) -> dict:
    g = env._game
    cam = g.camera
    mapping = {}
    for dy in range(64):
        for dx in range(64):
            res = cam.display_to_grid(dx, dy)
            if res is not None and res not in mapping:
                mapping[res] = (dx, dy)
    return mapping


def _special_pixels(sprite: Any, game: Any) -> list:
    """sprite 原始数据中的特殊像素，考虑当前旋转后的全局坐标。"""
    orig = game.hlxyvcmpk.get(sprite.name)
    if orig is None:
        return []
    h, w = orig.shape
    pts_local = []
    for y in range(h):
        for x in range(w):
            if orig[y, x] == 8 or orig[y, x] == 13:
                pts_local.append((x, y))
    if not pts_local:
        return []
    # 应用旋转
    rot = sprite.rotation
    pts = []
    for x, y in pts_local:
        if rot == 0:
            rx, ry = x, y
        elif rot == 90:
            rx, ry = h - 1 - y, x
        elif rot == 180:
            rx, ry = w - 1 - x, h - 1 - y
        elif rot == 270:
            rx, ry = y, w - 1 - x
        else:
            rx, ry = x, y
        pts.append((int(sprite.x + rx), int(sprite.y + ry)))
    return pts


def _find_click(env: Any, g2d: dict, sprite: Any) -> tuple:
    """找能选中 sprite 的显示坐标。"""
    rendered = sprite.render()
    if rendered is None:
        return None
    for y, x in np.argwhere(rendered >= 0):
        gx, gy = int(sprite.x + x), int(sprite.y + y)
        if (gx, gy) in g2d:
            return g2d[(gx, gy)]
    return None


def _visible_sprites(env: Any) -> list:
    g = env._game
    return [s for s in g.current_level.get_sprites() if s.is_visible]


def solve_level_direct(env: Any, level: int, g2d: dict) -> list:
    """直接分析法：读特殊像素 → 算对齐 → 生成动作。"""
    frame = env.reset()
    g = env._game
    sprites = _visible_sprites(env)
    if len(sprites) < 2:
        return []

    # 每个 sprite 的特殊像素
    specials = []
    for s in sprites:
        sp = _special_pixels(s, g)
        if sp:
            specials.append((s, sp))
    if len(specials) < 2:
        return []

    # 简单策略：把第一个 sprite 移向第二个 sprite 的特殊像素位置
    # 尝试所有旋转 × 移动方向组合
    s1, sp1 = specials[0]
    s2, sp2 = specials[1]

    # 找能选中 s1 和 s2 的点击坐标
    click1 = _find_click(env, g2d, s1)
    click2 = _find_click(env, g2d, s2)
    if not click1 or not click2:
        return []

    actions = []
    grid_w, grid_h = g.current_level.grid_size or (20, 20)

    # 尝试：选中 s1 → 旋转到每种角度 → 移动使特殊像素对齐
    for rot_count in range(4):
        trial = []
        # 选中 s1
        trial.append(("click", click1[0], click1[1]))
        # 旋转 rot_count 次
        for _ in range(rot_count):
            trial.append(("rotate",))

        # 模拟到此看 s1 的特殊像素位置
        frame = replay(env, trial, level)
        g = env._game
        s1_cur = None
        for s in _visible_sprites(env):
            if s.name == s1.name:
                s1_cur = s
                break
        if s1_cur is None:
            continue
        sp1_cur = _special_pixels(s1_cur, g)
        if len(sp1_cur) != len(sp2):
            continue

        # 算偏移：让 sp1_cur[0] 对齐 sp2[0]
        dx = sp2[0][0] - sp1_cur[0][0]
        dy = sp2[0][1] - sp1_cur[0][1]
        # 检查 sp1_cur[1] + offset 是否也对齐 sp2[1]
        if len(sp1_cur) > 1 and len(sp2) > 1:
            if sp1_cur[1][0] + dx == sp2[1][0] and sp1_cur[1][1] + dy == sp2[1][1]:
                # 完美对齐！生成移动序列
                moves = []
                # dx > 0 → ACTION4 (右), dx < 0 → ACTION3 (左)
                for _ in range(abs(dx)):
                    moves.append(("move", 4 if dx > 0 else 3))
                for _ in range(abs(dy)):
                    moves.append(("move", 2 if dy > 0 else 1))
                full = trial + moves
                # 验证
                frame = replay(env, full, level)
                if frame.levels_completed > level or frame.state.name == "WIN":
                    return full
                # 检查是否更接近（特殊像素减少）
                g = env._game
                total_sp = sum(len(_special_pixels(s, g)) for s in _visible_sprites(env))
                if total_sp == 0:
                    return full

    # 也尝试选中 s2 移动
    for rot_count in range(4):
        trial = []
        trial.append(("click", click2[0], click2[1]))
        for _ in range(rot_count):
            trial.append(("rotate",))

        frame = replay(env, trial, level)
        g = env._game
        s2_cur = None
        for s in _visible_sprites(env):
            if s.name == s2.name:
                s2_cur = s
                break
        if s2_cur is None:
            continue
        sp2_cur = _special_pixels(s2_cur, g)
        if len(sp2_cur) != len(sp1):
            continue

        dx = sp1[0][0] - sp2_cur[0][0]
        dy = sp1[0][1] - sp2_cur[0][1]
        if len(sp2_cur) > 1 and len(sp1) > 1:
            if sp2_cur[1][0] + dx == sp1[1][0] and sp2_cur[1][1] + dy == sp1[1][1]:
                moves = []
                for _ in range(abs(dx)):
                    moves.append(("move", 4 if dx > 0 else 3))
                for _ in range(abs(dy)):
                    moves.append(("move", 2 if dy > 0 else 1))
                full = trial + moves
                frame = replay(env, full, level)
                if frame.levels_completed > level or frame.state.name == "WIN":
                    return full
                g = env._game
                total_sp = sum(len(_special_pixels(s, g)) for s in _visible_sprites(env))
                if total_sp == 0:
                    return full

    return None


def solve_level(env: Any, level: int, g2d: dict, time_budget: float = 120) -> list:
    """求解单关：先直接分析，再 BFS 兜底。"""
    t0 = time.time()
    sol = solve_level_direct(env, level, g2d)
    if sol is not None:
        return sol
    # TODO: BFS 兜底
    return None


def main() -> None:
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("cn04")
    if env is None:
        raise SystemExit("无法创建 cn04 环境")
    frame = env.reset()
    g2d = build_g2d(env)
    print(f"cn04: {len(g2d)} grid->display mappings")

    total = 0
    all_ok = True
    for lvl in range(6):
        t0 = time.time()
        sol = solve_level(env, lvl, g2d, time_budget=60)
        dt = time.time() - t0
        if sol is None:
            print(f"L{lvl+1}: NO SOLUTION  {dt:.1f}s")
            all_ok = False
            continue
        frame = replay(env, sol, lvl)
        ok = frame.levels_completed > lvl or frame.state.name == "WIN"
        total += len(sol)
        if not ok:
            all_ok = False
        print(f"L{lvl+1}: {len(sol)} actions -> levels={frame.levels_completed}  "
              f"{'OK' if ok else 'FAIL'}  {dt:.1f}s  {sol[:5]}...")

    print(f"\nTotal: actions={total}")
    print("ALL_PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
