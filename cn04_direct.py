# -*- coding: utf-8 -*-
"""CN04 专用求解器 — 分析特殊像素对齐，生成动作序列。

游戏机制:
  - 每个 sprite 有特殊像素 (色8/13)
  - 需要两个 sprite 的特殊像素在同一格重叠 → 匹配
  - 所有特殊像素匹配 → 通关

策略:
  1. 读每个 sprite 的特殊像素（考虑当前旋转）
  2. 对每对 sprite，尝试所有旋转组合，找对齐方案
  3. 生成动作序列：点击选中 → 旋转 → 移动 → 点击另一个
"""
import sys, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KAGGLE = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"
sys.path.insert(0, str(KAGGLE))
os.environ.setdefault("LINGJING_SRC", str(ROOT / "lingjing_solo"))
sys.path.insert(0, str(ROOT))
VENDOR = KAGGLE / "vendor" / "ARC-AGI-3-Agents"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, ActionInput, GameState
import numpy as np


def make_env(gid="cn04"):
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    return arc.make(gid)


def apply_action(env, action):
    kind = action[0]
    if kind == "click":
        return env.step(GameAction.ACTION6, data={"x": action[1], "y": action[2]})
    elif kind == "move":
        return env.step(GameAction.from_id(action[1]), data={})
    elif kind == "rotate":
        return env.step(GameAction.ACTION5, data={})


def replay(env, actions, level):
    frame = env.reset()
    for a in actions:
        frame = apply_action(env, a)
        if frame.levels_completed > level or frame.state.name == "WIN":
            break
    return frame


def _special_pixels_local(game, sprite):
    """sprite 原始数据中的特殊像素（本地坐标）。"""
    orig = game.hlxyvcmpk.get(sprite.name)
    if orig is None:
        return []
    pts = []
    for y in range(orig.shape[0]):
        for x in range(orig.shape[1]):
            if orig[y, x] in (8, 13):
                pts.append((x, y))
    return pts


def _rotate_point(x, y, rot, h, w):
    """旋转一个点。"""
    if rot == 0:
        return x, y
    elif rot == 90:
        return h - 1 - y, x
    elif rot == 180:
        return w - 1 - x, h - 1 - y
    elif rot == 270:
        return y, w - 1 - x
    return x, y


def _special_pixels_global(game, sprite):
    """sprite 的特殊像素全局坐标（考虑当前旋转）。"""
    orig = game.hlxyvcmpk.get(sprite.name)
    if orig is None:
        return []
    h, w = orig.shape
    rot = sprite.rotation
    pts = []
    for y in range(h):
        for x in range(w):
            if orig[y, x] in (8, 13):
                rx, ry = _rotate_point(x, y, rot, h, w)
                pts.append((int(sprite.x + rx), int(sprite.y + ry)))
    return pts


def _special_pixels_global_at_rot(game, sprite, target_rot):
    """sprite 在目标旋转下的特殊像素全局坐标。"""
    orig = game.hlxyvcmpk.get(sprite.name)
    if orig is None:
        return []
    h, w = orig.shape
    pts = []
    for y in range(h):
        for x in range(w):
            if orig[y, x] in (8, 13):
                rx, ry = _rotate_point(x, y, target_rot, h, w)
                pts.append((int(sprite.x + rx), int(sprite.y + ry)))
    return pts


def _find_click_pos(game, sprite):
    """找能选中 sprite 的显示坐标（用非透明像素）。"""
    rendered = sprite.render()
    if rendered is None:
        return None
    # 找非透明像素
    import numpy as np
    non_tp = np.argwhere(rendered >= 0)
    if len(non_tp) == 0:
        return None
    # 取第一个非透明像素
    y, x = non_tp[0]
    gx, gy = int(sprite.x + x), int(sprite.y + y)
    # 预计算 grid->display 映射
    g2d = {}
    for dy in range(64):
        for dx in range(64):
            res = game.camera.display_to_grid(dx, dy)
            if res is not None and res not in g2d:
                g2d[res] = (dx, dy)
    if (gx, gy) in g2d:
        return g2d[(gx, gy)]
    return gx, gy


def _try_align(game, s1, s2):
    """尝试对齐 s1 和 s2 的特殊像素。

    返回: (rot_count, dx, dy) 或 None
      rot_count: s1 需要旋转的次数 (0-3)
      dx, dy: s1 需要移动的格子数
    """
    sp1_local = _special_pixels_local(game, s1)
    sp2_local = _special_pixels_local(game, s2)
    if not sp1_local or not sp2_local:
        return None

    h1, w1 = s1.pixels.shape if s1.pixels is not None else (1, 1)
    h2, w2 = s2.pixels.shape if s2.pixels is not None else (1, 1)

    # 尝试 s1 的每种旋转
    for rot_count in range(4):
        target_rot = (s1.rotation + rot_count * 90) % 360
        sp1_rot = []
        for x, y in sp1_local:
            rx, ry = _rotate_point(x, y, target_rot, h1, w1)
            sp1_rot.append((rx, ry))

        # 尝试 s1 的每个特殊像素对齐 s2 的每个特殊像素
        for i, (rx1, ry1) in enumerate(sp1_rot):
            for j, (lx2, ly2) in enumerate(sp2_local):
                # s2 的局部坐标需要应用旋转
                rx2, ry2 = _rotate_point(lx2, ly2, s2.rotation, h2, w2)
                # 计算偏移
                dx = s2.x + rx2 - s1.x - rx1
                dy = s2.y + ry2 - s1.y - ry1

                # 检查其他特殊像素是否也对齐（按配对）
                all_aligned = True
                for k in range(len(sp1_rot)):
                    # s1 的第 k 个特殊像素 + 偏移
                    gx1 = s1.x + sp1_rot[k][0] + dx
                    gy1 = s1.y + sp1_rot[k][1] + dy
                    # 配对：s1[i] 对齐 s2[j]，s1[k] 对齐 s2[k]（如果 k == j，则 s1[j] 对齐 s2[i]）
                    if k == i:
                        # s1[i] 应该对齐 s2[j]
                        rx2_k, ry2_k = _rotate_point(sp2_local[j][0], sp2_local[j][1], s2.rotation, h2, w2)
                    elif k == j:
                        # s1[j] 应该对齐 s2[i]
                        rx2_k, ry2_k = _rotate_point(sp2_local[i][0], sp2_local[i][1], s2.rotation, h2, w2)
                    else:
                        # s1[k] 应该对齐 s2[k]
                        rx2_k, ry2_k = _rotate_point(sp2_local[k][0], sp2_local[k][1], s2.rotation, h2, w2)

                    if gx1 != s2.x + rx2_k or gy1 != s2.y + ry2_k:
                        all_aligned = False
                        break

                if all_aligned:
                    print(f"  [align] rot_count={rot_count} (s1.rot={s1.rotation}->{target_rot}), dx={dx}, dy={dy}, i={i}, j={j}", flush=True)
                    # Debug: print the alignment
                    for k in range(len(sp1_rot)):
                        gx1 = s1.x + sp1_rot[k][0] + dx
                        gy1 = s1.y + sp1_rot[k][1] + dy
                        if k == i:
                            rx2_k, ry2_k = _rotate_point(sp2_local[j][0], sp2_local[j][1], s2.rotation, h2, w2)
                        elif k == j:
                            rx2_k, ry2_k = _rotate_point(sp2_local[i][0], sp2_local[i][1], s2.rotation, h2, w2)
                        else:
                            rx2_k, ry2_k = _rotate_point(sp2_local[k][0], sp2_local[k][1], s2.rotation, h2, w2)
                        print(f"    s1[{k}] + offset = ({gx1},{gy1}), s2配对 = ({s2.x+rx2_k},{s2.y+ry2_k})", flush=True)
                    return rot_count, dx, dy

    return None


def solve_level_direct(game, t_limit):
    """直接分析法：找特殊像素对齐方案。"""
    try:
        sprites = [s for s in game.current_level.get_sprites() if s.is_visible]
        if len(sprites) < 2:
            return None

        # 找能对齐的 sprite 对
        for i in range(len(sprites)):
            for j in range(len(sprites)):
                if i == j:
                    continue
                s1, s2 = sprites[i], sprites[j]

                # 尝试 s1 对齐 s2
                result = _try_align(game, s1, s2)
                if result:
                    rot_count, dx, dy = result
                    click2 = _find_click_pos(game, s2)
                    if not click2:
                        continue

                    actions = []
                    # 如果 s1 已经选中，不需要点击
                    if game.xseexqzst != s1:
                        click1 = _find_click_pos(game, s1)
                        if not click1:
                            continue
                        actions.append(("click", click1[0], click1[1]))
                    for _ in range(rot_count):
                        actions.append(("rotate",))
                    # 移动
                    for _ in range(abs(dx)):
                        actions.append(("move", 4 if dx > 0 else 3))
                    for _ in range(abs(dy)):
                        actions.append(("move", 2 if dy > 0 else 1))
                    actions.append(("click", click2[0], click2[1]))

                    print(f"[cn04] direct: {len(actions)} steps (s1->s2, rot={rot_count}, move=({dx},{dy}))", flush=True)
                    return actions

                # 尝试 s2 对齐 s1
                result = _try_align(game, s2, s1)
                if result:
                    rot_count, dx, dy = result
                    click1 = _find_click_pos(game, s1)
                    click2 = _find_click_pos(game, s2)
                    if not click1 or not click2:
                        continue

                    actions = []
                    actions.append(("click", click2[0], click2[1]))
                    for _ in range(rot_count):
                        actions.append(("rotate",))
                    for _ in range(abs(dx)):
                        actions.append(("move", 4 if dx > 0 else 3))
                    for _ in range(abs(dy)):
                        actions.append(("move", 2 if dy > 0 else 1))
                    actions.append(("click", click1[0], click1[1]))

                    print(f"[cn04] direct: {len(actions)} steps (s2->s1, rot={rot_count}, move=({dx},{dy}))", flush=True)
                    return actions

        print(f"[cn04] direct: no alignment found", flush=True)
        return None

    except Exception as e:
        print(f"[cn04] direct error: {e}", flush=True)
        return None


def solve_level(env, level, time_budget=60):
    """求解单关。"""
    t0 = time.time()
    frame = env.reset()
    if frame.levels_completed > level or frame.state.name == "WIN":
        return []

    g = env._game
    sol = solve_level_direct(g, time_budget)
    if sol:
        # 验证
        env.reset()
        for a in sol:
            frame = apply_action(env, a)
            if frame.levels_completed > level or frame.state.name == "WIN":
                return sol
        # 验证失败，尝试 BFS
        print(f"[cn04] direct solution failed, trying BFS", flush=True)

    # BFS 兜底
    from lingjing_solo.planning.search.generic_shadow import r3_generic_search
    from arcengine import GameAction, ActionInput, GameState

    act_map = {n: getattr(GameAction, f"ACTION{n}") for n in (1, 2, 3, 4, 5, 6)}

    # 预计算 grid->display 映射
    g2d = {}
    for dy in range(64):
        for dx in range(64):
            res = g.camera.display_to_grid(dx, dy)
            if res is not None and res not in g2d:
                g2d[res] = (dx, dy)

    def make_action(aid):
        if aid == 6:
            # 找可点击的 sprite
            sprites = [s for s in g.current_level.get_sprites() if s.is_visible]
            if sprites:
                s = sprites[0]
                h, w = s.pixels.shape if s.pixels is not None else (1, 1)
                gx, gy = int(s.x + w // 2), int(s.y + h // 2)
                if (gx, gy) in g2d:
                    dx, dy = g2d[(gx, gy)]
                else:
                    dx, dy = gx, gy
                return ActionInput(id=aid, data={"x": dx, "y": dy}, reasoning=None)
            return ActionInput(id=aid, data={"x": 160, "y": 120}, reasoning=None)
        return ActionInput(id=aid, data={}, reasoning=None)

    path = r3_generic_search(
        env,
        t_limit=min(time_budget, 30),
        max_nodes=20000,
        max_depth=30,
        act_map=act_map,
        make_action=make_action,
        game_over_state=GameState.GAME_OVER,
        win_state=GameState.WIN,
    )
    return path


def main():
    env = make_env("cn04")
    if env is None:
        raise SystemExit("无法创建 cn04 环境")

    total = 0
    all_ok = True
    for lvl in range(6):
        t0 = time.time()
        sol = solve_level(env, lvl, time_budget=60)
        dt = time.time() - t0
        if sol is None:
            print(f"L{lvl+1}: NO SOLUTION  {dt:.1f}s")
            all_ok = False
            break
        frame = replay(env, sol, lvl)
        ok = frame.levels_completed > lvl or frame.state.name == "WIN"
        total += len(sol)
        if not ok:
            all_ok = False
        print(f"L{lvl+1}: {len(sol)} actions -> levels={frame.levels_completed}  "
              f"{'OK' if ok else 'FAIL'}  {dt:.1f}s")

    print(f"\nTotal: actions={total}")
    print("ALL_PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
