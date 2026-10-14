"""re86 机制级求解 + plan 生成 + 引擎回放验证。

机制（无障碍物关卡）：
- ACTION1=up(dy=-3) ACTION2=down(dy=+3) ACTION3=left(dx=-3) ACTION4=right(dx=+3)
- ACTION5=切换选中块（中心像素==0 表示选中）
- 块按 sprite 列表顺序轮换
- win: 所有块叠加到 64x64 画布，目标有意义像素(非-1非4)须匹配
"""
from __future__ import annotations
import sys, os, copy, itertools, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CANON = ROOT.parents[1]
sys.path.insert(0, str(CANON))
os.environ.setdefault("LINGJING_SRC", str(CANON / "lingjing_solo"))
VENDOR = ROOT / "vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

import arc_agi
from arcengine import ActionInput, GameAction, GameState
import numpy as np

arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.NORMAL)
env = arc.make("re86")
g = env._game
STEP = 3


def _is_win(g, level):
    st = str(g._state)
    return st == "GameState.WIN" or int(g._current_level_index) > level


def find_positions(g, lv):
    """对无障碍物关卡，找每块的目标位置。返回 list[(x,y)] 或 None。"""
    g._current_level_index = lv
    g.level_reset()
    blocks = g.current_level.get_sprites_by_tag("0031cppcuvqlbi")
    target = g.current_level.get_sprites_by_tag("0054xnsuqceejm")
    obstacles = g.current_level.get_sprites_by_tag("0003dlchiwseii")
    if obstacles:
        return None  # 有障碍物，本求解器不处理

    t = target[0]
    t_mask = (t.pixels != -1) & (t.pixels != 4)
    target_pixels = {}
    for r, c in np.argwhere(t_mask):
        v = int(t.pixels[r, c])
        target_pixels.setdefault(v, []).append((int(r), int(c)))

    block_info = []
    for b in blocks:
        px = b.pixels.copy()
        non_trans = px != -1
        by_val = {}
        for r, c in np.argwhere(non_trans):
            v = int(px[r, c])
            by_val.setdefault(v, []).append((int(r), int(c)))
        block_info.append({
            "x0": b.x, "y0": b.y, "w": b.width, "h": b.height,
            "by_val": by_val,
            "center": (b.height // 2, b.width // 2),
        })

    target_vals = list(target_pixels.keys())
    n_blocks = len(blocks)

    for assignment in itertools.product(range(n_blocks), repeat=len(target_vals)):
        block_targets = [{} for _ in range(n_blocks)]
        for i, tv in enumerate(target_vals):
            block_targets[assignment[i]][tv] = target_pixels[tv]
        block_positions = [None] * n_blocks
        feasible = True
        for bi in range(n_blocks):
            if not block_targets[bi]:
                block_positions[bi] = [(block_info[bi]["x0"], block_info[bi]["y0"])]
                continue
            vals = list(block_targets[bi].keys())
            first_val = vals[0]
            val_px = block_info[bi]["by_val"].get(first_val, [])
            # 排除选中标记 0
            val_px = [p for p in val_px if p != block_info[bi]["center"]]
            if not val_px:
                feasible = False
                break
            gy0, gx0 = block_targets[bi][first_val][0]
            cands = set()
            for r, c in val_px:
                cands.add((gx0 - c, gy0 - r))
            valid_cands = []
            for x, y in cands:
                ok = True
                for gy, gx in block_targets[bi][first_val][1:]:
                    if (gy - y, gx - x) not in val_px:
                        ok = False
                        break
                if not ok:
                    continue
                for v in vals[1:]:
                    vpx = [p for p in block_info[bi]["by_val"].get(v, [])
                           if p != block_info[bi]["center"]]
                    for gy, gx in block_targets[bi][v]:
                        if (gy - y, gx - x) not in vpx:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                dx = x - block_info[bi]["x0"]
                dy = y - block_info[bi]["y0"]
                if dx % STEP != 0 or dy % STEP != 0:
                    continue
                valid_cands.append((x, y))
            if not valid_cands:
                feasible = False
                break
            block_positions[bi] = valid_cands
        if not feasible:
            continue
        for combo in itertools.product(*block_positions):
            ok = True
            for bi in range(n_blocks):
                bx, by = combo[bi]
                for v, all_px in block_info[bi]["by_val"].items():
                    for r, c in all_px:
                        gy, gx = by + r, bx + c
                        for tv, tcoords in target_pixels.items():
                            if tv != v and (gy, gx) in tcoords:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                return list(combo), block_info
    return None


def make_plan(g, lv, target_positions, block_info):
    """生成动作序列：依次选中每块，移到目标位置。"""
    g._current_level_index = lv
    g.level_reset()
    blocks = g.current_level.get_sprites_by_tag("0031cppcuvqlbi")
    n = len(blocks)

    # 找初始选中块
    def selected_index():
        for i, b in enumerate(blocks):
            cy, cx = b.height // 2, b.width // 2
            if b.pixels[cy, cx] == 0:
                return i
        return -1

    plan = []
    cur_sel = selected_index()
    # 按块顺序依次移动
    for bi in range(n):
        tx, ty = target_positions[bi]
        x0, y0 = block_info[bi]["x0"], block_info[bi]["y0"]
        dx = tx - x0
        dy = ty - y0
        n_right = dx // STEP if dx > 0 else 0
        n_left = -dx // STEP if dx < 0 else 0
        n_down = dy // STEP if dy > 0 else 0
        n_up = -dy // STEP if dy < 0 else 0
        if n_right == 0 and n_left == 0 and n_down == 0 and n_up == 0:
            continue  # 不需移动
        # 切换到块 bi
        while cur_sel != bi:
            plan.append(5)
            # 模拟切换
            g.perform_action(ActionInput(id=GameAction.from_id(5), data={}), raw=True)
            cur_sel = selected_index()
            if _is_win(g, lv):
                return plan  # 提前过关
        # 移动
        for _ in range(n_up):
            plan.append(1)
            g.perform_action(ActionInput(id=GameAction.from_id(1), data={}), raw=True)
            if _is_win(g, lv):
                return plan
        for _ in range(n_down):
            plan.append(2)
            g.perform_action(ActionInput(id=GameAction.from_id(2), data={}), raw=True)
            if _is_win(g, lv):
                return plan
        for _ in range(n_left):
            plan.append(3)
            g.perform_action(ActionInput(id=GameAction.from_id(3), data={}), raw=True)
            if _is_win(g, lv):
                return plan
        for _ in range(n_right):
            plan.append(4)
            g.perform_action(ActionInput(id=GameAction.from_id(4), data={}), raw=True)
            if _is_win(g, lv):
                return plan
    return plan


def verify_plan(g, lv, plan):
    g._current_level_index = lv
    g.level_reset()
    for aid in plan:
        g.perform_action(ActionInput(id=GameAction.from_id(aid), data={}), raw=True)
        if _is_win(g, lv):
            return True, int(g._current_level_index), str(g._state)
    return _is_win(g, lv), int(g._current_level_index), str(g._state)


num_levels = len(g._levels)
print(f"re86 总关数={num_levels}")
plans = {}
for lv in range(num_levels):
    print(f"\n=== re86 L{lv} ===")
    res = find_positions(g, lv)
    if res is None:
        print(f"  L{lv}: 无纯位置解（有障碍物或无匹配）")
        continue
    target_positions, block_info = res
    for bi, (tx, ty) in enumerate(target_positions):
        dx = tx - block_info[bi]["x0"]
        dy = ty - block_info[bi]["y0"]
        print(f"  block {bi}: ({block_info[bi]['x0']},{block_info[bi]['y0']}) "
              f"-> ({tx},{ty}) dx={dx} dy={dy}")
    plan = make_plan(g, lv, target_positions, block_info)
    ok, li, st = verify_plan(g, lv, plan)
    tag = "✓" if ok else "✗"
    print(f"  plan len={len(plan)} 回放 li={li} st={st} {tag}")
    print(f"  plan={plan}")
    if ok:
        plans[lv] = plan
    else:
        print(f"  L{lv} 回放未过关！")

print(f"\n=== re86 plans 汇总 ===")
print(json.dumps({k: v for k, v in plans.items()}, indent=2))
