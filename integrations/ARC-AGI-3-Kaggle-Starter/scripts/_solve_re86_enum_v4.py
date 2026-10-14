"""re86 L2+ enumeration v4: prune positions to only those covering target pixels."""
from __future__ import annotations
import sys, os, itertools, json
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


def is_win(g, level):
    st = str(g._state)
    return st == "GameState.WIN" or int(g._current_level_index) > level


def verify_plan(g, lv, plan):
    g._current_level_index = lv
    g.level_reset()
    for aid in plan:
        g.perform_action(ActionInput(id=GameAction.from_id(aid), data={}), raw=True)
        if is_win(g, lv):
            return True
    return is_win(g, lv)


def make_plan(g, lv, target_positions, block_info):
    g._current_level_index = lv
    g.level_reset()
    blocks = g.current_level.get_sprites_by_tag("0031cppcuvqlbi")

    def selected_index():
        for i, b in enumerate(blocks):
            cy, cx = b.height // 2, b.width // 2
            if b.pixels[cy, cx] == 0:
                return i
        return -1

    plan = []
    cur_sel = selected_index()
    for bi in range(len(blocks)):
        tx, ty = target_positions[bi]
        x0, y0 = block_info[bi]["x0"], block_info[bi]["y0"]
        dx = tx - x0
        dy = ty - y0
        n_right = dx // STEP if dx > 0 else 0
        n_left = -dx // STEP if dx < 0 else 0
        n_down = dy // STEP if dy > 0 else 0
        n_up = -dy // STEP if dy < 0 else 0
        if n_right == 0 and n_left == 0 and n_down == 0 and n_up == 0:
            continue
        while cur_sel != bi:
            plan.append(5)
            g.perform_action(ActionInput(id=GameAction.from_id(5), data={}), raw=True)
            cur_sel = selected_index()
            if is_win(g, lv):
                return plan
        for _ in range(n_up):
            plan.append(1)
            g.perform_action(ActionInput(id=GameAction.from_id(1), data={}), raw=True)
            if is_win(g, lv):
                return plan
        for _ in range(n_down):
            plan.append(2)
            g.perform_action(ActionInput(id=GameAction.from_id(2), data={}), raw=True)
            if is_win(g, lv):
                return plan
        for _ in range(n_left):
            plan.append(3)
            g.perform_action(ActionInput(id=GameAction.from_id(3), data={}), raw=True)
            if is_win(g, lv):
                return plan
        for _ in range(n_right):
            plan.append(4)
            g.perform_action(ActionInput(id=GameAction.from_id(4), data={}), raw=True)
            if is_win(g, lv):
                return plan
    return plan


def solve_enum_pruned(g, lv):
    g._current_level_index = lv
    g.level_reset()
    blocks = g.current_level.get_sprites_by_tag("0031cppcuvqlbi")
    target = g.current_level.get_sprites_by_tag("0054xnsuqceejm")
    obstacles = g.current_level.get_sprites_by_tag("0003dlchiwseii")
    if obstacles:
        print(f"  L{lv}: has obstacles, skip")
        return None

    cam_w, cam_h = g.camera.width, g.camera.height
    t = target[0]
    t_mask = (t.pixels != -1) & (t.pixels != 4)
    target_coords = [(int(r), int(c), int(t.pixels[r, c])) for r, c in np.argwhere(t_mask)]
    target_set = {(r, c) for r, c, _ in target_coords}
    n_blocks = len(blocks)
    print(f"  L{lv}: {n_blocks} blocks, {len(target_coords)} target pixels")

    block_info = []
    for b in blocks:
        px = b.pixels.copy()
        non_trans = px != -1
        pixels = {}
        for r, c in np.argwhere(non_trans):
            v = int(px[r, c])
            pixels[(int(r), int(c))] = v
        center = (b.height // 2, b.width // 2)
        xs = []
        x = b.x
        while x + b.width <= cam_w:
            xs.append(x)
            x += STEP
        x = b.x - STEP
        while x >= -b.width + STEP:
            xs.append(x)
            x -= STEP
        ys = []
        y = b.y
        while y + b.height <= cam_h:
            ys.append(y)
            y += STEP
        y = b.y - STEP
        while y >= -b.height + STEP:
            ys.append(y)
            y -= STEP
        all_positions = [(x, y) for x in sorted(set(xs)) for y in sorted(set(ys))]
        # Prune: only keep positions where at least one non-center pixel covers a target pixel
        pruned = []
        for bx, by in all_positions:
            covers = False
            for (r, c), v in pixels.items():
                if (r, c) == center:
                    continue
                gy, gx = by + r, bx + c
                if (gy, gx) in target_set:
                    covers = True
                    break
            if covers:
                pruned.append((bx, by))
        block_info.append({
            "name": b.name, "x0": b.x, "y0": b.y,
            "w": b.width, "h": b.height,
            "pixels": pixels, "center": center,
            "positions": pruned,
        })
        print(f"    block {len(block_info)-1}: {len(all_positions)} -> {len(pruned)} pruned positions")

    total = 1
    for bi in block_info:
        total *= len(bi["positions"])
    print(f"  pruned total combos: {total}")
    if total > 5000000:
        print(f"  still too many, skip")
        return None

    for exclude_mode in ["all", "last", "none"]:
        sel_range = range(n_blocks) if exclude_mode == "last" else [0]
        for last_sel in sel_range:
            for combo in itertools.product(*[bi["positions"] for bi in block_info]):
                canvas = np.full((cam_h, cam_w), -1, dtype=np.int64)
                for bi_idx, (bx, by) in enumerate(combo):
                    bi = block_info[bi_idx]
                    for (r, c), v in bi["pixels"].items():
                        if exclude_mode == "all" and (r, c) == bi["center"]:
                            continue
                        if exclude_mode == "last" and bi_idx == last_sel and (r, c) == bi["center"]:
                            continue
                        gy, gx = by + r, bx + c
                        if 0 <= gy < cam_h and 0 <= gx < cam_w:
                            canvas[gy, gx] = v
                ok = True
                for tr, tc, tv in target_coords:
                    if canvas[tr, tc] != tv:
                        ok = False
                        break
                if ok:
                    print(f"  MATCH (mode={exclude_mode}, last_sel={last_sel}): "
                          f"{[(bx, by) for bx, by in combo]}")
                    return combo, block_info
    print(f"  no match")
    return None


num_levels = len(g._levels)
plans = {}
for lv in range(num_levels):
    print(f"\n=== re86 L{lv} ===", flush=True)
    res = solve_enum_pruned(g, lv)
    if res is None:
        continue
    combo, block_info = res
    for perm in itertools.permutations(range(len(combo))):
        plan = make_plan(g, lv, [combo[p] for p in perm], block_info)
        ok = verify_plan(g, lv, plan)
        if ok:
            plans[lv] = plan
            print(f"  plan (perm {perm}) len={len(plan)} verify=OK", flush=True)
            print(f"  plan={plan}", flush=True)
            break
    else:
        print(f"  verify failed (all perms)", flush=True)

print(f"\n=== re86 plans summary ===", flush=True)
print(json.dumps({k: v for k, v in plans.items()}, indent=2), flush=True)
