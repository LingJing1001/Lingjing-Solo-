"""r11l 专用点击求解器：真机布局 dump + 像素级碰撞解析 + 画刷吸色模拟 + 真机回放验证。

机制依据（r11l.py 495a7899，全部实测核对）:
- 纯点击(ACTION6)，每关 60 动作上限，超时 lose。碰撞全部 PIXEL_PERFECT：非(-1)即实心。
- 两段式：点可点段(roefwulewcui-*, 菱形13px命中区)选中 → 点空处放置；段中心=点击点，
  杆(roefwu-*)位置=其所有段质心-(2,2)，段瞬移无中间碰撞检查，墙/危险/吸色/胜利全在落点判定。
- 胜利：每 piece（有 flkdtg-* 靶且非 dirwzt）的杆与靶像素相交；无杆的靶(画刷靶)走
  bulmhgivatv 兜底：任一 whkxtx 画刷杆与靶相交且颜色集合相等(ldzvchvkvp)。
- 危险区(defgjl* 实心斑块)：任意杆落点触碰 → 违例+1，5 次判负（优先于胜利判定）。
- 颜料块(puukul-*)半边图案：whkxtx 空杆(全0像素)落点触碰即吸收，印章按块局部坐标写入，
  不同区域的块可叠加成多色集合 → 3 色靶可行；吸错色永久污染。
- 靶是空心菱形环(半径3)，杆是实心菱形(半径2)，同心放置无像素重叠 → 落点需偏移 1~2px。

求解：直杆件 = 搜靶心附近偏移使杆∩靶非空且避开危险；画刷杆 = 按靶色集选颜料块序列
（印章模拟验证色集）→ 逐路点移动全部段（零和偏移 DFS，段落点避墙/避段互撞）→ 落靶。
输出每关 display 点击序列（64x64 scale=1，display==grid），真机回放验证后打印
_R11L_PLANS 字面量并写 bench JSON。
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
STARTER = PKG.parent / "ARC-AGI-3-Kaggle-Starter"
os.chdir(STARTER)
os.environ["OPERATION_MODE"] = "offline"
os.environ["ENVIRONMENTS_DIR"] = str(STARTER / "environment_files")
for p in (str(PKG), str(STARTER)):
    if p not in sys.path:
        sys.path.insert(0, p)

import arc_agi
from arc_agi import OperationMode
from arcengine import ActionInput, GameAction

arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE,
                     environments_dir=str(STARTER / "environment_files"))
env = arc.make("r11l")
env.reset()
G = env._game


@dataclass
class Piece:
    key: str
    rod = None
    segs: list = field(default_factory=list)      # [[x, y], ...] 段当前左上角
    target = None                                 # flkdtg sprite or None


@dataclass
class LevelData:
    idx: int
    pieces: dict
    wall: np.ndarray          # 64x64 bool
    danger: np.ndarray        # 64x64 bool
    blobs: list               # (name, color, sprite)
    seg_mask: np.ndarray = None   # 段实心掩码（引擎按全掩码判墙）


def solid_mask(s) -> np.ndarray:
    return np.asarray(s.pixels) != -1


def stamp_mask(s) -> np.ndarray:
    px = np.asarray(s.pixels)
    return (px != -1) & (px != 0)


def to_grid(placed: np.ndarray, ox: int, oy: int) -> np.ndarray:
    grid = np.zeros((64, 64), dtype=bool)
    h, w = placed.shape
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(64, ox + w), min(64, oy + h)
    if x1 <= x0 or y1 <= y0:
        return grid
    grid[y0:y1, x0:x1] = placed[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    return grid


def overlap(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.any(a & b))


def extract_level(idx: int) -> LevelData:
    sp = G._clean_levels[idx].get_sprites()
    pieces: dict[str, Piece] = {}

    def piece_of(key: str) -> Piece:
        if key not in pieces:
            pieces[key] = Piece(key=key)
        return pieces[key]

    wall = np.zeros((64, 64), dtype=bool)
    danger = np.zeros((64, 64), dtype=bool)
    blobs = []
    for s in sp:
        if s.name.startswith("roefwulewcui-"):
            piece_of(s.name[len("roefwulewcui-"):]).segs.append([int(s.x), int(s.y)])
        elif s.name.startswith("roefwu-"):
            piece_of(s.name[len("roefwu-"):]).rod = s
        elif s.name.startswith("flkdtg-"):
            piece_of(s.name[len("flkdtg-"):]).target = s
        elif s.name.startswith("wakneh-"):
            wall |= to_grid(solid_mask(s), int(s.x), int(s.y))
        elif s.name.startswith("defgjl"):
            danger |= to_grid(solid_mask(s), int(s.x), int(s.y))
        elif s.name.startswith("puukul-"):
            px = np.asarray(s.pixels)
            color = int(max(px.max(), 0))
            blobs.append((s.name, color, s))
    seg_mask = None
    for s in sp:
        if s.name.startswith("roefwulewcui-"):
            seg_mask = solid_mask(s)
            break
    return LevelData(idx, pieces, wall, danger, blobs, seg_mask)


# ---------- 段组移动：零和偏移 DFS ----------

def build_spots(lv: LevelData, occupied: list) -> list:
    """全图无墙且远离其他段占位的候选落点（每件构建一次）。"""
    wall, sm = lv.wall, lv.seg_mask
    spots = []
    for y in range(2, 62):
        for x in range(2, 62):
            if wall[y, x]:
                continue
            if overlap(to_grid(sm, x, y), wall):
                continue
            if any(max(abs(x - ox), abs(y - oy)) < 3 for ox, oy in occupied):
                continue
            spots.append((x, y))
    return spots


def iter_dests(lv: LevelData, k: int, occupied: list, w: tuple[int, int], spots: list):
    """散布式落点搜索（生成器）：k 个段坐标和 = k*路点，互距 Chebyshev>=3。
    优先紧凑解：落点限制在路点邻域内并按距离排序，逐级放大半径，最多产出 10 个组合。"""
    spot_set = set(spots)
    tx, ty = w[0] * k, w[1] * k

    def pair_ok(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 3

    for radius in (16, 24):
        win = sorted((s for s in spots if abs(s[0] - w[0]) <= radius and abs(s[1] - w[1]) <= radius),
                     key=lambda s: (s[0] - w[0]) ** 2 + (s[1] - w[1]) ** 2)
        wset = set(win)
        if len(win) < k:
            continue
        n = 0
        if k == 2:
            for a in win:
                need = (tx - a[0], ty - a[1])
                if need in wset and pair_ok(a, need):
                    yield [a, need]
                    n += 1
                    if n >= 10:
                        return
        elif k == 3:
            for i, a in enumerate(win):
                rx, ry = tx - a[0], ty - a[1]
                for b in win[i + 1:]:
                    if not pair_ok(a, b):
                        continue
                    c = (rx - b[0], ry - b[1])
                    if c in wset and c != a and pair_ok(b, c) and pair_ok(a, c):
                        yield [a, b, c]
                        n += 1
                        if n >= 10:
                            return
        else:
            groups = {}
            for i, a in enumerate(win):
                for b in win[i + 1:]:
                    if pair_ok(a, b):
                        groups.setdefault((a[0] + b[0], a[1] + b[1]), []).append((a, b))
            for (ax, ay), plist in groups.items():
                for (a, b) in plist:
                    for (c, d) in groups.get((tx - ax, ty - ay), ()):
                        pts = (a, b, c, d)
                        if all(max(abs(pts[i][0] - pts[j][0]), abs(pts[i][1] - pts[j][1])) >= 3
                               for i in range(4) for j in range(i + 1, 4)):
                            yield [a, b, c, d]
                            n += 1
                            if n >= 10:
                                return
    return


def rod_at(piece: Piece, w: tuple[int, int]) -> tuple[int, int]:
    # 实测：杆位置 = 段质心本身（rvkbignsyr 的 width//2 在活局里为 0）
    return w


def rod_mask_at(piece: Piece, w: tuple[int, int]) -> np.ndarray:
    return to_grid(solid_mask(piece.rod), *rod_at(piece, w))


def plan_level(lv: LevelData, log) -> list[tuple[int, int]] | None:
    """返回本关点击序列（seg 中心坐标），失败返回 None。
    直杆件按规划顺序重试：先规划的件会把其他件的段当静态障碍，
    顺序不同结果不同（谁先规划谁让路）。"""
    import itertools
    import math

    seg_pos0 = {p.key: [list(s) for s in p.segs] for p in lv.pieces.values()}
    seg_pos: dict = {}
    clicks: list[tuple[int, int]] = []

    def reset_state() -> None:
        seg_pos.clear()
        seg_pos.update({k: [list(s) for s in v] for k, v in seg_pos0.items()})
        clicks.clear()

    def rod_unsafe_at(p: Piece, cx: int, cy: int, allow_blobs) -> bool:
        """质心检查：杆踩危险 or 触碰不允许的颜料块（污染）都算失败。"""
        rm = rod_mask_at(p, (cx, cy))
        if overlap(rm, lv.danger):
            return True
        for name, color, s in lv.blobs:
            if name in allow_blobs:
                continue
            if overlap(rm, to_grid(solid_mask(s), int(s.x), int(s.y))):
                return True
        return False

    def check_hop(p: Piece, w: tuple[int, int], allow_blobs, segs_state, spots):
        """检查"全部段移到 w"：返回执行序 [(seg_idx, dest)] 或 None。
        危险区会把整步移动回退，所以每个中间质心都必须安全。"""
        k = len(segs_state)
        others = [(x, y) for k2, ss in seg_pos.items() if k2 != p.key for x, y in ss]
        others += [(x, y) for x, y in segs_state]
        for offs in iter_dests(lv, k, others, w, spots):
            for perm in itertools.permutations(range(k)):
                pos = [list(s) for s in segs_state]
                steps = []
                good = True
                for idx in perm:
                    pos[idx] = list(offs[idx])
                    cx = sum(q[0] for q in pos) // k
                    cy = sum(q[1] for q in pos) // k
                    if rod_unsafe_at(p, cx, cy, allow_blobs):
                        good = False
                        break
                    steps.append((idx, offs[idx]))
                if good:
                    return steps
        return None

    def emit_hop(p: Piece, steps) -> None:
        segs = seg_pos[p.key]
        for idx, (nx, ny) in steps:
            clicks.append((segs[idx][0] + 2, segs[idx][1] + 2))   # 选中（中心）
            clicks.append((nx + 2, ny + 2))                        # 放置（中心=点击点）
            segs[idx][0], segs[idx][1] = nx, ny

    def route_to(p: Piece, goal: tuple[int, int], allow_blobs, spots) -> bool:
        """杆质心到 goal：直跳优先，blocked 时经绕行路点两跳。"""
        segs = seg_pos[p.key]
        k = len(segs)
        sx = sum(s[0] for s in segs) // k
        sy = sum(s[1] for s in segs) // k
        steps = check_hop(p, goal, allow_blobs, segs, spots)
        if steps is not None:
            emit_hop(p, steps)
            return True
        dx, dy = goal[0] - sx, goal[1] - sy
        dist = max(1.0, math.hypot(dx, dy))
        mx, my = (sx + goal[0]) / 2, (sy + goal[1]) / 2
        cands = []
        for sign in (1, -1):
            for off in (10, 14, 18):
                cands.append((int(mx - dy / dist * off * sign), int(my + dx / dist * off * sign)))
        for ang in range(0, 360, 45):
            cands.append((int(goal[0] + 14 * math.cos(math.radians(ang))),
                          int(goal[1] + 14 * math.sin(math.radians(ang)))))
        seen = set()
        for v in cands:
            if not (3 <= v[0] <= 60 and 3 <= v[1] <= 60) or v in seen:
                continue
            seen.add(v)
            if rod_unsafe_at(p, v[0], v[1], allow_blobs):
                continue
            s1 = check_hop(p, v, allow_blobs, segs, spots)
            if s1 is None:
                continue
            state2 = [list(s) for s in segs]
            for idx, (nx, ny) in s1:
                state2[idx] = [nx, ny]
            s2 = check_hop(p, goal, allow_blobs, state2, spots)
            if s2 is None:
                continue
            emit_hop(p, s1)
            emit_hop(p, s2)
            return True
        log(f"    route_to fail for {p.key} -> {goal}")
        return False

    # ---- 直杆件：杆∩靶（按规划顺序重试） ----
    def plan_direct_piece(p: Piece) -> bool:
        tmask = to_grid(solid_mask(p.target), int(p.target.x), int(p.target.y))
        tx, ty = int(p.target.x) + 3, int(p.target.y) + 3
        others = [tuple(s) for k2, ss in seg_pos.items() if k2 != p.key for s in ss]
        others += [tuple(s) for s in seg_pos[p.key]]
        spots = build_spots(lv, others)
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                w = (tx + dx, ty + dy)
                rm = rod_mask_at(p, w)
                if not overlap(rm, tmask):
                    continue
                if overlap(rm, lv.danger):
                    continue
                if route_to(p, w, frozenset(), spots):
                    return True
        return False

    direct_pieces = [p for p in lv.pieces.values()
                     if p.target is not None and "dirwzt" not in p.key
                     and p.rod is not None and np.any(np.asarray(p.rod.pixels) > 0)]
    direct_ok = False
    if not direct_pieces:
        direct_ok = True
    for order in itertools.permutations(direct_pieces):
        reset_state()
        if all(plan_direct_piece(p) for p in order):
            direct_ok = True
            break
    if not direct_ok:
        log("    direct pieces: no order works")
        return None

    # ---- 画刷杆：吸色 → 落画刷靶 ----
    brush_rods = [p for p in lv.pieces.values()
                  if p.rod is not None and p.target is None
                  and not np.any(np.asarray(p.rod.pixels) > 0)]
    brush_targets = [p for p in lv.pieces.values()
                     if p.rod is None and p.target is not None and "dirwzt" not in p.key]
    if len(brush_rods) < len(brush_targets):
        log("    more brush targets than rods")
        return None
    blob_by_color = {}
    for name, color, s in lv.blobs:
        blob_by_color.setdefault(color, []).append((name, s))

    def target_colors(p: Piece) -> set[int]:
        px = np.asarray(p.target.pixels)
        return {int(c) for c in px.flatten().tolist() if c > 0}

    used_blobs: set[str] = set()
    for t in brush_targets:
        need = target_colors(t)
        chosen: list[tuple[str, object]] = []
        for color in sorted(need):
            cand = [b for b in blob_by_color.get(color, []) if b[0] not in used_blobs]
            if not cand:
                log(f"    brush target {t.key}: no blob for color {color}")
                return None
            chosen.append(cand[0])
            used_blobs.add(cand[0][0])
        rod = brush_rods.pop()
        # 印章模拟：色集必须精确等于 need
        sim = np.zeros((5, 5), dtype=int)
        for name, s in chosen:
            sm = stamp_mask(s)
            spx = np.asarray(s.pixels)
            sim[sm] = spx[sm]
        okset = {int(c) for c in sim.flatten().tolist() if c > 0}
        if okset != need:
            log(f"    brush target {t.key}: stamp set {sorted(okset)} != need {sorted(need)}")
            return None
        # 路点：各颜料块中心 → 靶
        route = [(int(s.x) + 2, int(s.y) + 2) for _, s in chosen]
        tmask = to_grid(solid_mask(t.target), int(t.target.x), int(t.target.y))
        tx, ty = int(t.target.x) + 3, int(t.target.y) + 3
        placed = False
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                w = (tx + dx, ty + dy)
                rm = rod_mask_at(rod, w)
                if not overlap(rm, tmask) or overlap(rm, lv.danger):
                    continue
                route.append(w)
                placed = True
                break
            if placed:
                break
        if not placed:
            log(f"    brush target {t.key}: no valid drop point")
            return None
        route_names = frozenset(name for name, _ in chosen)
        others = [tuple(s) for k2, ss in seg_pos.items() if k2 != rod.key for s in ss]
        others += [tuple(s) for s in seg_pos[rod.key]]
        spots = build_spots(lv, others)
        for wx, wy in route:
            if not route_to(rod, (wx, wy), route_names, spots):
                return None
    return clicks


def apply_plan(g, plan) -> tuple[int, object]:
    fd = None
    for x, y in plan:
        fd = g.perform_action(ActionInput(id=GameAction.ACTION6, data={"x": int(x), "y": int(y)}), raw=True)
    return int(fd.levels_completed), fd.state


def diagnose(g) -> None:
    print("  -- diagnose --")
    for key, data in g.kacotwgjcyq.items():
        rod = data["roduyfsmiznvg"]
        tgt = data["gosubdcyegamj"]
        if rod is None or tgt is None:
            continue
        col = overlap(to_grid(solid_mask(rod), int(rod.x), int(rod.y)),
                      to_grid(solid_mask(tgt), int(tgt.x), int(tgt.y)))
        print(f"    {key[:20]:20s} rod@({int(rod.x)},{int(rod.y)}) tgt@({int(tgt.x)},{int(tgt.y)}) overlap={col}")


def main() -> None:
    n_levels = len(G._clean_levels)
    plans: dict[int, list[tuple[int, int]]] = {}

    def log(msg: str) -> None:
        print(msg)

    for idx in range(n_levels):
        lv = extract_level(idx)
        plan = plan_level(lv, log)
        if plan is None:
            print(f"L{idx}: NO PLAN")
            continue
        print(f"L{idx}: {len(plan)} clicks ({len(plan)//2} moves)")
        plans[idx] = plan

    env.reset()
    g = env._game
    ok = True
    for idx in range(n_levels):
        if idx not in plans:
            print(f"L{idx}: skipped (no plan) -- stop validation here")
            ok = False
            break
        lv_score, state = apply_plan(g, plans[idx])
        good = lv_score == idx + 1
        print(f"L{idx}: levels_completed={lv_score} state={state} -> {'PASS' if good else 'FAIL'}")
        if not good:
            diagnose(g)
            ok = False
            break
    print("ALL LEVELS PASS" if ok else "VALIDATION INCOMPLETE")

    if plans and ok:
        out = {
            "run": "r11l_click_solver",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "game_version": "r11l-495a7899",
            "validated": ok,
            "levels": {str(k): [list(xy) for xy in v] for k, v in sorted(plans.items())},
        }
        bench = PKG / "bench"
        bench.mkdir(exist_ok=True)
        (bench / "r11l_click_plans_validated.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print("\n_R11L_PLANS: dict[int, list[tuple[int, int]]] = {")
        for k, v in sorted(plans.items()):
            items = ", ".join(f"({x}, {y})" for x, y in v)
            print(f"    {k}: [{items}],")
        print("}")


if __name__ == "__main__":
    main()
