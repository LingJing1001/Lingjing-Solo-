"""sb26 专用点击求解器：真机布局 dump + 织布路径模拟 + 岔路摆放搜索 + 真机回放验证。

机制依据（sb26.py 7fbdac44，实体名已核对）:
- 两段式点击：点调色块(lngftsryyw/vgszefyyyp, y>53, sys_click)选中 → 点空槽(susublrply)放置；
  每次放置 -1 能量(共64)，能量 0 判负。槽内块/槽内岔路在 on_set_level 被摘除 sys_click = 固定。
- ACTION5 提交(-1 能量)后进入织布：从 qaagahahj[0]（框架按 (y,x) 排序）逐槽走，
  槽上色块颜色 == 当前顶栏(quhhhthrri, pixels[0,0])颜色 → 上色并推进；
  岔路块(vgszefyyyp) → 跳转到同色框架 slot0；空槽被走到 = 直接失败；
  全部顶栏上色 → WIN；颜色错/走完没上完 → 全盘擦除软失败（摆放保留，可再提交）。
- 框架槽位坐标 = (frame.x+2+i*6, frame.y+2)，i < int(frame.name[-1])。
- 重复颜色顶栏靠"岔路弹出后重访框架"让同一块多次刷色（idx4/5/7）。

搜索空间：可移动岔路(调色区 vgszefyyyp) → 空槽的摆放组合；摆放定下后走路径模拟，
每个待填槽所需颜色被顶栏唯一确定，色块分配无自由度 → 模拟确定性。
输出：每关 display 点击序列（grid==display, 64x64 scale=1），真机回放验证后打印
_SB26_PLANS 字面量并写 bench JSON。
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

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
env = arc.make("sb26")
env.reset()
G = env._game

PIECE, FORK, SPOT = "lngftsryyw", "vgszefyyyp", "susublrply"


@dataclass
class Frame:
    x: int
    y: int
    color: int
    slots: list[tuple[int, int]]


@dataclass
class Level:
    idx: int
    frames: list[Frame]          # qaagahahj 顺序 (y,x)
    bars: list[int]              # 顶栏颜色, (y,x) 排序
    fixed: dict[tuple[int, int], tuple[str, int]]   # slot -> ('piece'|'fork', color)
    empty: list[tuple[int, int]]                    # 空槽
    movable: list[tuple[str, int, tuple[int, int]]]  # (kind, color, origin) 调色区可点块


def cover(sprites, x: int, y: int):
    for s in sprites:
        if s.name in (PIECE, FORK, SPOT) and s.x <= x < s.x + s.width and s.y <= y < s.y + s.height:
            return s
    return None


def extract_level(idx: int) -> Level:
    sp = G._clean_levels[idx].get_sprites()
    frames = []
    for f in sorted([s for s in sp if "pkpgflvjel" in s.tags], key=lambda s: (s.y, s.x)):
        n = int(f.name[-1])
        slots = [(f.x + 2 + i * 6, f.y + 2) for i in range(n)]
        frames.append(Frame(int(f.x), int(f.y), int(f.pixels[0, 0]), slots))
    bars = [int(b.pixels[0, 0]) for b in
            sorted([s for s in sp if s.name == "quhhhthrri"], key=lambda s: (s.y, s.x))]
    fixed: dict[tuple[int, int], tuple[str, int]] = {}
    empty: list[tuple[int, int]] = []
    slot_of = {s: (fi, si) for fi, fr in enumerate(frames) for si, s in enumerate(fr.slots)}
    covered: set[tuple[int, int]] = set()
    for fi, fr in enumerate(frames):
        for si, (sx, sy) in enumerate(fr.slots):
            hit = cover(sp, sx, sy)
            if hit is None:
                raise RuntimeError(f"L{idx} slot {(sx, sy)} has no sprite")
            covered.add((hit.x, hit.y))
            if hit.name == SPOT:
                empty.append((fi, si))
            elif hit.name == FORK:
                fixed[(fi, si)] = ("fork", int(hit.pixels[1, 1]))
            else:
                fixed[(fi, si)] = ("piece", int(hit.pixels[1, 1]))
    movable = []
    for s in sp:
        if s.name in (PIECE, FORK) and s.y > 53 and (s.x, s.y) not in covered:
            movable.append((FORK if s.name == FORK else PIECE, int(s.pixels[1, 1]), (int(s.x), int(s.y))))
    movable.sort(key=lambda t: t[2])
    return Level(idx, frames, bars, fixed, empty, movable)


def construct(lv: Level, fork_slots: dict[int, tuple[int, int]]):
    """给定岔路摆放，模拟织布路径；成功返回 [(origin, dest)] 移动清单。"""
    assign: dict[tuple[int, int], tuple[str, int, tuple[int, int] | None]] = {}
    for key, (kind, color) in lv.fixed.items():
        assign[key] = (kind, color, None)
    for fi, (kind, color, origin) in enumerate(lv.movable):
        if kind == FORK and fi in fork_slots:
            assign[fork_slots[fi]] = ("fork", color, origin)
    pool = [(color, origin) for kind, color, origin in lv.movable if kind == PIECE]
    fcolor = {}
    for i, fr in enumerate(lv.frames):
        fcolor.setdefault(fr.color, i)

    stack = [(0, 0)]
    pps = False
    pm = 0
    painted = [False] * len(lv.bars)
    moves: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for _ in range(2000):
        fi, si = stack[-1]
        key = (fi, si)
        s = assign.get(key)
        if s is None:
            need = lv.bars[pm]
            cand = next(((c, o) for c, o in pool if c == need), None)
            if cand is None:
                return None
            assign[key] = ("piece", cand[0], cand[1])
            moves.append((cand[1], lv.frames[fi].slots[si]))
            pool = [p for p in pool if p[1] != cand[1]]
            continue
        if s and s[0] in ("piece", "fixed") and not painted[pm]:
            if s[1] != lv.bars[pm]:
                return None
            painted[pm] = True
            continue
        if pm == len(lv.bars) - 1 and painted[pm]:
            return moves
        if pps or (s and s[0] in ("piece", "fixed")):
            if pps:
                pps = False
            ni = si + 1
            if ni < len(lv.frames[fi].slots):
                pm += 1
                if pm >= len(lv.bars):
                    return None
                stack[-1] = (fi, ni)
                continue
            if len(stack) > 1:
                stack.pop()
                pps = True
                continue
            return None
        if s and s[0] == "fork":
            if si == 0 and key in stack[:-1] and stack[-2][1] == 0:
                return None
            nf = fcolor.get(s[1])
            if nf is None:
                return None
            stack.append((nf, 0))
            continue
        return None
    return None


def solve_level(lv: Level):
    """枚举可移动岔路的摆放（含留在调色区），返回 (摆放, 岔路移动, 色块移动)。"""
    fork_ids = [i for i, (kind, _, _) in enumerate(lv.movable) if kind == FORK]
    empties = lv.empty
    choices = [None] + empties

    def rec(bi: int, used: set, cur: dict):
        if bi == len(fork_ids):
            moves = construct(lv, cur)
            if moves is not None:
                return dict(cur), moves
            return None
        for c in choices:
            if c is not None and c in used:
                continue
            if c is not None:
                used.add(c)
            cur[fork_ids[bi]] = c
            r = rec(bi + 1, used, cur)
            if r:
                return r
            cur.pop(fork_ids[bi], None)
            if c is not None:
                used.discard(c)
        return None

    return rec(0, set(), {})


def apply_plan(g, plan) -> tuple[int, int]:
    """真机执行点击序列，返回 (levels_completed, state)。"""
    fd = None
    for act, xy in plan:
        if act == 5:
            inp = ActionInput(id=GameAction.ACTION5)
        else:
            inp = ActionInput(id=GameAction.ACTION6, data={"x": int(xy[0]), "y": int(xy[1])})
        fd = g.perform_action(inp, raw=True)
    assert fd is not None
    return int(fd.levels_completed), fd.state


def main() -> None:
    levels = [extract_level(i) for i in range(len(G._clean_levels))]
    plans: dict[int, list] = {}
    for lv in levels:
        sol = solve_level(lv)
        if sol is None:
            print(f"L{lv.idx}: NO SOLUTION (fork placements exhausted)")
            continue
        placement, piece_moves = sol
        fork_moves = [
            (lv.movable[fi][2], lv.frames[slot[0]].slots[slot[1]])
            for fi, slot in placement.items() if slot is not None
        ]
        moves = fork_moves + piece_moves
        plan: list[tuple[int, tuple[int, int] | None]] = []
        for origin, dest in moves:
            plan += [(6, origin), (6, dest)]
        plan.append((5, None))
        n_slots = sum(len(f.slots) for f in lv.frames)
        print(f"L{lv.idx}: {len(moves)} moves + 1 submit = {len(plan)} actions "
              f"(slots={n_slots}, bars={len(lv.bars)})")
        plans[lv.idx] = plan

    # 真机回放验证：从 L0 一路打到 WIN
    env.reset()
    g = env._game
    ok = True
    for idx in range(len(levels)):
        if idx not in plans:
            print(f"L{idx}: skipped (no plan)")
            continue
        lv_score, state = apply_plan(g, plans[idx])
        expect = idx + 1
        good = lv_score == expect
        ok &= good
        print(f"L{idx}: levels_completed={lv_score} state={state} -> {'PASS' if good else 'FAIL'}")
        if not good:
            ok = False
            break
    print("ALL LEVELS PASS" if ok else "VALIDATION INCOMPLETE")

    if plans:
        out = {
            "run": "sb26_click_solver",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "game_version": "sb26-7fbdac44",
            "validated": ok,
            "levels": {
                str(k): [{"a": a, "xy": list(xy) if xy else None} for a, xy in v]
                for k, v in sorted(plans.items())
            },
        }
        bench = PKG / "bench"
        bench.mkdir(exist_ok=True)
        out_path = bench / "sb26_click_plans_validated.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"wrote {out_path}")

        print("\n_SB26_PLANS: dict[int, list[tuple[int, int, int]]] = {")
        for k, v in sorted(plans.items()):
            items = ", ".join(
                f"({a}, {xy[0]}, {xy[1]})" if xy else f"({a}, 0, 0)" for a, xy in v
            )
            print(f"    {k}: [{items}],")
        print("}")


if __name__ == "__main__":
    main()
