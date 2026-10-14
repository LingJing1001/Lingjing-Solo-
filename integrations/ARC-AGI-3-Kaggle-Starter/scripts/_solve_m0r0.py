"""m0r0 求解：镜像合并 + 引擎回放验证。

display_to_grid 映射: display = grid * 5 + 5 (camera 11x11)
机制:
- 点击非 mosdlc → mirror mode (pyhtlpzlmnr=True)
- mirror mode 下 ACTION1-4 移动所有 pikgci:
  leklkn=(dx,dy), rivmdg=(-dx,dy), boweok-leklkn=(dx,-dy), boweok-rivmdg=(-dx,-dy)
- win: 所有 pikgci 合并 (同位置 → INTANGIBLE)
"""
from __future__ import annotations
import sys, os, copy, time, json
from pathlib import Path
from collections import deque

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
env = arc.make("m0r0")
g = env._game


def _is_win(g, level):
    st = str(g._state)
    return st == "GameState.WIN" or int(g._current_level_index) > level


def do_action(g, item):
    if isinstance(item, int):
        g.perform_action(ActionInput(id=GameAction.from_id(item), data={}), raw=True)
    else:
        aid, x, y = item
        if aid == 6:
            g.perform_action(ActionInput(id=GameAction.ACTION6,
                                         data={"x": int(x), "y": int(y)}), raw=True)
        else:
            g.perform_action(ActionInput(id=GameAction.from_id(aid), data={}), raw=True)


def show_state(g, level):
    li = int(g._current_level_index)
    st = str(g._state)
    print(f"  li={li} st={st} win={_is_win(g, level)}")
    for name in ["pikgci-toljda-leklkn", "pikgci-toljda-rivmdg",
                 "pikgci-boweok-leklkn", "pikgci-boweok-rivmdg"]:
        sp = g.current_level.get_sprites_by_name(name)
        if sp:
            s = sp[0]
            print(f"    {name}: x={s.x} y={s.y} inter={s.interaction}")


def grid_to_display(gx, gy, scale=5, offset=5):
    return (gx * scale + offset, gy * scale + offset)


def get_click_targets(g, lv):
    """获取 sys_click sprite 的 grid 坐标 + display 坐标。"""
    g._current_level_index = lv
    g.level_reset()
    targets = []
    for s in g.current_level.get_sprites():
        if s.is_visible and hasattr(s, "tags") and "sys_click" in (s.tags or []):
            dx, dy = grid_to_display(s.x, s.y)
            targets.append((s.name, s.x, s.y, dx, dy))
    return targets


def verify_plan(g, lv, plan):
    g._current_level_index = lv
    g.level_reset()
    for item in plan:
        do_action(g, item)
        if _is_win(g, lv):
            return True, int(g._current_level_index), str(g._state)
    return _is_win(g, lv), int(g._current_level_index), str(g._state)


num_levels = len(g._levels)
print(f"m0r0 总关数={num_levels}")

# === L0: 2 pikgci at (3,9) and (7,9), mirror right ×2 → merge at (5,9) ===
print("\n=== m0r0 L0 ===")
targets = get_click_targets(g, 0)
print(f"  click targets: {targets}")
# 点击 leklkn 进入 mirror mode，然后右移 2 次
dx, dy = grid_to_display(3, 9)
plan_l0 = [(6, dx, dy), 4, 4]
ok, li, st = verify_plan(g, 0, plan_l0)
print(f"  plan={plan_l0} len={len(plan_l0)}")
print(f"  回放: li={li} st={st} {'✓' if ok else '✗'}")
if not ok:
    show_state(g, 0)
    # 尝试更多步
    for n_right in range(1, 6):
        g._current_level_index = 0
        g.level_reset()
        plan = [(6, dx, dy)] + [4] * n_right
        ok2, li2, st2 = verify_plan(g, 0, plan)
        print(f"  try {n_right} right: li={li2} st={st2} {'✓' if ok2 else '✗'}")
        if ok2:
            plan_l0 = plan
            ok = True
            break
    if not ok:
        # 也试左移
        for n_left in range(1, 6):
            g._current_level_index = 0
            g.level_reset()
            plan = [(6, dx, dy)] + [3] * n_left
            ok2, li2, st2 = verify_plan(g, 0, plan)
            print(f"  try {n_left} left: li={li2} st={st2} {'✓' if ok2 else '✗'}")
            if ok2:
                plan_l0 = plan
                ok = True
                break

# === L1: 29 sprites, many spswjz ===
print("\n=== m0r0 L1 ===")
g._current_level_index = 1
g.level_reset()
targets1 = get_click_targets(g, 1)
print(f"  click targets: {targets1}")
# L1 有 spswjz (触发动画) 和 2 个 pikgci
# 先看初始状态
show_state(g, 1)
# pikgci 位置
for name in ["pikgci-toljda-leklkn", "pikgci-toljda-rivmdg"]:
    sp = g.current_level.get_sprites_by_name(name)
    if sp:
        s = sp[0]
        print(f"    {name}: x={s.x} y={s.y}")

# === L2: 3 mosdlc + 2 pikgci ===
print("\n=== m0r0 L2 ===")
g._current_level_index = 2
g.level_reset()
targets2 = get_click_targets(g, 2)
print(f"  click targets: {targets2}")
show_state(g, 2)

# === BFS 求解 L0 (已知解，但验证 BFS 框架) ===
print("\n=== BFS 验证 L0 ===")
def state_key(g):
    items = []
    for s in g.current_level.get_sprites():
        if s.is_visible:
            items.append((s.name, s.x, s.y, s.interaction))
    items.append(int(g._current_level_index))
    return hash(tuple(items))

def bfs_solve(g, level, actions, max_depth=40, max_nodes=100000, t_limit=300):
    g._current_level_index = level
    g.level_reset()
    if _is_win(g, level):
        return []
    t0 = time.time()
    start = copy.deepcopy(g)
    sk = state_key(start)
    parent = {sk: (None, None)}
    queue = deque([(start, sk, 0)])
    nodes = 0
    while queue and nodes < max_nodes:
        if time.time() - t0 > t_limit:
            print(f"  BFS TIMEOUT nodes={nodes} seen={len(parent)} t={time.time()-t0:.1f}s")
            return None
        cur, ck, depth = queue.popleft()
        if depth >= max_depth:
            continue
        nodes += 1
        if nodes % 5000 == 0:
            print(f"  BFS nodes={nodes} queue={len(queue)} seen={len(parent)} depth={depth} t={time.time()-t0:.1f}s")
        for act in actions:
            ng = copy.deepcopy(cur)
            do_action(ng, act)
            if _is_win(ng, level):
                path = [act]
                k = ck
                while parent[k][0] is not None:
                    pk, a = parent[k]
                    path.append(a)
                    k = pk
                path.reverse()
                print(f"  BFS FOUND len={len(path)} nodes={nodes} t={time.time()-t0:.1f}s")
                return path
            nk = state_key(ng)
            if nk not in parent:
                parent[nk] = (ck, act)
                queue.append((ng, nk, depth + 1))
    print(f"  BFS EXHAUSTED nodes={nodes} seen={len(parent)}")
    return None

# L0: 动作 = 1-5 + click on each sys_click target
g._current_level_index = 0
g.level_reset()
clicks_l0 = []
for s in g.current_level.get_sprites():
    if s.is_visible and hasattr(s, "tags") and "sys_click" in (s.tags or []):
        dx, dy = grid_to_display(s.x, s.y)
        clicks_l0.append((6, dx, dy))
# 也加一个空点击 (0,0)
clicks_l0.append((6, 0, 0))
actions_l0 = [1, 2, 3, 4, 5] + clicks_l0
print(f"  L0 actions: {actions_l0}")
plan = bfs_solve(g, 0, actions_l0, max_depth=30, max_nodes=100000, t_limit=300)
if plan:
    ok, li, st = verify_plan(g, 0, plan)
    print(f"  L0 BFS plan={plan} len={len(plan)} 回放={'✓' if ok else '✗'}")

# L1 BFS
print("\n=== BFS L1 ===")
g._current_level_index = 1
g.level_reset()
clicks_l1 = []
for s in g.current_level.get_sprites():
    if s.is_visible and hasattr(s, "tags") and "sys_click" in (s.tags or []):
        dx, dy = grid_to_display(s.x, s.y)
        clicks_l1.append((6, dx, dy))
clicks_l1.append((6, 0, 0))
actions_l1 = [1, 2, 3, 4, 5] + clicks_l1
print(f"  L1 actions: {actions_l1}")
plan1 = bfs_solve(g, 1, actions_l1, max_depth=40, max_nodes=100000, t_limit=300)
if plan1:
    ok1, li1, st1 = verify_plan(g, 1, plan1)
    print(f"  L1 BFS plan={plan1} len={len(plan1)} 回放={'✓' if ok1 else '✗'}")

# L2 BFS
print("\n=== BFS L2 ===")
g._current_level_index = 2
g.level_reset()
clicks_l2 = []
for s in g.current_level.get_sprites():
    if s.is_visible and hasattr(s, "tags") and "sys_click" in (s.tags or []):
        dx, dy = grid_to_display(s.x, s.y)
        clicks_l2.append((6, dx, dy))
clicks_l2.append((6, 0, 0))
actions_l2 = [1, 2, 3, 4, 5] + clicks_l2
print(f"  L2 actions: {actions_l2}")
plan2 = bfs_solve(g, 2, actions_l2, max_depth=40, max_nodes=100000, t_limit=300)
if plan2:
    ok2, li2, st2 = verify_plan(g, 2, plan2)
    print(f"  L2 BFS plan={plan2} len={len(plan2)} 回放={'✓' if ok2 else '✗'}")
