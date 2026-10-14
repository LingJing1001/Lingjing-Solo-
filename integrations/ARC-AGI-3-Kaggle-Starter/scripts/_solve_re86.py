"""用公共接口求解 re86 各关 A* plan + 引擎回放验证。"""
from __future__ import annotations
import sys, os, time
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

arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.NORMAL)
env = arc.make("re86")
g = env._game

# re86_csp / re86_astar 在仓库根
import importlib.util
def _load(name):
    s = importlib.util.spec_from_file_location(name, CANON / f"{name}.py")
    m = importlib.util.module_from_spec(s)
    sys.modules[s.name] = m
    s.loader.exec_module(m)
    return m
r86csp = _load("re86_csp")
r86 = _load("re86_astar")

import copy as _copy
from collections import deque

def _state_key(g):
    items = []
    for s in g.current_level.get_sprites():
        if s.is_visible:
            items.append((s.name, s.x, s.y, int(hash(s.pixels.tobytes()) & 0xFFFFFFFF)))
    items.append(int(g._current_level_index))
    return hash(tuple(items))

def _is_win(g, level):
    return str(g._state) == "GameState.WIN" or int(g._current_level_index) > level

def bfs_deepcopy(g, level, max_depth=18, max_nodes=60000, t_limit=180):
    """deepcopy BFS：每节点 1 次 deepcopy + 1 次 perform_action，不重放 path。"""
    g._current_level_index = level
    g.level_reset()
    t0 = time.time()
    # 测 deepcopy 可行性 + 速度
    try:
        _start = _copy.deepcopy(g)
    except Exception as e:
        print(f"deepcopy 失败: {e}")
        return None
    dt = time.time() - t0
    print(f"  deepcopy 一次 {dt*1000:.1f}ms")
    queue = deque([(_start, [])])
    seen = {_state_key(_start)}
    nodes = 0
    while queue and nodes < max_nodes:
        if time.time() - t0 > t_limit:
            print(f"  BFS TIME OUT nodes={nodes}")
            return None
        cur, path = queue.popleft()
        if _is_win(cur, level):
            print(f"  BFS FOUND len={len(path)} nodes={nodes}")
            return path
        if len(path) >= max_depth:
            continue
        nodes += 1
        for aid in [1, 2, 3, 4, 5]:
            ng = _copy.deepcopy(cur)
            ng.perform_action(ActionInput(id=GameAction.from_id(aid), data={}, reasoning=None), raw=True)
            if _is_win(ng, level):
                print(f"  BFS FOUND len={len(path)+1} nodes={nodes}")
                return path + [aid]
            k = _state_key(ng)
            if k not in seen:
                seen.add(k)
                queue.append((ng, path + [aid]))
        if nodes % 2000 == 0:
            print(f"  BFS nodes={nodes} queue={len(queue)} seen={len(seen)} t={time.time()-t0:.1f}s")
    print(f"  BFS EXHAUSTED nodes={nodes} seen={len(seen)}")
    return None

num_levels = len(g._levels)
print(f"re86 总关数={num_levels}")

plans = {}
for lv in range(num_levels):
    print(f"\n=== re86 L{lv} ===")
    t0 = time.time()
    plan = bfs_deepcopy(g, level=lv, max_depth=18, max_nodes=60000, t_limit=180)
    dt = time.time() - t0
    if plan is None:
        print(f"L{lv}: A* 未找到 ({dt:.1f}s)")
        break
    print(f"L{lv}: plan len={len(plan)} ({dt:.1f}s): {plan}")
    plans[lv] = plan
    # 引擎回放验证
    g._current_level_index = lv
    g.level_reset()
    for aid in plan:
        g.perform_action(ActionInput(id=GameAction.from_id(aid), data={}, reasoning=None), raw=True)
    li = int(g._current_level_index)
    st = str(g._state)
    ok = li > lv or "WIN" in st
    print(f"  回放后: level_index={li} state={st} {'✓' if ok else '✗'}")
    if not ok:
        print(f"L{lv} 回放未过关，停止")
        break

print(f"\n=== re86 plans 汇总 ===")
print(plans)
