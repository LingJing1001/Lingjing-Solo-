"""按 r2r3r4_combo.py 的 R2+R3+R4 组合设计跑 LS20。

设计映射 (AR25 → LS20):
  R2 感知: 玩家 color 5 → color 12;  目标 color 11 → color 11;  墙 color 4
  R3 寻路: passable {0,5,10,11} → {0,3,9,11,12}  (不含 4=墙)
  R4 剪枝: 动作 {1,2,3,4,5} → {1,2,3,4}  (LS20 无切换)
"""
import sys, os, json, hashlib, collections, heapq
from collections import deque

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from paths import add_to_sys_path, environments_dir  # noqa: E402

add_to_sys_path()

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

# ── LS20 颜色语义 ──
C_PLAYER = 12   # 玩家精灵顶部 (稀少, 10格)
C_TARGET = 11   # 目标标记 (npxgalaybz)
C_WALL   = 4    # 墙/障碍
PASSABLE = {0, 3, 9, 11, 12}  # 可通行: 空地/背景/目标/玩家

# ═══════════════════════════════════════════════════════════
# R2: 感知 (LS20 适配)
# ═══════════════════════════════════════════════════════════

def r2_state_hash(grid):
    return hashlib.sha256(json.dumps(grid).encode()).hexdigest()[:16]

def r2_find_cells(grid, color):
    return [(x, y) for y in range(len(grid)) for x in range(len(grid[0])) if grid[y][x] == color]

def r2_find_connected_regions(cells):
    if not cells:
        return []
    cell_set, visited, regions = set(cells), set(), []
    for cell in cells:
        if cell not in visited:
            queue, region = deque([cell]), []
            while queue:
                x, y = queue.popleft()
                if (x, y) not in visited:
                    visited.add((x, y)); region.append((x, y))
                    for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
                        nx, ny = x+dx, y+dy
                        if (nx, ny) in cell_set and (nx, ny) not in visited:
                            queue.append((nx, ny))
            regions.append(region)
    return regions

def r2_perceive_ls20(grid):
    players = r2_find_cells(grid, C_PLAYER)
    targets = r2_find_cells(grid, C_TARGET)
    player_regions = r2_find_connected_regions(players)
    player_pos = None
    if player_regions:
        main = max(player_regions, key=len)
        player_pos = (int(sum(r[0] for r in main)/len(main)),
                      int(sum(r[1] for r in main)/len(main)))
    target_center = None
    if targets:
        target_center = (int(sum(t[0] for t in targets)/len(targets)),
                         int(sum(t[1] for t in targets)/len(targets)))
    return {
        'state_hash': r2_state_hash(grid),
        'player_pos': player_pos,
        'target_center': target_center,
        'player_count': len(players),
        'target_count': len(targets),
    }

# ═══════════════════════════════════════════════════════════
# R3: 寻路 (A*, LS20 passable)
# ═══════════════════════════════════════════════════════════

def r3_astar(grid, start, goal):
    h, w = len(grid), len(grid[0])
    def heur(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_set = [(0, heur(start, goal), start)]
    came_from = {start: None}
    g_score = {start: 0}
    while open_set:
        _, _, cur = heapq.heappop(open_set)
        if cur == goal:
            path = []
            while came_from[cur] is not None:
                prev = came_from[cur]
                dx, dy = cur[0]-prev[0], cur[1]-prev[1]
                path.append({(0,-1):1,(0,1):2,(-1,0):3,(1,0):4}[(dx,dy)])
                cur = prev
            return list(reversed(path))
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            nx, ny = cur[0]+dx, cur[1]+dy
            if 0 <= ny < h and 0 <= nx < w and grid[ny][nx] in PASSABLE:
                tg = g_score[cur] + 1.0
                if (nx,ny) not in g_score or tg < g_score[(nx,ny)]:
                    g_score[(nx,ny)] = tg
                    came_from[(nx,ny)] = cur
                    heapq.heappush(open_set, (tg+heur((nx,ny),goal), heur((nx,ny),goal), (nx,ny)))
    return None

# ═══════════════════════════════════════════════════════════
# R4: 剪枝 (LS20 动作 {1,2,3,4})
# ═══════════════════════════════════════════════════════════

def r4_score(action, player_pos, target_center, recent_actions):
    # 反循环
    window = 3
    loop = recent_actions[-window:].count(action) * 0.3 if len(recent_actions) >= window else 0.0
    # 目标接近度
    dx, dy = 0, 0
    if action == 1: dy = -1
    elif action == 2: dy = 1
    elif action == 3: dx = -1
    elif action == 4: dx = 1
    goal_prox = 0.0
    if player_pos and target_center:
        new_pos = (player_pos[0]+dx, player_pos[1]+dy)
        old_d = abs(player_pos[0]-target_center[0]) + abs(player_pos[1]-target_center[1])
        new_d = abs(new_pos[0]-target_center[0]) + abs(new_pos[1]-target_center[1])
        goal_prox = (old_d - new_d) * 0.5
    return goal_prox - loop

def r4_choose(player_pos, target_center, recent_actions):
    candidates = [(a, r4_score(a, player_pos, target_center, recent_actions))
                  for a in [1, 2, 3, 4]]
    candidates.sort(key=lambda x: -x[1])
    return candidates[0][0], candidates

# ═══════════════════════════════════════════════════════════
# 组合入口
# ═══════════════════════════════════════════════════════════

_path_cache, _path_index, _recent = [], 0, []
ACTION_NAMES = {1: 'UP↑', 2: 'DOWN↓', 3: 'LEFT←', 4: 'RIGHT→'}

def smart_step(grid):
    global _path_cache, _path_index
    p = r2_perceive_ls20(grid)
    player_pos, target_center = p['player_pos'], p['target_center']
    if not player_pos:
        return 2, p, 'no_player'  # 找不到玩家就往下走试试
    # 沿缓存路径
    if _path_cache and _path_index < len(_path_cache):
        a = _path_cache[_path_index]; _path_index += 1
        _recent.append(a)
        return a, p, 'path'
    # R3 寻路
    if target_center:
        path = r3_astar(grid, player_pos, target_center)
        if path:
            _path_cache, _path_index = path, 0
            a = _path_cache[_path_index]; _path_index += 1
            _recent.append(a)
            return a, p, 'astar'
    # R4 剪枝
    a, _ = r4_choose(player_pos, target_center, _recent)
    _recent.append(a)
    return a, p, 'prune'

# ═══════════════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════════════

def get_grid(frame):
    ff = frame.frame
    if isinstance(ff, list) and len(ff) == 1 and isinstance(ff[0], np.ndarray):
        return ff[0].astype(int).tolist()
    return np.array(ff).astype(int).tolist()

def main():
    STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    arcade = Arcade(environments_dir=environments_dir("ls20"),
                    operation_mode=OperationMode.OFFLINE)
    gid = [e.game_id for e in arcade.get_environments() if e.game_id.startswith("ls20")][0]
    print(f"game_id = {gid}")
    env = arcade.make(gid)
    frame = env.reset()
    grid = get_grid(frame)
    p = r2_perceive_ls20(grid)
    print(f"reset → 玩家={p['player_pos']} 目标={p['target_center']} "
          f"玩家格={p['player_count']} 目标格={p['target_count']} "
          f"levels={frame.levels_completed} state={frame.state}")
    print(f"{'步':>3} {'动作':<8} {'方法':<6} {'玩家':<12} {'目标':<12} {'lvl':>3} {'state'}")
    print("-" * 70)

    for step in range(1, STEPS + 1):
        action, perc, method = smart_step(grid)
        gaction = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
                   3: GameAction.ACTION3, 4: GameAction.ACTION4}[action]
        frame = env.step(gaction, data=None)
        grid = get_grid(frame)
        p2 = r2_perceive_ls20(grid)
        print(f"{step:3d} {ACTION_NAMES[action]:<8} {method:<6} "
              f"{str(p2['player_pos']):<12} {str(p2['target_center']):<12} "
              f"{frame.levels_completed:3d} {frame.state}")
        if frame.state in (GameState.WIN, GameState.GAME_OVER):
            print(f"\n游戏结束: state={frame.state} levels={frame.levels_completed}")
            break
    print(f"\n路径缓存: {len(_path_cache)}步, 已走 {_path_index}")
    print(f"动作历史: {[ACTION_NAMES[a] for a in _recent]}")

if __name__ == '__main__':
    main()