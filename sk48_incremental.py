# -*- coding: utf-8 -*-
"""SK48 增量状态恢复 A* 求解器

关键优化：使用 Undo 代替完全重置，减少状态恢复开销
"""
import sys
import os
import time
import heapq
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KAGGLE = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"
sys.path.insert(0, str(KAGGLE))
os.environ.setdefault("LINGJING_SRC", str(ROOT / "lingjing_solo"))
sys.path.insert(0, str(ROOT))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, ActionInput, GameState
import numpy as np


def get_state_hash(g):
    """获取状态哈希 - 不包含能量（因为 Undo 不恢复能量）"""
    state = []
    for s in g.current_level.get_sprites():
        if s.is_visible:
            state.append((s.name, s.x, s.y, s.rotation))
    if hasattr(g, 'vzvypfsnt') and g.vzvypfsnt:
        state.append(('selected', g.vzvypfsnt.name))
    state.append(g._current_level_index)
    return hash(tuple(state))


def get_heuristic(g):
    """启发式函数 - 基于珠子匹配进度"""
    if g._state == GameState.WIN:
        return 0
    
    matched = 0
    total = 0
    
    if hasattr(g, 'vjfbwggsd') and hasattr(g, 'xpmcmtbcv'):
        for bcwsrdcswp, pphvqzjirf in g.xpmcmtbcv.items():
            if bcwsrdcswp in g.vjfbwggsd and pphvqzjirf in g.vjfbwggsd:
                path1 = g.vjfbwggsd[bcwsrdcswp]
                path2 = g.vjfbwggsd[pphvqzjirf]
                total += max(len(path1), len(path2))
                for i in range(min(len(path1), len(path2))):
                    if path1[i].pixels[1, 1] == path2[i].pixels[1, 1]:
                        matched += 1
    
    h = total - matched
    energy = getattr(g, 'qiercdohl', 196)
    if energy < 50:
        h += (50 - energy) * 2
    
    return h


def get_actions(g):
    """获取可用动作"""
    actions = []
    
    # 移动动作
    for n in [1, 2, 3, 4]:
        actions.append(('move', n))
    
    # 点击动作
    for s in g.current_level.get_sprites():
        if s.is_visible and hasattr(s, 'tags') and 'sys_click' in s.tags:
            cx = s.x + s.width // 2
            cy = s.y + s.height // 2
            actions.append(('click', cx, cy))
    
    return actions


def apply_action(g, action):
    """应用动作"""
    if action[0] == 'move':
        g.perform_action(ActionInput(id=GameAction.from_id(action[1]), data={}, reasoning=None), raw=True)
    elif action[0] == 'click':
        g.perform_action(ActionInput(id=GameAction.ACTION6, data={'x': action[1], 'y': action[2]}, reasoning=None), raw=True)


def undo_action(g):
    """撤销动作"""
    g.perform_action(ActionInput(id=GameAction.ACTION7, data={}, reasoning=None), raw=True)


def is_win(g, level):
    """检查是否过关"""
    return g._state == GameState.WIN or int(g._current_level_index) > level


def solve_level_dfs(g, level, max_time=300, max_depth=100):
    """DFS 求解 - 使用 Undo 进行增量状态恢复
    
    关键优化：不再每次重置关卡，而是用 Undo 回退
    """
    t0 = time.time()
    
    g._current_level_index = level
    g.level_reset()
    
    visited = set()
    best_solution = None
    best_len = max_depth + 1
    
    def dfs(path, depth):
        nonlocal best_solution, best_len
        
        if time.time() - t0 > max_time:
            return False
        
        if len(path) >= best_len:
            return False
        
        if is_win(g, level):
            if len(path) < best_len:
                best_len = len(path)
                best_solution = path.copy()
                print(f"  [DFS] FOUND SOLUTION in {len(path)} steps", flush=True)
            return True
        
        if len(path) >= max_depth:
            return False
        
        state_h = get_state_hash(g)
        if state_h in visited:
            return False
        visited.add(state_h)
        
        for action in get_actions(g):
            # 应用动作
            apply_action(g, action)
            path.append(action)
            
            # 递归搜索
            found = dfs(path, depth + 1)
            
            # 撤销动作 - 增量恢复
            undo_action(g)
            path.pop()
            
            if found and len(path) == 0:
                return True
        
        visited.discard(state_h)
        return False
    
    # 迭代加深
    for depth in range(10, max_depth + 1, 5):
        print(f"  [DFS] Trying depth={depth}", flush=True)
        visited.clear()
        g._current_level_index = level
        g.level_reset()
        
        if dfs([], 0):
            if best_solution:
                return best_solution
        
        if time.time() - t0 > max_time:
            print(f"  [DFS] TIME OUT", flush=True)
            break
    
    return best_solution


def solve_level_astar_incremental(g, level, max_time=300, max_nodes=100000):
    """A* 求解 - 增量状态恢复版本
    
    关键优化：使用 Undo 回退，而不是完全重置
    """
    t0 = time.time()
    
    g._current_level_index = level
    g.level_reset()
    
    # 优先队列: (f, g_cost, counter, path_length)
    counter = 0
    h0 = get_heuristic(g)
    heap = [(h0, 0, counter, 0)]
    visited = {get_state_hash(g): 0}
    
    nodes = 0
    current_path = []
    
    while heap and nodes < max_nodes:
        if time.time() - t0 > max_time:
            print(f"  [A*] TIME OUT nodes={nodes}", flush=True)
            return None
        
        f, g_cost, _, target_len = heapq.heappop(heap)
        
        # 增量恢复：用 Undo 回退到 target_len
        while len(current_path) > target_len:
            undo_action(g)
            current_path.pop()
        
        # 如果需要前进，从初始状态开始
        if len(current_path) < target_len:
            # 需要完全重置（这种情况应该很少）
            g._current_level_index = level
            g.level_reset()
            current_path = []
        
        # 检查是否过关
        if is_win(g, level):
            print(f"  [A*] FOUND SOLUTION in {len(current_path)} steps, {nodes} nodes", flush=True)
            return current_path.copy()
        
        nodes += 1
        
        if nodes % 1000 == 0:
            h = get_heuristic(g)
            print(f"  [A*] nodes={nodes} g={g_cost} h={h} heap={len(heap)} path={len(current_path)}", flush=True)
        
        # 生成候选动作
        for action in get_actions(g):
            # 应用动作
            apply_action(g, action)
            current_path.append(action)
            
            # 检查是否过关
            if is_win(g, level):
                print(f"  [A*] FOUND SOLUTION in {len(current_path)} steps, {nodes} nodes", flush=True)
                return current_path.copy()
            
            # 计算新的状态
            new_h = get_heuristic(g)
            new_g = g_cost + 1
            new_f = new_g + new_h
            new_state_h = get_state_hash(g)
            
            if new_state_h not in visited or visited[new_state_h] > new_g:
                visited[new_state_h] = new_g
                counter += 1
                heapq.heappush(heap, (new_f, new_g, counter, len(current_path)))
            
            # 撤销动作 - 增量恢复
            undo_action(g)
            current_path.pop()
    
    print(f"  [A*] EXHAUSTED nodes={nodes}", flush=True)
    return None


def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("sk48")
    
    if env is None:
        print("无法创建 sk48 环境")
        return
    
    print("=== SK48 增量状态恢复 A* 求解器 ===")
    print("优化：使用 Undo 代替完全重置")
    
    g = env._game
    num_levels = len(g._levels)
    print(f"总关卡数: {num_levels}")
    
    total_actions = 0
    all_ok = True
    
    for lvl in range(num_levels):
        t0 = time.time()
        
        sol = solve_level_astar_incremental(g, lvl, max_time=300, max_nodes=100000)
        
        dt = time.time() - t0
        
        if sol is None:
            print(f"L{lvl+1}: NO SOLUTION  {dt:.1f}s")
            all_ok = False
            break
        
        # 验证
        g._current_level_index = lvl
        g.level_reset()
        for a in sol:
            apply_action(g, a)
        
        ok = is_win(g, lvl)
        total_actions += len(sol)
        
        print(f"L{lvl+1}: {len(sol)} actions -> {'OK' if ok else 'FAIL'}  {dt:.1f}s")
        
        if not ok:
            all_ok = False
            break
    
    print(f"\nTotal: actions={total_actions}")
    print("ALL_PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
