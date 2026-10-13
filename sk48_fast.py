# -*- coding: utf-8 -*-
"""SK48 高效增量 A* 求解器

关键优化：使用 Undo 逐步回退，避免完全重置
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
    """状态哈希 - 排序后比较"""
    state = []
    for s in g.current_level.get_sprites():
        if s.is_visible:
            state.append((s.name, s.x, s.y, s.rotation))
    state.sort()
    if hasattr(g, 'vzvypfsnt') and g.vzvypfsnt:
        state.append(('selected', g.vzvypfsnt.name))
    state.append(g._current_level_index)
    return hash(tuple(state))


def get_heuristic(g):
    """启发式 - 基于匹配进度"""
    if g._state == GameState.WIN:
        return 0
    
    matched = 0
    total = 0
    path_diff = 0
    
    if hasattr(g, 'vjfbwggsd') and hasattr(g, 'xpmcmtbcv'):
        for bcwsrdcswp, pphvqzjirf in g.xpmcmtbcv.items():
            if bcwsrdcswp in g.vjfbwggsd and pphvqzjirf in g.vjfbwggsd:
                path1 = g.vjfbwggsd[bcwsrdcswp]
                path2 = g.vjfbwggsd[pphvqzjirf]
                total += max(len(path1), len(path2))
                path_diff += abs(len(path1) - len(path2))
                for i in range(min(len(path1), len(path2))):
                    if path1[i].pixels[1, 1] == path2[i].pixels[1, 1]:
                        matched += 1
    
    h = (total - matched) + path_diff
    energy = getattr(g, 'qiercdohl', 196)
    if energy < 50:
        h += (50 - energy) * 2
    return h


def get_actions(g):
    """获取可用动作"""
    actions = []
    for n in [1, 2, 3, 4]:
        actions.append(('move', n))
    for s in g.current_level.get_sprites():
        if s.is_visible and hasattr(s, 'tags') and 'sys_click' in s.tags:
            actions.append(('click', s.x + s.width // 2, s.y + s.height // 2))
    return actions


def apply_action(g, action):
    if action[0] == 'move':
        g.perform_action(ActionInput(id=GameAction.from_id(action[1]), data={}, reasoning=None), raw=True)
    elif action[0] == 'click':
        g.perform_action(ActionInput(id=GameAction.ACTION6, data={'x': action[1], 'y': action[2]}, reasoning=None), raw=True)


def undo_action(g):
    g.perform_action(ActionInput(id=GameAction.ACTION7, data={}, reasoning=None), raw=True)


def is_win(g, level):
    return g._state == GameState.WIN or int(g._current_level_index) > level


def solve_level(g, level, max_time=300):
    """增量 A* 求解 - 使用 Undo 回退"""
    t0 = time.time()
    
    g._current_level_index = level
    g.level_reset()
    
    # 优先队列: (f, g_cost, counter, path)
    counter = 0
    h0 = get_heuristic(g)
    heap = [(h0, 0, counter, [])]
    visited = {get_state_hash(g): 0}
    
    # 当前路径 - 用于增量回退
    current_path = []
    
    nodes = 0
    
    while heap:
        if time.time() - t0 > max_time:
            print(f"  [A*] TIME OUT nodes={nodes}", flush=True)
            return None
        
        f, g_cost, _, target_path = heapq.heappop(heap)
        
        # 增量回退：用 Undo 回退到分叉点
        # 找到当前路径和目标路径的共同前缀
        common_len = 0
        while common_len < min(len(current_path), len(target_path)):
            if current_path[common_len] == target_path[common_len]:
                common_len += 1
            else:
                break
        
        # 回退到分叉点
        while len(current_path) > common_len:
            undo_action(g)
            current_path.pop()
        
        # 前进到目标路径
        for i in range(common_len, len(target_path)):
            apply_action(g, target_path[i])
            current_path.append(target_path[i])
        
        # 检查是否过关
        if is_win(g, level):
            print(f"  [A*] FOUND in {len(current_path)} steps, {nodes} nodes", flush=True)
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
            
            if is_win(g, level):
                print(f"  [A*] FOUND in {len(current_path)} steps, {nodes} nodes", flush=True)
                return current_path.copy()
            
            new_h = get_heuristic(g)
            new_g = g_cost + 1
            new_f = new_g + new_h
            new_state_h = get_state_hash(g)
            
            if new_state_h not in visited or visited[new_state_h] > new_g:
                visited[new_state_h] = new_g
                counter += 1
                heapq.heappush(heap, (new_f, new_g, counter, current_path.copy()))
            
            # 撤销动作
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
    
    print("=== SK48 高效增量 A* 求解器 ===")
    print("优化：使用 Undo 逐步回退，避免完全重置")
    
    g = env._game
    num_levels = len(g._levels)
    print(f"总关卡数: {num_levels}")
    
    total_actions = 0
    all_ok = True
    
    for lvl in range(num_levels):
        t0 = time.time()
        
        sol = solve_level(g, lvl, max_time=300)
        
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
