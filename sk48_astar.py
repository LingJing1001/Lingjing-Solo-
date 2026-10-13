# -*- coding: utf-8 -*-
"""SK48 A* 求解器 - 基于路径匹配的启发式搜索"""
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
    """获取状态哈希"""
    state = []
    for s in g.current_level.get_sprites():
        if s.is_visible:
            state.append((s.name, s.x, s.y, s.rotation))
    state.append(g._current_level_index)
    state.append(getattr(g, 'qiercdohl', 0))
    return hash(tuple(state))


def get_heuristic(g):
    """启发式函数 - 评估状态的好坏"""
    # 检查是否过关
    if g._state == GameState.WIN:
        return 0
    
    # 获取匹配进度
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
    
    # 启发式：未匹配的数量
    return total - matched


def get_actions(g):
    """获取可用动作"""
    actions = []
    
    # 移动动作
    for n in [1, 2, 3, 4]:
        actions.append(('move', n))
    
    # 点击动作 - 点击可见的 sprite
    for s in g.current_level.get_sprites():
        if s.is_visible and hasattr(s, 'tags') and 'sys_click' in s.tags:
            # 使用 sprite 的中心位置
            cx = s.x + s.width // 2
            cy = s.y + s.height // 2
            actions.append(('click', cx, cy))
    
    # 撤销动作
    actions.append(('undo',))
    
    return actions


def apply_action(g, action):
    """应用动作"""
    if action[0] == 'move':
        g.perform_action(ActionInput(id=GameAction.from_id(action[1]), data={}, reasoning=None), raw=True)
    elif action[0] == 'click':
        g.perform_action(ActionInput(id=GameAction.ACTION6, data={'x': action[1], 'y': action[2]}, reasoning=None), raw=True)
    elif action[0] == 'undo':
        g.perform_action(ActionInput(id=GameAction.ACTION7, data={}, reasoning=None), raw=True)


def is_win(g, level=0):
    """检查是否过关"""
    return g._state == GameState.WIN or int(g._current_level_index) > level


def solve_level_astar(g, level, max_time=120, max_nodes=50000):
    """A* 求解单关"""
    t0 = time.time()
    
    # 设置关卡
    g._current_level_index = level
    g.level_reset()
    
    # 优先队列: (f, g_cost, counter, actions, state_hash)
    counter = 0
    h0 = get_heuristic(g)
    heap = [(h0, 0, counter, [], get_state_hash(g))]
    visited = {get_state_hash(g): 0}
    
    nodes = 0
    
    while heap and nodes < max_nodes:
        if time.time() - t0 > max_time:
            print(f"  [A*] TIME OUT nodes={nodes}")
            return None
        
        f, g_cost, _, actions, state_h = heapq.heappop(heap)
        
        if g_cost != visited.get(state_h):
            continue
        
        nodes += 1
        
        if nodes % 1000 == 0:
            h = get_heuristic(g)
            print(f"  [A*] nodes={nodes} g={g_cost} h={h} heap={len(heap)}")
        
        # 恢复状态
        g._current_level_index = level
        g.level_reset()
        for a in actions:
            apply_action(g, a)
        
        # 检查是否过关
        if is_win(g, level):
            print(f"  [A*] FOUND SOLUTION in {len(actions)} steps, {nodes} nodes")
            return actions
        
        # 生成候选动作
        for action in get_actions(g):
            # 保存状态
            g._current_level_index = level
            g.level_reset()
            for a in actions:
                apply_action(g, a)
            
            # 应用动作
            apply_action(g, action)
            
            # 检查是否过关
            if is_win(g, level):
                print(f"  [A*] FOUND SOLUTION in {len(actions)+1} steps, {nodes} nodes")
                return actions + [action]
            
            # 计算新的状态
            new_h = get_heuristic(g)
            new_g = g_cost + 1
            new_f = new_g + new_h
            new_state_h = get_state_hash(g)
            
            if new_state_h not in visited or visited[new_state_h] > new_g:
                visited[new_state_h] = new_g
                counter += 1
                heapq.heappush(heap, (new_f, new_g, counter, actions + [action], new_state_h))
    
    print(f"  [A*] EXHAUSTED nodes={nodes}")
    return None


def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("sk48")
    
    if env is None:
        print("无法创建 sk48 环境")
        return
    
    print("=== SK48 A* 求解器 ===")
    
    g = env._game
    num_levels = len(g._levels)
    print(f"总关卡数: {num_levels}")
    
    total_actions = 0
    all_ok = True
    
    for lvl in range(num_levels):
        t0 = time.time()
        
        sol = solve_level_astar(g, lvl, max_time=300, max_nodes=200000)
        
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
