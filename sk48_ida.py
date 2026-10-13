# -*- coding: utf-8 -*-
"""SK48 IDA* 求解器 - 迭代加深 A* 搜索"""
import sys
import os
import time
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


def get_heuristic(g):
    """启发式函数 - 评估状态的好坏"""
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
    
    return total - matched


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


def is_win(g, level):
    """检查是否过关"""
    return g._state == GameState.WIN or int(g._current_level_index) > level


def save_state(g):
    """保存状态"""
    state = []
    for s in g.current_level.get_sprites():
        if s.is_visible:
            state.append((s.name, s.x, s.y, s.rotation, s.is_visible))
    state.append(g._current_level_index)
    state.append(getattr(g, 'qiercdohl', 0))
    return state


def restore_state(g, level, state):
    """恢复状态"""
    g._current_level_index = level
    g.level_reset()
    # 状态恢复通过重置实现，因为游戏引擎不支持部分恢复


def ida_search(g, level, path, g_cost, bound, t0, max_time):
    """IDA* 搜索"""
    h = get_heuristic(g)
    f = g_cost + h
    
    if f > bound:
        return f, None
    
    if is_win(g, level):
        return -1, path
    
    if time.time() - t0 > max_time:
        return float('inf'), None
    
    min_cost = float('inf')
    
    for action in get_actions(g):
        # 保存状态
        saved_level = g._current_level_index
        
        # 应用动作
        apply_action(g, action)
        
        # 递归搜索
        t, result = ida_search(g, level, path + [action], g_cost + 1, bound, t0, max_time)
        
        if t == -1:
            return -1, result
        
        if t < min_cost:
            min_cost = t
        
        # 恢复状态
        g._current_level_index = level
        g.level_reset()
        for a in path:
            apply_action(g, a)
    
    return min_cost, None


def solve_level_ida(g, level, max_time=300, max_depth=100):
    """IDA* 求解单关"""
    t0 = time.time()
    
    g._current_level_index = level
    g.level_reset()
    
    bound = get_heuristic(g)
    path = []
    
    while bound < max_depth:
        if time.time() - t0 > max_time:
            print(f"  [IDA*] TIME OUT bound={bound}")
            return None
        
        print(f"  [IDA*] Trying bound={bound}")
        
        t, result = ida_search(g, level, [], 0, bound, t0, max_time)
        
        if t == -1:
            print(f"  [IDA*] FOUND SOLUTION in {len(result)} steps")
            return result
        
        if t == float('inf'):
            print(f"  [IDA*] TIME OUT")
            return None
        
        bound = t
        print(f"  [IDA*] New bound={bound}")
    
    return None


def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("sk48")
    
    if env is None:
        print("无法创建 sk48 环境")
        return
    
    print("=== SK48 IDA* 求解器 ===")
    
    g = env._game
    num_levels = len(g._levels)
    print(f"总关卡数: {num_levels}")
    
    total_actions = 0
    all_ok = True
    
    for lvl in range(num_levels):
        t0 = time.time()
        
        sol = solve_level_ida(g, lvl, max_time=300, max_depth=100)
        
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
