# -*- coding: utf-8 -*-
"""SK48 优化求解器 - 基于游戏机制的启发式搜索

游戏机制：
- 延伸(Extend)：按签子方向延伸，推动前方的珠子
- 缩回(Retract)：按反方向缩回，拖动签子上的珠子
- 滑动(Slide)：按侧方向滑动，携带签子上的珠子
- 点击(Click)：切换签子（第6关开始）
- 撤销(Undo)：恢复签子和珠子位置

目标：将每个签子上的珠子排列成与参考签子相同的颜色顺序
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
    """获取状态哈希 - 记录珠子位置和签子状态"""
    state = []
    # 记录所有可见 sprite 的位置
    for s in g.current_level.get_sprites():
        if s.is_visible:
            state.append((s.name, s.x, s.y, s.rotation))
    # 记录当前选中的签子
    if hasattr(g, 'vzvypfsnt') and g.vzvypfsnt:
        state.append(('selected', g.vzvypfsnt.name))
    # 记录关卡索引
    state.append(g._current_level_index)
    # 记录能量
    state.append(getattr(g, 'qiercdohl', 0))
    return hash(tuple(state))


def get_heuristic(g):
    """启发式函数 - 基于珠子匹配进度
    
    返回未匹配的珠子数量
    """
    if g._state == GameState.WIN:
        return 0
    
    matched = 0
    total = 0
    
    # 检查路径匹配
    if hasattr(g, 'vjfbwggsd') and hasattr(g, 'xpmcmtbcv'):
        for bcwsrdcswp, pphvqzjirf in g.xpmcmtbcv.items():
            if bcwsrdcswp in g.vjfbwggsd and pphvqzjirf in g.vjfbwggsd:
                path1 = g.vjfbwggsd[bcwsrdcswp]
                path2 = g.vjfbwggsd[pphvqzjirf]
                total += max(len(path1), len(path2))
                for i in range(min(len(path1), len(path2))):
                    if path1[i].pixels[1, 1] == path2[i].pixels[1, 1]:
                        matched += 1
    
    # 启发式：未匹配的数量 + 能量消耗
    h = total - matched
    energy = getattr(g, 'qiercdohl', 196)
    # 如果能量低，增加惩罚
    if energy < 50:
        h += (50 - energy) * 2
    
    return h


def get_actions(g):
    """获取可用动作 - 智能限制动作空间"""
    actions = []
    
    # 移动动作 - 只考虑有意义的方向
    for n in [1, 2, 3, 4]:
        actions.append(('move', n))
    
    # 点击动作 - 只点击可见的 sys_click sprite
    for s in g.current_level.get_sprites():
        if s.is_visible and hasattr(s, 'tags') and 'sys_click' in s.tags:
            cx = s.x + s.width // 2
            cy = s.y + s.height // 2
            actions.append(('click', cx, cy))
    
    # 撤销动作 - 在搜索中使用
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


def solve_level_astar(g, level, max_time=300, max_nodes=100000):
    """A* 求解单关 - 优化版本"""
    t0 = time.time()
    
    # 设置关卡
    g._current_level_index = level
    g.level_reset()
    
    # 优先队列: (f, g_cost, counter, actions)
    counter = 0
    h0 = get_heuristic(g)
    heap = [(h0, 0, counter, [])]
    visited = {get_state_hash(g): 0}
    
    nodes = 0
    
    while heap and nodes < max_nodes:
        if time.time() - t0 > max_time:
            print(f"  [A*] TIME OUT nodes={nodes}", flush=True)
            return None
        
        f, g_cost, _, actions = heapq.heappop(heap)
        
        # 恢复状态
        g._current_level_index = level
        g.level_reset()
        for a in actions:
            apply_action(g, a)
        
        # 检查是否过关
        if is_win(g, level):
            print(f"  [A*] FOUND SOLUTION in {len(actions)} steps, {nodes} nodes", flush=True)
            return actions
        
        nodes += 1
        
        if nodes % 1000 == 0:
            h = get_heuristic(g)
            print(f"  [A*] nodes={nodes} g={g_cost} h={h} heap={len(heap)}", flush=True)
        
        # 生成候选动作
        for action in get_actions(g):
            # 应用动作
            apply_action(g, action)
            
            # 检查是否过关
            if is_win(g, level):
                print(f"  [A*] FOUND SOLUTION in {len(actions)+1} steps, {nodes} nodes", flush=True)
                return actions + [action]
            
            # 计算新的状态
            new_h = get_heuristic(g)
            new_g = g_cost + 1
            new_f = new_g + new_h
            new_state_h = get_state_hash(g)
            
            if new_state_h not in visited or visited[new_state_h] > new_g:
                visited[new_state_h] = new_g
                counter += 1
                heapq.heappush(heap, (new_f, new_g, counter, actions + [action]))
            
            # 恢复状态
            g._current_level_index = level
            g.level_reset()
            for a in actions:
                apply_action(g, a)
    
    print(f"  [A*] EXHAUSTED nodes={nodes}", flush=True)
    return None


def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("sk48")
    
    if env is None:
        print("无法创建 sk48 环境")
        return
    
    print("=== SK48 优化 A* 求解器 ===")
    print("游戏机制：推块游戏，需要将珠子排列成参考顺序")
    
    g = env._game
    num_levels = len(g._levels)
    print(f"总关卡数: {num_levels}")
    
    total_actions = 0
    all_ok = True
    
    for lvl in range(num_levels):
        t0 = time.time()
        
        sol = solve_level_astar(g, lvl, max_time=300, max_nodes=100000)
        
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
