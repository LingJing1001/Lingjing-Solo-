# -*- coding: utf-8 -*-
"""re86 A* 求解器 - 拼图游戏

游戏机制：
- ACTION1-4: 移动选中的块
- ACTION5: 切换到下一个块
- 目标：将块移动到正确位置，使目标sprite匹配
"""
import sys, os, time, heapq
sys.path.insert(0, 'D:/Quan/Lingjing-Solo-main-bench')
os.environ.setdefault('LINGJING_SRC', 'D:/Quan/Lingjing-Solo-main-bench/lingjing_solo')
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, ActionInput, GameState
import numpy as np

def get_state_hash(g):
    """状态哈希"""
    state = []
    for s in g.current_level.get_sprites():
        if s.is_visible:
            state.append((s.name, s.x, s.y))
    state.append(g._current_level_index)
    return hash(tuple(state))

def get_heuristic(g):
    """启发式 - 基于目标匹配度"""
    if g._state == GameState.WIN:
        return 0
    
    # 获取目标sprite和可移动块
    targets = g.current_level.get_sprites_by_tag("0054xnsuqceejm")
    blocks = g.current_level.get_sprites_by_tag("0031cppcuvqlbi")
    
    if not targets or not blocks:
        return 100
    
    # 计算未匹配的目标数量
    unmatched = 0
    for target in targets:
        # 检查目标是否匹配
        pixels = target.pixels
        if np.any((pixels != -1) & (pixels != 4)):
            unmatched += 1
    
    return unmatched

def get_actions(g):
    """获取可用动作"""
    return [1, 2, 3, 4, 5]  # 移动+旋转

def apply_action(g, action_id):
    """应用动作"""
    g.perform_action(ActionInput(
        id=GameAction.from_id(action_id), data={}, reasoning=None), raw=True)

def is_win(g, level):
    """检查是否过关"""
    return g._state == GameState.WIN or int(g._current_level_index) > level

def solve_level_astar(g, level, max_time=120, max_nodes=50000):
    """A* 求解"""
    t0 = time.time()
    
    g._current_level_index = level
    g.level_reset()
    
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
        
        if is_win(g, level):
            print(f"  [A*] FOUND in {len(actions)} steps, {nodes} nodes", flush=True)
            return actions
        
        nodes += 1
        
        if nodes % 1000 == 0:
            h = get_heuristic(g)
            print(f"  [A*] nodes={nodes} g={g_cost} h={h} heap={len(heap)}", flush=True)
        
        for action_id in get_actions(g):
            apply_action(g, action_id)
            
            if is_win(g, level):
                print(f"  [A*] FOUND in {len(actions)+1} steps, {nodes} nodes", flush=True)
                return actions + [action_id]
            
            new_h = get_heuristic(g)
            new_g = g_cost + 1
            new_f = new_g + new_h
            new_state = get_state_hash(g)
            
            if new_state not in visited or visited[new_state] > new_g:
                visited[new_state] = new_g
                counter += 1
                heapq.heappush(heap, (new_f, new_g, counter, actions + [action_id]))
            
            # 恢复
            g._current_level_index = level
            g.level_reset()
            for a in actions:
                apply_action(g, a)
    
    print(f"  [A*] EXHAUSTED nodes={nodes}", flush=True)
    return None

def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("re86")
    g = env._game
    
    num_levels = len(g._levels)
    print(f"re86: {num_levels} 关")
    
    total_actions = 0
    all_ok = True
    
    for lvl in range(min(3, num_levels)):
        print(f"\n--- L{lvl+1} ---")
        t0 = time.time()
        
        sol = solve_level_astar(g, lvl, max_time=120, max_nodes=50000)
        dt = time.time() - t0
        
        if sol:
            # 验证
            g._current_level_index = lvl
            g.level_reset()
            for a in sol:
                apply_action(g, a)
            
            ok = is_win(g, lvl)
            status = 'OK' if ok else 'FAIL'
            total_actions += len(sol)
            print(f"L{lvl+1}: {len(sol)} actions -> {status}  {dt:.1f}s")
            
            if not ok:
                all_ok = False
        else:
            print(f"L{lvl+1}: NO SOLUTION  {dt:.1f}s")
            all_ok = False
    
    print(f"\nTotal: actions={total_actions}")
    print("ALL_PASS" if all_ok else "FAIL")

if __name__ == "__main__":
    main()
