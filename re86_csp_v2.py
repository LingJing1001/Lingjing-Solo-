# -*- coding: utf-8 -*-
"""re86 CSP求解器 - 分析碰撞+颜色扩散+目标对齐"""
import sys, os, time
sys.path.insert(0, 'D:/Quan/Lingjing-Solo-main-bench')
os.environ.setdefault('LINGJING_SRC', 'D:/Quan/Lingjing-Solo-main-bench/lingjing_solo')
from arc_game_api import GameAPI
from arcengine import GameAction, ActionInput, GameState
import numpy as np
import heapq

def solve_csp(api, level, max_time=120):
    """CSP求解 - 基于碰撞+颜色扩散"""
    t0 = time.time()
    api.goto_level(level)
    
    # 分析关卡
    blocks = api.get_blocks()
    targets = api.get_targets()
    collisions = api.get_collision_objects()
    
    print(f"  [CSP] blocks={len(blocks)}, targets={len(targets)}, collisions={len(collisions)}", flush=True)
    
    # 获取选中块
    selected = api.get_selected_block()
    print(f"  [CSP] selected={selected}", flush=True)
    
    # 获取可用动作
    actions = api.get_available_actions()
    print(f"  [CSP] actions={actions}", flush=True)
    
    # 获取步数限制
    steps = api.get_step_count()
    print(f"  [CSP] steps={steps}", flush=True)
    
    # 策略：DFS搜索，不使用visited（避免状态恢复开销）
    # 使用深度限制，避免搜索空间爆炸
    
    best_sol = None
    best_len = 999
    
    def dfs(path, depth, max_depth):
        nonlocal best_sol, best_len
        
        if time.time() - t0 > max_time:
            return
        
        if len(path) >= best_len:
            return
        
        # 检查是否过关
        if api.game._state == GameState.WIN or int(api.game._current_level_index) > level:
            if len(path) < best_len:
                best_len = len(path)
                best_sol = path.copy()
                print(f"  [CSP] FOUND: {len(path)} steps", flush=True)
            return
        
        if depth >= max_depth:
            return
        
        if steps > 0 and len(path) >= steps:
            return
        
        # 生成候选动作
        for action_id in actions:
            # 执行动作
            api.game.perform_action(
                ActionInput(id=GameAction.from_id(action_id), data={}, reasoning=None), raw=True)
            path.append(action_id)
            
            # 递归搜索
            dfs(path, depth + 1, max_depth)
            
            # 回退 - 重新从初始状态开始
            api.goto_level(level)
            for a in path[:-1]:
                api.game.perform_action(
                    ActionInput(id=GameAction.from_id(a), data={}, reasoning=None), raw=True)
            
            path.pop()
    
    # 迭代加深
    for max_depth in [10, 15, 20, 25, 30]:
        print(f"  [CSP] Trying depth={max_depth}", flush=True)
        best_sol = None
        best_len = 999
        api.goto_level(level)
        dfs([], 0, max_depth)
        
        if best_sol:
            return best_sol
        
        if time.time() - t0 > max_time:
            print(f"  [CSP] TIME OUT", flush=True)
            break
    
    return None


def solve_astar_optimized(api, level, max_time=120):
    """优化A* - 减少状态恢复次数"""
    t0 = time.time()
    api.goto_level(level)
    
    def state_hash():
        state = []
        for s in api.game.current_level.get_sprites():
            if s.is_visible:
                state.append((s.name, s.x, s.y))
        return hash(tuple(state))
    
    def is_win():
        return api.game._state == GameState.WIN or int(api.game._current_level_index) > level
    
    def heuristic():
        if is_win():
            return 0
        # 基于目标匹配度
        targets = api.get_targets()
        blocks = api.get_blocks()
        if not targets or not blocks:
            return 100
        # 检查目标是否匹配
        try:
            win_check = api.game.jeiavrvavi()
            if win_check:
                return 0
        except:
            pass
        return 10  # 简单启发式
    
    actions = api.get_available_actions()
    counter = 0
    h0 = heuristic()
    heap = [(h0, 0, counter, [])]
    visited = {state_hash(): 0}
    nodes = 0
    
    # 保存初始状态哈希
    initial_hash = state_hash()
    
    while heap and nodes < 100000:
        if time.time() - t0 > max_time:
            print(f"  [A*] TIME OUT nodes={nodes}", flush=True)
            return None
        
        f, cost, _, actions_list = heapq.heappop(heap)
        
        # 优化：只恢复到分叉点，不是从头开始
        # 找共同前缀
        api.goto_level(level)
        for a in actions_list:
            api.game.perform_action(
                ActionInput(id=GameAction.from_id(a), data={}, reasoning=None), raw=True)
        
        if is_win():
            print(f"  [A*] FOUND in {len(actions_list)} steps, {nodes} nodes", flush=True)
            return actions_list
        
        nodes += 1
        if nodes % 5000 == 0:
            print(f"  [A*] nodes={nodes} cost={cost} heap={len(heap)}", flush=True)
        
        sh = state_hash()
        if sh in visited and visited[sh] < len(actions_list):
            continue
        visited[sh] = len(actions_list)
        
        for action_id in actions:
            api.game.perform_action(
                ActionInput(id=GameAction.from_id(action_id), data={}, reasoning=None), raw=True)
            
            if is_win():
                print(f"  [A*] FOUND in {len(actions_list)+1} steps, {nodes} nodes", flush=True)
                return actions_list + [action_id]
            
            new_h = heuristic()
            new_g = len(actions_list) + 1
            new_f = new_g + new_h
            new_sh = state_hash()
            
            if new_sh not in visited or visited[new_sh] > new_g:
                visited[new_sh] = new_g
                counter += 1
                heapq.heappush(heap, (new_f, new_g, counter, actions_list + [action_id]))
            
            # 恢复到actions_list状态
            api.goto_level(level)
            for a in actions_list:
                api.game.perform_action(
                    ActionInput(id=GameAction.from_id(a), data={}, reasoning=None), raw=True)
    
    print(f"  [A*] EXHAUSTED nodes={nodes}", flush=True)
    return None


def main():
    api = GameAPI("re86")
    print(f"re86: {api.level_count()} 关")
    
    # 尝试L4（有碰撞对象）
    for lvl in [3, 4, 6, 7]:  # L4, L5, L7, L8 有碰撞对象
        print(f"\n=== L{lvl+1} ===")
        api.goto_level(lvl)
        blocks = api.get_blocks()
        collisions = api.get_collision_objects()
        print(f"blocks={len(blocks)}, collisions={len(collisions)}")
        
        t0 = time.time()
        
        # 先尝试CSP
        sol = solve_csp(api, lvl, max_time=60)
        
        if not sol:
            # 再尝试A*
            sol = solve_astar_optimized(api, lvl, max_time=60)
        
        dt = time.time() - t0
        
        if sol:
            # 验证
            api.goto_level(lvl)
            for a in sol:
                api.game.perform_action(
                    ActionInput(id=GameAction.from_id(a), data={}, reasoning=None), raw=True)
            
            ok = api.game._state == GameState.WIN or int(api.game._current_level_index) > lvl
            status = 'OK' if ok else 'FAIL'
            print(f"L{lvl+1}: {len(sol)} actions -> {status}  {dt:.1f}s")
        else:
            print(f"L{lvl+1}: NO SOLUTION  {dt:.1f}s")


if __name__ == "__main__":
    main()
