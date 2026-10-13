# -*- coding: utf-8 -*-
"""re86 求解器 - 使用公共接口"""
import sys, os, time
sys.path.insert(0, 'D:/Quan/Lingjing-Solo-main-bench')
os.environ.setdefault('LINGJING_SRC', 'D:/Quan/Lingjing-Solo-main-bench/lingjing_solo')
from arc_game_api import GameAPI
from arcengine import GameAction, ActionInput, GameState
import numpy as np
import heapq

def solve_level(api, level, max_time=120):
    """A*求解单关"""
    t0 = time.time()
    api.goto_level(level)
    
    def state_hash():
        state = []
        for s in api.game.current_level.get_sprites():
            if s.is_visible:
                state.append((s.name, s.x, s.y))
        state.append(api.game._current_level_index)
        return hash(tuple(state))
    
    def heuristic():
        if api.game._state == GameState.WIN:
            return 0
        targets = api.get_targets()
        blocks = api.get_blocks()
        if not targets or not blocks:
            return 100
        # 计算块到目标中心的距离
        total = 0
        for b in blocks:
            cx, cy = b['x'] + b.get('width', 0)//2, b['y'] + b.get('height', 0)//2
            total += abs(cx - 32) + abs(cy - 32)
        return total // 3
    
    def is_win():
        return api.game._state == GameState.WIN or int(api.game._current_level_index) > level
    
    def get_actions():
        return api.get_available_actions()
    
    def apply(action_id):
        api.game.perform_action(
            ActionInput(id=GameAction.from_id(action_id), data={}, reasoning=None), raw=True)
    
    counter = 0
    h0 = heuristic()
    heap = [(h0, 0, counter, [])]
    visited = {state_hash(): 0}
    nodes = 0
    
    while heap and nodes < 50000:
        if time.time() - t0 > max_time:
            print(f"  [A*] TIME OUT nodes={nodes}", flush=True)
            return None
        
        f, cost, _, actions = heapq.heappop(heap)
        
        # 恢复状态
        api.goto_level(level)
        for a in actions:
            apply(a)
        
        if is_win():
            print(f"  [A*] FOUND in {len(actions)} steps, {nodes} nodes", flush=True)
            return actions
        
        nodes += 1
        if nodes % 2000 == 0:
            print(f"  [A*] nodes={nodes} cost={cost} h={heuristic()} heap={len(heap)}", flush=True)
        
        sh = state_hash()
        if sh in visited and visited[sh] < len(actions):
            continue
        visited[sh] = len(actions)
        
        for action_id in get_actions():
            apply(action_id)
            
            if is_win():
                print(f"  [A*] FOUND in {len(actions)+1} steps, {nodes} nodes", flush=True)
                return actions + [action_id]
            
            new_h = heuristic()
            new_g = len(actions) + 1
            new_f = new_g + new_h
            new_sh = state_hash()
            
            if new_sh not in visited or visited[new_sh] > new_g:
                visited[new_sh] = new_g
                counter += 1
                heapq.heappush(heap, (new_f, new_g, counter, actions + [action_id]))
            
            # 恢复
            api.goto_level(level)
            for a in actions:
                apply(a)
    
    print(f"  [A*] EXHAUSTED nodes={nodes}", flush=True)
    return None


def main():
    api = GameAPI("re86")
    print(f"re86: {api.level_count()} 关")
    
    # 先检查每关的碰撞对象
    for lvl in range(api.level_count()):
        api.goto_level(lvl)
        blocks = api.get_blocks()
        targets = api.get_targets()
        collisions = api.get_collision_objects()
        print(f"L{lvl+1}: blocks={len(blocks)}, targets={len(targets)}, collisions={len(collisions)}")
    
    # 尝试L4（有碰撞对象）
    print(f"\n=== 尝试 L4 ===")
    t0 = time.time()
    sol = solve_level(api, 3, max_time=120)
    dt = time.time() - t0
    
    if sol:
        # 验证
        api.goto_level(3)
        for a in sol:
            api.game.perform_action(
                ActionInput(id=GameAction.from_id(a), data={}, reasoning=None), raw=True)
        
        ok = api.game._state == GameState.WIN or int(api.game._current_level_index) > 3
        status = 'OK' if ok else 'FAIL'
        print(f"L4: {len(sol)} actions -> {status}  {dt:.1f}s")
    else:
        print(f"L4: NO SOLUTION  {dt:.1f}s")
    
    # 尝试L1（无碰撞对象）
    print(f"\n=== 尝试 L1 ===")
    t0 = time.time()
    sol = solve_level(api, 0, max_time=120)
    dt = time.time() - t0
    
    if sol:
        api.goto_level(0)
        for a in sol:
            api.game.perform_action(
                ActionInput(id=GameAction.from_id(a), data={}, reasoning=None), raw=True)
        
        ok = api.game._state == GameState.WIN or int(api.game._current_level_index) > 0
        status = 'OK' if ok else 'FAIL'
        print(f"L1: {len(sol)} actions -> {status}  {dt:.1f}s")
    else:
        print(f"L1: NO SOLUTION  {dt:.1f}s")


if __name__ == "__main__":
    main()
