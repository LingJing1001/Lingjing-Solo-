# -*- coding: utf-8 -*-
"""re86 快照+碰撞路径求解器

1. 快照机制 - 保存/恢复游戏状态，避免重放
2. 碰撞路径分析 - 直接计算块到碰撞对象的路径
"""
import sys, os, time, copy
sys.path.insert(0, 'D:/Quan/Lingjing-Solo-main-bench')
os.environ.setdefault('LINGJING_SRC', 'D:/Quan/Lingjing-Solo-main-bench/lingjing_solo')
from arc_game_api import GameAPI
from arcengine import GameAction, ActionInput, GameState
import numpy as np
import heapq


class Snapshot:
    """游戏状态快照"""
    def __init__(self, api):
        self.level = api.game._current_level_index
        self.sprites = []
        for s in api.game.current_level.get_sprites():
            self.sprites.append({
                'name': s.name,
                'x': s.x,
                'y': s.y,
                'rotation': s.rotation,
                'visible': s.is_visible,
                'pixels': s.pixels.copy() if hasattr(s, 'pixels') else None,
            })
        self.state = api.game._state
        self.steps = api.get_step_count()
    
    def restore(self, api):
        """恢复快照"""
        api.goto_level(self.level)
        # 恢复sprite状态
        current_sprites = {s.name: s for s in api.game.current_level.get_sprites()}
        for snap in self.sprites:
            if snap['name'] in current_sprites:
                s = current_sprites[snap['name']]
                s.set_position(snap['x'], snap['y'])
                s.set_rotation(snap['rotation'])
                if snap['pixels'] is not None and hasattr(s, 'pixels'):
                    s.pixels = snap['pixels'].copy()
                if snap['visible'] != s.is_visible:
                    s.set_visible(snap['visible'])


def solve_with_snapshot(api, level, max_time=120):
    """使用快照的A*求解"""
    t0 = time.time()
    api.goto_level(level)
    
    def state_hash():
        state = []
        for s in api.game.current_level.get_sprites():
            if s.is_visible:
                state.append((s.name, s.x, s.y, s.rotation))
        return hash(tuple(state))
    
    def is_win():
        return api.game._state == GameState.WIN or int(api.game._current_level_index) > level
    
    def heuristic():
        if is_win():
            return 0
        try:
            if api.game.jeiavrvavi():
                return 0
        except:
            pass
        return 10
    
    actions = api.get_available_actions()
    
    # 初始快照
    init_snap = Snapshot(api)
    
    counter = 0
    h0 = heuristic()
    heap = [(h0, 0, counter, [], init_snap)]
    visited = {state_hash(): 0}
    nodes = 0
    
    while heap and nodes < 100000:
        if time.time() - t0 > max_time:
            print(f"  [A*] TIME OUT nodes={nodes}", flush=True)
            return None
        
        f, cost, _, actions_list, snap = heapq.heappop(heap)
        
        # 用快照恢复状态
        snap.restore(api)
        
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
                new_snap = Snapshot(api)
                heapq.heappush(heap, (new_f, new_g, counter, actions_list + [action_id], new_snap))
            
            # 恢复到当前节点状态
            snap.restore(api)
    
    print(f"  [A*] EXHAUSTED nodes={nodes}", flush=True)
    return None


def solve_collision_path(api, level, max_time=120):
    """碰撞路径分析 - 直接计算块到碰撞对象的路径"""
    t0 = time.time()
    api.goto_level(level)
    
    blocks = api.get_blocks()
    targets = api.get_targets()
    collisions = api.get_collision_objects()
    selected = api.get_selected_block()
    
    print(f"  [路径] blocks={len(blocks)}, collisions={len(collisions)}", flush=True)
    print(f"  [路径] selected={selected}", flush=True)
    
    if not selected or not collisions:
        return None
    
    # 获取选中块的位置
    sel_x, sel_y = selected['x'], selected['y']
    
    # 计算到每个碰撞对象的路径
    STEP = 3  # 每次移动3格
    
    for col in collisions:
        col_x, col_y = col['x'], col['y']
        
        # 计算需要移动的步数
        dx = col_x - sel_x
        dy = col_y - sel_y
        
        # 计算移动步数（每次3格）
        steps_x = dx // STEP if dx % STEP == 0 else None
        steps_y = dy // STEP if dy % STEP == 0 else None
        
        if steps_x is not None and steps_y is not None:
            # 可以精确到达碰撞对象
            print(f"  [路径] 到{col['name']}: dx={dx}({steps_x}步), dy={dy}({steps_y}步)", flush=True)
            
            # 生成动作序列
            actions = []
            
            # 水平移动
            for _ in range(abs(steps_x)):
                actions.append(4 if steps_x > 0 else 3)
            
            # 垂直移动
            for _ in range(abs(steps_y)):
                actions.append(2 if steps_y > 0 else 1)
            
            # 验证
            api.goto_level(level)
            for a in actions:
                api.game.perform_action(
                    ActionInput(id=GameAction.from_id(a), data={}, reasoning=None), raw=True)
            
            if api.game._state == GameState.WIN or int(api.game._current_level_index) > level:
                print(f"  [路径] FOUND: {len(actions)} steps", flush=True)
                return actions
            
            # 碰撞后继续移动
            # 尝试碰撞后向不同方向移动
            for d in [1, 2, 3, 4]:
                for s in range(1, 10):
                    api.goto_level(level)
                    for a in actions:
                        api.game.perform_action(
                            ActionInput(id=GameAction.from_id(a), data={}, reasoning=None), raw=True)
                    
                    for _ in range(s):
                        api.game.perform_action(
                            ActionInput(id=GameAction.from_id(d), data={}, reasoning=None), raw=True)
                    
                    if api.game._state == GameState.WIN or int(api.game._current_level_index) > level:
                        full = actions + [d] * s
                        print(f"  [路径] FOUND: {len(full)} steps (碰撞+移动)", flush=True)
                        return full
    
    # 尝试切换到其他块
    for switch in range(1, len(blocks)):
        for col in collisions:
            col_x, col_y = col['x'], col['y']
            
            api.goto_level(level)
            
            # 切换
            for _ in range(switch):
                api.game.perform_action(
                    ActionInput(id=GameAction.ACTION5, data={}, reasoning=None), raw=True)
            
            # 获取新选中块
            new_sel = api.get_selected_block()
            if not new_sel:
                continue
            
            dx = col_x - new_sel['x']
            dy = col_y - new_sel['y']
            
            steps_x = dx // STEP if dx % STEP == 0 else None
            steps_y = dy // STEP if dy % STEP == 0 else None
            
            if steps_x is not None and steps_y is not None:
                print(f"  [路径] 切换{switch}后到{col['name']}: dx={dx}({steps_x}步), dy={dy}({steps_y}步)", flush=True)
                
                actions = [5] * switch
                for _ in range(abs(steps_x)):
                    actions.append(4 if steps_x > 0 else 3)
                for _ in range(abs(steps_y)):
                    actions.append(2 if steps_y > 0 else 1)
                
                # 验证
                api.goto_level(level)
                for a in actions:
                    api.game.perform_action(
                        ActionInput(id=GameAction.from_id(a), data={}, reasoning=None), raw=True)
                
                if api.game._state == GameState.WIN or int(api.game._current_level_index) > level:
                    print(f"  [路径] FOUND: {len(actions)} steps", flush=True)
                    return actions
                
                # 碰撞后继续移动
                for d in [1, 2, 3, 4]:
                    for s in range(1, 10):
                        api.goto_level(level)
                        for a in actions:
                            api.game.perform_action(
                                ActionInput(id=GameAction.from_id(a), data={}, reasoning=None), raw=True)
                        
                        for _ in range(s):
                            api.game.perform_action(
                                ActionInput(id=GameAction.from_id(d), data={}, reasoning=None), raw=True)
                        
                        if api.game._state == GameState.WIN or int(api.game._current_level_index) > level:
                            full = actions + [d] * s
                            print(f"  [路径] FOUND: {len(full)} steps", flush=True)
                            return full
    
    print(f"  [路径] NO SOLUTION", flush=True)
    return None


def main():
    api = GameAPI("re86")
    print(f"re86: {api.level_count()} 关")
    
    # 尝试有碰撞对象的关卡
    for lvl in [3, 4, 6, 7]:  # L4, L5, L7, L8
        print(f"\n=== L{lvl+1} ===")
        api.goto_level(lvl)
        collisions = api.get_collision_objects()
        print(f"碰撞对象: {len(collisions)}")
        
        t0 = time.time()
        
        # 先尝试碰撞路径分析
        sol = solve_collision_path(api, lvl, max_time=60)
        
        if not sol:
            # 再尝试快照A*
            sol = solve_with_snapshot(api, lvl, max_time=60)
        
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
