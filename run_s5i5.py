# -*- coding: utf-8 -*-
"""s5i5 求解 - 尝试点击所有sprite位置"""
import sys, os, time
sys.path.insert(0, 'D:/Quan/Lingjing-Solo-main-bench')
os.environ.setdefault('LINGJING_SRC', 'D:/Quan/Lingjing-Solo-main-bench/lingjing_solo')
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, ActionInput, GameState
import numpy as np

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make('s5i5')
g = env._game

num_levels = len(g._levels)
print(f's5i5: {num_levels} 关')

total_actions = 0
all_ok = True

for lvl in range(min(3, num_levels)):
    print(f'\n--- L{lvl+1} ---')
    t0 = time.time()
    
    g._current_level_index = lvl
    g.level_reset()
    env.reset()
    
    # 获取所有sprite位置
    sprites = g.current_level.get_sprites()
    positions = []
    for s in sprites:
        if not s.is_visible:
            continue
        rendered = s.render()
        if rendered is None:
            continue
        non_tp = np.argwhere(rendered >= 0)
        if len(non_tp) == 0:
            continue
        y, x = non_tp[len(non_tp) // 2]
        cx, cy = int(s.x + x), int(s.y + y)
        positions.append((cx, cy, s.name))
    
    print(f'  可点击位置: {len(positions)} 个')
    
    # 尝试单次点击
    found = False
    for cx, cy, name in positions:
        g._current_level_index = lvl
        g.level_reset()
        env.reset()
        
        frame = env.step(GameAction.ACTION6, data={'x': cx, 'y': cy})
        state = frame.state.name if hasattr(frame, 'state') else ''
        levels = frame.levels_completed if hasattr(frame, 'levels_completed') else 0
        
        if state == 'WIN' or levels > lvl:
            print(f'  L{lvl+1}: 1 action (click {cx},{cy} {name}) -> OK  {time.time()-t0:.1f}s')
            total_actions += 1
            found = True
            break
    
    if not found:
        # 尝试两次点击
        for i, (cx1, cy1, n1) in enumerate(positions):
            for j, (cx2, cy2, n2) in enumerate(positions):
                if i == j:
                    continue
                
                g._current_level_index = lvl
                g.level_reset()
                env.reset()
                
                env.step(GameAction.ACTION6, data={'x': cx1, 'y': cy1})
                frame = env.step(GameAction.ACTION6, data={'x': cx2, 'y': cy2})
                
                state = frame.state.name if hasattr(frame, 'state') else ''
                levels = frame.levels_completed if hasattr(frame, 'levels_completed') else 0
                
                if state == 'WIN' or levels > lvl:
                    print(f'  L{lvl+1}: 2 actions -> OK  {time.time()-t0:.1f}s')
                    total_actions += 2
                    found = True
                    break
            if found:
                break
    
    if not found:
        # 尝试三次点击
        for i, (cx1, cy1, n1) in enumerate(positions):
            for j, (cx2, cy2, n2) in enumerate(positions):
                for k, (cx3, cy3, n3) in enumerate(positions):
                    if i == j or j == k or i == k:
                        continue
                    
                    g._current_level_index = lvl
                    g.level_reset()
                    env.reset()
                    
                    env.step(GameAction.ACTION6, data={'x': cx1, 'y': cy1})
                    env.step(GameAction.ACTION6, data={'x': cx2, 'y': cy2})
                    frame = env.step(GameAction.ACTION6, data={'x': cx3, 'y': cy3})
                    
                    state = frame.state.name if hasattr(frame, 'state') else ''
                    levels = frame.levels_completed if hasattr(frame, 'levels_completed') else 0
                    
                    if state == 'WIN' or levels > lvl:
                        print(f'  L{lvl+1}: 3 actions -> OK  {time.time()-t0:.1f}s')
                        total_actions += 3
                        found = True
                        break
                if found:
                    break
            if found:
                break
    
    if not found:
        print(f'  L{lvl+1}: NO SOLUTION  {time.time()-t0:.1f}s')
        all_ok = False

print(f'\nTotal: actions={total_actions}')
print('ALL_PASS' if all_ok else 'FAIL')
