# -*- coding: utf-8 -*-
"""re86 求解 - 键盘游戏"""
import sys, os, time
sys.path.insert(0, 'D:/Quan/Lingjing-Solo-main-bench')
os.environ.setdefault('LINGJING_SRC', 'D:/Quan/Lingjing-Solo-main-bench/lingjing_solo')
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, ActionInput, GameState
import lingjing_solo.planning.arc_shadow_re86 as shadow

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make('re86')
g = env._game

shadow._game = g
shadow._env = env

num_levels = len(g._levels)
print(f're86: {num_levels} 关')
print(f'可用动作: {g._available_actions}')

total_actions = 0
all_ok = True

for lvl in range(min(3, num_levels)):
    print(f'\n--- L{lvl+1} ---')
    t0 = time.time()
    
    g._current_level_index = lvl
    g.level_reset()
    env.reset()
    
    sol = shadow.solve_level(t_limit=120.0)
    dt = time.time() - t0
    
    if sol:
        g._current_level_index = lvl
        g.level_reset()
        env.reset()
        
        for action_id in sol:
            g.perform_action(ActionInput(
                id=GameAction.from_id(action_id), data={}, reasoning=None), raw=True)
        
        ok = int(g._current_level_index) > lvl or g._state == GameState.WIN
        status = 'OK' if ok else 'FAIL'
        total_actions += len(sol)
        print(f'L{lvl+1}: {len(sol)} actions -> {status}  {dt:.1f}s')
        
        if not ok:
            all_ok = False
    else:
        print(f'L{lvl+1}: NO SOLUTION  {dt:.1f}s')
        all_ok = False

print(f'\nTotal: actions={total_actions}')
print('ALL_PASS' if all_ok else 'FAIL')
