# -*- coding: utf-8 -*-
"""cn04 第2关"""
import sys, os
sys.path.insert(0, 'D:/Quan/Lingjing-Solo-main-bench')
os.environ.setdefault('LINGJING_SRC', 'D:/Quan/Lingjing-Solo-main-bench/lingjing_solo')
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
import lingjing_solo.planning.arc_shadow_cn04 as shadow

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make('cn04')
g = env._game

shadow._ensure()
sg = shadow._game
sg._current_level_index = 2
sg.level_reset()
sol = shadow.solve_level(t_limit=60.0)

if sol:
    actions = []
    for a in sol:
        if a[0] == 'c': actions.append(('click', a[1], a[2]))
        elif a[0] == 'm': actions.append(('move', a[1]))
        elif a[0] == 'r': actions.append(('rotate',))
        elif a[0] == 'i': actions.append(('move', a[1]))

    # 使用 shadow._game 验证
    sg._current_level_index = 2
    sg.level_reset()
    
    for action in actions:
        if action[0] == 'click':
            sg.perform_action(__import__('arcengine').ActionInput(
                id=GameAction.ACTION6, data={'x': action[1], 'y': action[2]}, reasoning=None), raw=True)
        elif action[0] == 'move':
            sg.perform_action(__import__('arcengine').ActionInput(
                id=GameAction.from_id(action[1]), data={}, reasoning=None), raw=True)
        elif action[0] == 'rotate':
            sg.perform_action(__import__('arcengine').ActionInput(
                id=GameAction.ACTION5, data={}, reasoning=None), raw=True)
    
    ok = int(sg._current_level_index) > 2 or sg._state.name == 'WIN'
    status = 'OK' if ok else 'FAIL'
    print(f'L3: {len(actions)} actions -> {status}')
else:
    print('L2: NO SOLUTION')
