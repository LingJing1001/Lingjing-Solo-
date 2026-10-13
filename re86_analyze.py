# -*- coding: utf-8 -*-
"""re86 精确分析"""
import sys, os
sys.path.insert(0, 'D:/Quan/Lingjing-Solo-main-bench')
os.environ.setdefault('LINGJING_SRC', 'D:/Quan/Lingjing-Solo-main-bench/lingjing_solo')
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, ActionInput, GameState
import numpy as np

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make('re86')
g = env._game

g._current_level_index = 0
g.level_reset()
env.reset()

# 获取目标
targets = g.current_level.get_sprites_by_tag('0054xnsuqceejm')
target = targets[0]
tp = target.pixels

# 打印颜色11和9的精确位置
print('颜色11位置:')
pos11 = np.argwhere(tp == 11)
for p in pos11:
    print(f'  ({p[1]},{p[0]})')

print('颜色9位置:')
pos9 = np.argwhere(tp == 9)
for p in pos9:
    print(f'  ({p[1]},{p[0]})')

# 获取块
blocks = g.current_level.get_sprites_by_tag('0031cppcuvqlbi')
for b in blocks:
    bp = b.pixels
    print(f'块{b.name}: pos=({b.x},{b.y})')
    
    for color in [9, 11]:
        pos = np.argwhere(bp == color)
        if len(pos) > 0:
            print(f'  颜色{color}在块中:')
            for p in pos[:5]:
                print(f'    块内({p[1]},{p[0]}) -> 全局({b.x+p[1]},{b.y+p[0]})')

# 计算需要的位置
print()
print('计算:')
# 块1(颜色11)需要使全局位置对齐目标
# 目标颜色11在(6,3)等位置
# 块1颜色11在块内(0,0)等位置
# 块1需要放在(6,3)
# 当前(10,16), 需要移动(-4,-13)
# -4和-13不是3的倍数

# 检查是否有其他sprite
print('所有sprite:')
for s in g.current_level.get_sprites():
    tags = list(s.tags) if hasattr(s, 'tags') else []
    print(f'  {s.name}: pos=({s.x},{s.y}), visible={s.is_visible}, tags={tags}')

# 尝试更多组合 - 包括ACTION5切换
print()
print('尝试更多组合:')
import time
t0 = time.time()
found = False

# 尝试切换不同次数 + 移动
for switch in range(3):
    for d1 in [1, 2, 3, 4]:
        for s1 in range(20):
            for d2 in [1, 2, 3, 4]:
                for s2 in range(20):
                    if time.time() - t0 > 60:
                        break
                    
                    g._current_level_index = 0
                    g.level_reset()
                    env.reset()
                    
                    # 切换
                    for _ in range(switch):
                        g.perform_action(ActionInput(id=GameAction.ACTION5, data={}, reasoning=None), raw=True)
                    
                    # 移动方向1
                    for _ in range(s1):
                        g.perform_action(ActionInput(id=GameAction.from_id(d1), data={}, reasoning=None), raw=True)
                    
                    # 移动方向2
                    for _ in range(s2):
                        g.perform_action(ActionInput(id=GameAction.from_id(d2), data={}, reasoning=None), raw=True)
                    
                    if g._state == GameState.WIN or int(g._current_level_index) > 0:
                        actions = [5]*switch + [d1]*s1 + [d2]*s2
                        print(f'FOUND: switch={switch} {d1}x{s1} {d2}x{s2} = {len(actions)} steps')
                        print(f'Actions: {actions}')
                        found = True
                        break
                if found: break
            if found: break
        if found: break
    if found: break

if not found:
    print(f'NO SOLUTION in {time.time()-t0:.1f}s')
