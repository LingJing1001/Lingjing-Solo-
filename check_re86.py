# -*- coding: utf-8 -*-
"""检查re86的sprite定义和关卡"""
import sys, os
sys.path.insert(0, 'D:/Quan/Lingjing-Solo-main-bench')
os.environ.setdefault('LINGJING_SRC', 'D:/Quan/Lingjing-Solo-main-bench/lingjing_solo')
import arc_agi
from arc_agi import OperationMode
import numpy as np

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make('re86')
g = env._game

# 检查L4
g._current_level_index = 3
g.level_reset()
env.reset()

sprites = g.current_level.get_sprites()
print(f'L4: {len(sprites)} sprites')
for s in sprites:
    tags = list(s.tags) if hasattr(s, 'tags') else []
    print(f'  {s.name}: visible={s.is_visible}, pos=({s.x},{s.y}), tags={tags}')

# 搜索0007dtbisvazhv的sprite名称
print()
print('0007dtbisvazhv sprites:')
with open('D:/Quan/Lingjing-Solo-main-bench/environment_files/re86/8af5384d/re86.py') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if '0007dtbisvazhv' in line and 'tags=' in line:
        for j in range(i, max(0, i-20), -1):
            if 'name=' in lines[j]:
                name = lines[j].strip().split('name=')[1].strip('",')
                print(f'  Line {i+1}: {name}')
                break

# 检查这些sprite是否在关卡定义中使用
print()
print('检查关卡定义中的sprite:')
for i, line in enumerate(lines):
    if 'sprites[' in line and '].clone' in line:
        name = line.strip().split('sprites["')[1].split('"]')[0] if 'sprites["' in line else ''
        if name:
            # 检查这个sprite是否有0007dtbisvazhv tag
            for j, l in enumerate(lines):
                if f'name="{name}"' in l:
                    # 检查附近是否有0007dtbisvazhv
                    for k in range(j, min(len(lines), j+30)):
                        if '0007dtbisvazhv' in lines[k]:
                            print(f'  {name} (line {i+1}): HAS 0007dtbisvazhv')
                            break
                    break
