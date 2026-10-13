# -*- coding: utf-8 -*-
"""CN04 智能求解器 - 基于相对方向分析"""
import sys
import os
import time
import math
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
KAGGLE = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"
sys.path.insert(0, str(KAGGLE))
os.environ.setdefault("LINGJING_SRC", str(ROOT / "lingjing_solo"))
sys.path.insert(0, str(ROOT))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction


def get_special_pixels(game, sprites):
    """获取所有特殊像素"""
    specials = []
    for s in sprites:
        if not s.is_visible:
            continue
        orig = game.hlxyvcmpk.get(s.name)
        if orig is None:
            continue
        h, w = orig.shape
        for y in range(h):
            for x in range(w):
                if orig[y, x] in (8, 13):
                    rot = s.rotation
                    if rot == 0:
                        rx, ry = x, y
                    elif rot == 90:
                        rx, ry = h - 1 - y, x
                    elif rot == 180:
                        rx, ry = w - 1 - x, h - 1 - y
                    elif rot == 270:
                        rx, ry = y, w - 1 - x
                    else:
                        rx, ry = x, y
                    
                    gx, gy = int(s.x + rx), int(s.y + ry)
                    specials.append({
                        'sprite': s.name,
                        'pos': (gx, gy),
                        'type': '8' if orig[y, x] == 8 else '13',
                        'local': (x, y)
                    })
    return specials


def count_matched(specials):
    """计算匹配的对数"""
    from collections import Counter
    pos_count = Counter([s['pos'] for s in specials])
    return sum(1 for s in specials if pos_count[s['pos']] >= 2) // 2


def apply_action(env, action):
    """应用动作"""
    if action[0] == 'click':
        return env.step(GameAction.ACTION6, data={'x': action[1], 'y': action[2]})
    elif action[0] == 'move':
        return env.step(GameAction.from_id(action[1]), data={})
    elif action[0] == 'rotate':
        return env.step(GameAction.ACTION5, data={})
    return None


def find_click_pos(sprite):
    """找 sprite 的点击位置"""
    rendered = sprite.render()
    if rendered is None:
        return None
    non_tp = np.argwhere(rendered >= 0)
    if len(non_tp) == 0:
        return None
    y, x = non_tp[len(non_tp) // 2]
    return (int(sprite.x + x), int(sprite.y + y))


def get_relative_direction(specials):
    """获取特殊像素的相对方向向量"""
    if len(specials) < 2:
        return None
    
    # 找到8和13
    sp8 = None
    sp13 = None
    for s in specials:
        if s['type'] == '8':
            sp8 = s
        elif s['type'] == '13':
            sp13 = s
    
    if not sp8 or not sp13:
        return None
    
    # 支持 'pos' 或 'global' 键
    pos8 = sp8.get('pos') or sp8.get('global')
    pos13 = sp13.get('pos') or sp13.get('global')
    
    if not pos8 or not pos13:
        return None
    
    # 返回从13到8的向量
    return (pos8[0] - pos13[0], pos8[1] - pos13[1])


def solve_level_smart(env, level, max_time=60):
    """智能求解 - 基于相对方向分析"""
    t0 = time.time()
    actions = []
    
    # 重置
    frame = env.reset()
    g = env._game
    
    # 获取 sprite
    sprites = list(g.current_level.get_sprites())
    
    print(f"  [Smart] Level {level+1}: {len(sprites)} sprites")
    
    # 获取每个 sprite 的特殊像素和相对方向
    sprite_info = {}
    for s in sprites:
        if not s.is_visible:
            continue
        
        orig = g.hlxyvcmpk.get(s.name)
        if orig is None:
            continue
        
        h, w = orig.shape
        specials = []
        for y in range(h):
            for x in range(w):
                if orig[y, x] in (8, 13):
                    rot = s.rotation
                    if rot == 0:
                        rx, ry = x, y
                    elif rot == 90:
                        rx, ry = h - 1 - y, x
                    elif rot == 180:
                        rx, ry = w - 1 - x, h - 1 - y
                    elif rot == 270:
                        rx, ry = y, w - 1 - x
                    else:
                        rx, ry = x, y
                    
                    gx, gy = int(s.x + rx), int(s.y + ry)
                    specials.append({
                        'global': (gx, gy),
                        'type': '8' if orig[y, x] == 8 else '13'
                    })
        
        if len(specials) >= 2:
            rel_dir = get_relative_direction(specials)
            sprite_info[s.name] = {
                'sprite': s,
                'click_pos': find_click_pos(s),
                'specials': specials,
                'rel_dir': rel_dir,
                'x': s.x,
                'y': s.y,
                'rotation': s.rotation
            }
    
    print(f"  [Smart] Sprites with specials: {len(sprite_info)}")
    for name, info in sprite_info.items():
        print(f"    {name}: rel_dir={info['rel_dir']}, pos=({info['x']},{info['y']}), rot={info['rotation']}")
    
    # 策略：找到相对方向相同的 sprite，然后移动让它们重叠
    sprite_names = list(sprite_info.keys())
    
    for i, name1 in enumerate(sprite_names):
        for j, name2 in enumerate(sprite_names):
            if i >= j:
                continue
            
            if time.time() - t0 > max_time:
                break
            
            info1 = sprite_info[name1]
            info2 = sprite_info[name2]
            
            # 检查相对方向
            if info1['rel_dir'] is None or info2['rel_dir'] is None:
                continue
            
            print(f"  [Smart] Comparing {name1} ({info1['rel_dir']}) vs {name2} ({info2['rel_dir']})")
            
            # 如果相对方向相同，尝试移动让它们重叠
            if info1['rel_dir'] == info2['rel_dir']:
                print(f"  [Smart] Same relative direction! Try to align...")
                
                # 找到8和13的位置
                sp1_8 = next((s for s in info1['specials'] if s['type'] == '8'), None)
                sp1_13 = next((s for s in info1['specials'] if s['type'] == '13'), None)
                sp2_8 = next((s for s in info2['specials'] if s['type'] == '8'), None)
                sp2_13 = next((s for s in info2['specials'] if s['type'] == '13'), None)
                
                if not all([sp1_8, sp1_13, sp2_8, sp2_13]):
                    continue
                
                # 计算移动偏移：让 sprite1 的8和 sprite2 的8对齐
                dx = sp2_8['global'][0] - sp1_8['global'][0]
                dy = sp2_8['global'][1] - sp1_8['global'][1]
                
                print(f"  [Smart] Move offset: ({dx}, {dy})")
                
                # 重置并执行
                env.reset()
                actions = []
                
                # 点击 sprite1
                if info1['click_pos']:
                    a = ('click', info1['click_pos'][0], info1['click_pos'][1])
                    actions.append(a)
                    apply_action(env, a)
                    
                    # 移动
                    for _ in range(abs(dx)):
                        a = ('move', 4 if dx > 0 else 3)
                        actions.append(a)
                        apply_action(env, a)
                    
                    for _ in range(abs(dy)):
                        a = ('move', 2 if dy > 0 else 1)
                        actions.append(a)
                        apply_action(env, a)
                    
                    # 检查匹配
                    matched = count_matched(get_special_pixels(env._game, list(env._game.current_level.get_sprites())))
                    print(f"  [Smart] After move: {matched} matches")
                    
                    if matched >= 2:
                        print(f"  [Smart] Found solution!")
                        return actions
            
            # 如果相对方向不同，尝试旋转让它们相同
            else:
                print(f"  [Smart] Different relative direction, trying rotations...")
                
                # 找到 sprite1 需要旋转的次数
                # 相对方向向量
                dir1 = info1['rel_dir']
                dir2 = info2['rel_dir']
                
                # 计算需要旋转的角度
                # 90° 旋转会改变相对方向
                # (dx, dy) -> (-dy, dx) for 90° clockwise
                
                # 尝试不同旋转次数
                for rot_count in range(4):
                    # 重置
                    env.reset()
                    actions = []
                    
                    # 点击 sprite1
                    if info1['click_pos']:
                        a = ('click', info1['click_pos'][0], info1['click_pos'][1])
                        actions.append(a)
                        apply_action(env, a)
                        
                        # 旋转
                        for _ in range(rot_count):
                            a = ('rotate',)
                            actions.append(a)
                            apply_action(env, a)
                        
                        # 获取新的相对方向
                        new_specials = get_special_pixels(env._game, [info1['sprite']])
                        new_rel_dir = get_relative_direction(new_specials)
                        
                        print(f"    [Smart] After {rot_count} rotations: rel_dir={new_rel_dir}")
                        
                        if new_rel_dir == dir2:
                            print(f"    [Smart] Relative directions match!")
                            
                            # 现在移动让它们重叠
                            sp1_8 = next((s for s in new_specials if s['type'] == '8'), None)
                            sp1_13 = next((s for s in new_specials if s['type'] == '13'), None)
                            sp2_8 = next((s for s in info2['specials'] if s['type'] == '8'), None)
                            sp2_13 = next((s for s in info2['specials'] if s['type'] == '13'), None)
                            
                            if sp1_8 and sp2_8:
                                dx = sp2_8['global'][0] - sp1_8['global'][0]
                                dy = sp2_8['global'][1] - sp1_8['global'][1]
                                
                                for _ in range(abs(dx)):
                                    a = ('move', 4 if dx > 0 else 3)
                                    actions.append(a)
                                    apply_action(env, a)
                                
                                for _ in range(abs(dy)):
                                    a = ('move', 2 if dy > 0 else 1)
                                    actions.append(a)
                                    apply_action(env, a)
                                
                                matched = count_matched(get_special_pixels(env._game, list(env._game.current_level.get_sprites())))
                                print(f"    [Smart] After move: {matched} matches")
                                
                                if matched >= 2:
                                    return actions
    
    return None


def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("cn04")
    
    if env is None:
        print("无法创建 cn04 环境")
        return
    
    print("开始智能求解 cn04...")
    
    total_actions = 0
    all_ok = True
    
    for lvl in range(8):
        t0 = time.time()
        
        sol = solve_level_smart(env, lvl, max_time=30)
        
        dt = time.time() - t0
        
        if sol is None:
            print(f"L{lvl+1}: NO SOLUTION  {dt:.1f}s")
            all_ok = False
            break
        
        # 验证
        frame = env.reset()
        for a in sol:
            if a[0] == 'click':
                frame = env.step(GameAction.ACTION6, data={'x': a[1], 'y': a[2]})
            elif a[0] == 'move':
                frame = env.step(GameAction.from_id(a[1]), data={})
            elif a[0] == 'rotate':
                frame = env.step(GameAction.ACTION5, data={})
        
        ok = frame.levels_completed > lvl or frame.state.name == "WIN"
        total_actions += len(sol)
        
        print(f"L{lvl+1}: {len(sol)} actions -> levels={frame.levels_completed}  "
              f"{'OK' if ok else 'FAIL'}  {dt:.1f}s")
        
        if not ok:
            all_ok = False
            break
    
    print(f"\nTotal: actions={total_actions}")
    print("ALL_PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
