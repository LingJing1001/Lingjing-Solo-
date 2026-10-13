# -*- coding: utf-8 -*-
"""CN04 精确求解器 - 计算最优移动"""
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
    """获取所有特殊像素（考虑旋转）"""
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


def solve_level_optimal(env, level, max_time=60):
    """最优求解 - 计算精确移动"""
    t0 = time.time()
    actions = []
    
    # 重置
    frame = env.reset()
    g = env._game
    
    # 获取 sprite
    sprites = list(g.current_level.get_sprites())
    
    print(f"  [Optimal] Level {level+1}: {len(sprites)} sprites")
    
    # 对于每个 sprite，找到它的特殊像素
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
                        'type': '8' if orig[y, x] == 8 else '13',
                        'local': (x, y)
                    })
        
        sprite_info[s.name] = {
            'sprite': s,
            'click_pos': find_click_pos(s),
            'specials': specials,
            'x': s.x,
            'y': s.y,
            'rotation': s.rotation
        }
    
    print(f"  [Optimal] Sprites with specials: {len(sprite_info)}")
    
    # 策略：对于每对 sprite，尝试让它们的特殊像素配对
    sprite_names = list(sprite_info.keys())
    
    for i, name1 in enumerate(sprite_names):
        for j, name2 in enumerate(sprite_names):
            if i >= j:
                continue
            
            if time.time() - t0 > max_time:
                break
            
            info1 = sprite_info[name1]
            info2 = sprite_info[name2]
            
            # 找到所有可能的配对
            for sp1 in info1['specials']:
                for sp2 in info2['specials']:
                    # 8 需要和 13 配对
                    if sp1['type'] == sp2['type']:
                        continue
                    
                    # 计算需要的移动
                    # 如果移动 sprite1，让 sp1 移动到 sp2 的位置
                    dx = sp2['global'][0] - sp1['global'][0]
                    dy = sp2['global'][1] - sp1['global'][1]
                    
                    print(f"  [Optimal] Try: {name1}({sp1['type']}) -> {name2}({sp2['type']}), offset=({dx},{dy})")
                    
                    # 尝试不同旋转
                    for rot in range(4):
                        # 重置
                        env.reset()
                        
                        # 点击 sprite1
                        if info1['click_pos']:
                            a = ('click', info1['click_pos'][0], info1['click_pos'][1])
                            actions = [a]
                            apply_action(env, a)
                            
                            # 旋转
                            for _ in range(rot):
                                a = ('rotate',)
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
                            if matched > 0:
                                print(f"  [Optimal] Found {matched} matches!")
                                return actions
    
    return None


def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("cn04")
    
    if env is None:
        print("无法创建 cn04 环境")
        return
    
    print("开始精确求解 cn04...")
    
    total_actions = 0
    all_ok = True
    
    for lvl in range(8):
        t0 = time.time()
        
        sol = solve_level_optimal(env, lvl, max_time=30)
        
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
