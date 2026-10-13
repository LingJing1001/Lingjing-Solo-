# -*- coding: utf-8 -*-
"""CN04 完整求解器 - 多轮配对"""
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
                        'type': '8' if orig[y, x] == 8 else '13'
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


def get_sprite_info(env):
    """获取所有 sprite 信息"""
    g = env._game
    sprites = list(g.current_level.get_sprites())
    
    info = {}
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
        
        info[s.name] = {
            'sprite': s,
            'click_pos': find_click_pos(s),
            'specials': specials
        }
    
    return info


def solve_level_complete(env, level, max_time=60):
    """完整求解 - 多轮配对"""
    t0 = time.time()
    all_actions = []
    
    max_rounds = 10  # 最多尝试 10 轮
    
    for round_num in range(max_rounds):
        if time.time() - t0 > max_time:
            break
        
        # 检查是否完成
        matched = count_matched(get_special_pixels(env._game, list(env._game.current_level.get_sprites())))
        total = len(get_special_pixels(env._game, list(env._game.current_level.get_sprites()))) // 2
        
        if matched >= total and total > 0:
            print(f"  [Complete] Level {level+1} COMPLETED in {round_num} rounds!")
            return all_actions
        
        # 获取 sprite 信息
        sprite_info = get_sprite_info(env)
        
        # 找到未配对的 8 和 13
        specials = get_special_pixels(env._game, list(env._game.current_level.get_sprites()))
        
        # 按位置分组
        from collections import defaultdict
        pos_groups = defaultdict(list)
        for sp in specials:
            pos_groups[sp['pos']].append(sp)
        
        # 找到未配对的特殊像素
        unmatched = []
        for pos, sp_list in pos_groups.items():
            if len(sp_list) < 2:
                for sp in sp_list:
                    unmatched.append(sp)
        
        if len(unmatched) < 2:
            break
        
        # 找一个 8 和一个 13
        sp8 = None
        sp13 = None
        for sp in unmatched:
            if sp['type'] == '8' and not sp8:
                sp8 = sp
            elif sp['type'] == '13' and not sp13:
                sp13 = sp
        
        if not sp8 or not sp13:
            break
        
        # 计算移动
        dx = sp13['pos'][0] - sp8['pos'][0]
        dy = sp13['pos'][1] - sp8['pos'][1]
        
        # 找到对应的 sprite
        sprite8 = sprite_info.get(sp8['sprite'])
        sprite13 = sprite_info.get(sp13['sprite'])
        
        if not sprite8 or not sprite13:
            break
        
        # 选择移动 sprite8（让 8 移动到 13 的位置）
        if sprite8['click_pos']:
            # 点击
            a = ('click', sprite8['click_pos'][0], sprite8['click_pos'][1])
            all_actions.append(a)
            apply_action(env, a)
            
            # 移动
            for _ in range(abs(dx)):
                a = ('move', 4 if dx > 0 else 3)
                all_actions.append(a)
                apply_action(env, a)
            
            for _ in range(abs(dy)):
                a = ('move', 2 if dy > 0 else 1)
                all_actions.append(a)
                apply_action(env, a)
            
            print(f"  [Complete] Round {round_num+1}: moved {sp8['sprite']} by ({dx},{dy})")
    
    # 最终检查
    matched = count_matched(get_special_pixels(env._game, list(env._game.current_level.get_sprites())))
    total = len(get_special_pixels(env._game, list(env._game.current_level.get_sprites()))) // 2
    
    if matched >= total and total > 0:
        return all_actions
    
    return None


def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("cn04")
    
    if env is None:
        print("无法创建 cn04 环境")
        return
    
    print("开始完整求解 cn04...")
    
    total_actions = 0
    all_ok = True
    
    for lvl in range(8):
        t0 = time.time()
        
        # 重置环境
        env.reset()
        
        sol = solve_level_complete(env, lvl, max_time=30)
        
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
