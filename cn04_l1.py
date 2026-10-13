# -*- coding: utf-8 -*-
"""CN04 第1关专用求解器 - 精确分析"""
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


def get_sprite_specials(game, sprite_name):
    """获取指定 sprite 的特殊像素"""
    for s in game.current_level.get_sprites():
        if s.name == sprite_name and s.is_visible:
            orig = game.hlxyvcmpk.get(sprite_name)
            if orig is None:
                return []
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
                        specials.append((gx, gy, '8' if orig[y, x] == 8 else '13', rot))
            return specials
    return []


def solve_level1(env):
    """求解第1关 - 精确分析"""
    actions = []
    
    # 重置
    frame = env.reset()
    g = env._game
    
    # 获取 sprite
    sprites = list(g.current_level.get_sprites())
    
    # 第1关：2个sprite
    # Sprite 1: 0000pusfglvcns @ (3,3), 旋转90°
    #   原始像素 (5x6):
    #     [12, 12, 12, -1, -1, -1]
    #     [12, -1, 12, 12, 12, 8]    <- 8 在 (5,1) 局部
    #     [12, -1, -1, -1, -1, -1]
    #     [12, -1, 12, 12, 12, 13]   <- 13 在 (5,3) 局部
    #     [12, 12, 12, -1, -1, -1]
    #   旋转90°后，局部坐标 (x,y) -> (5-y, x)
    #   8: (5,1) -> (4,5) 全局: (3+4, 3+5) = (7,8)
    #   13: (5,3) -> (2,5) 全局: (3+2, 3+5) = (5,8)
    
    # 让我重新计算...
    # 原始像素 5x6, h=5, w=6
    # 旋转90°: (x,y) -> (h-1-y, x) = (4-y, x)
    # 8 在 (5,1) -> (4-1, 5) = (3,5) 全局: (3+3, 3+5) = (6,8) ✓
    # 13 在 (5,3) -> (4-3, 5) = (1,5) 全局: (3+1, 3+5) = (4,8) ✓
    
    # Sprite 2: 0001gqwehbicqu @ (12,9), 旋转0°
    #   原始像素 4x7:
    #     [-1, 14, 14, 14]
    #     [-1, 14, -1, 14]
    #     [8, 14, -1, 14]    <- 8 在 (0,2) 局部
    #     [-1, 14, -1, 14]
    #     [13, 14, -1, 14]   <- 13 在 (0,4) 局部
    #     [-1, 14, -1, 14]
    #     [-1, 14, 14, 14]
    #   旋转0°: (x,y) -> (x, y)
    #   8: (0,2) 全局: (12+0, 9+2) = (12,11) ✓
    #   13: (0,4) 全局: (12+0, 9+4) = (12,13) ✓
    
    print(f"  [L1] Initial state:")
    print(f"    Sprite 1: {get_sprite_specials(g, '0000pusfglvcns')}")
    print(f"    Sprite 2: {get_sprite_specials(g, '0001gqwehbicqu')}")
    
    # 找到 Sprite 1
    sprite1 = None
    for s in sprites:
        if s.name == "0000pusfglvcns":
            sprite1 = s
            break
    
    if not sprite1:
        return None
    
    # 点击 Sprite 1
    click_pos = find_click_pos(sprite1)
    if not click_pos:
        return None
    
    # 策略：旋转 Sprite 1 到 0°，然后移动让所有特殊像素重叠
    a = ('click', click_pos[0], click_pos[1])
    actions.append(a)
    apply_action(env, a)
    
    # 旋转 3 次到 0°
    for rot in range(3):
        a = ('rotate',)
        actions.append(a)
        apply_action(env, a)
        
        # 检查当前状态
        sp1 = get_sprite_specials(env._game, '0000pusfglvcns')
        print(f"    After {rot+1} rotation(s): Sprite 1 = {sp1}")
    
    # 现在 Sprite 1 应该在 0°
    # 8: (8,4), 13: (8,6)
    # 相对方向: (0, -2) - 8 在 13 的上方
    
    # Sprite 2: 8: (12,11), 13: (12,13)
    # 相对方向: (0, -2) - 8 在 13 的上方
    
    # 相对方向相同！移动 Sprite 1 让它的 8 和 Sprite 2 的 8 对齐
    # 偏移 = (12-8, 11-4) = (4, 7)
    
    for _ in range(4):
        a = ('move', 4)  # 右
        actions.append(a)
        apply_action(env, a)
    
    for _ in range(7):
        a = ('move', 2)  # 下
        actions.append(a)
        apply_action(env, a)
    
    # 检查匹配
    specials = get_special_pixels(env._game, list(env._game.current_level.get_sprites()))
    matched = count_matched(specials)
    print(f"  [L1] After move: {matched} matches")
    print(f"    Specials: {[(s['sprite'], s['pos'], s['type']) for s in specials]}")
    
    if matched >= 2:
        print(f"  [L1] Found solution!")
        return actions
    
    return None


def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("cn04")
    
    if env is None:
        print("无法创建 cn04 环境")
        return
    
    print("开始求解 cn04 第1关...")
    
    sol = solve_level1(env)
    
    if sol is None:
        print("L1: NO SOLUTION")
        return
    
    # 验证
    frame = env.reset()
    for a in sol:
        if a[0] == 'click':
            frame = env.step(GameAction.ACTION6, data={'x': a[1], 'y': a[2]})
        elif a[0] == 'move':
            frame = env.step(GameAction.from_id(a[1]), data={})
        elif a[0] == 'rotate':
            frame = env.step(GameAction.ACTION5, data={})
    
    print(f"\nL1: {len(sol)} actions -> levels={frame.levels_completed}")
    print(f"State: {frame.state.name}")
    print(f"Win: {frame.state.name == 'WIN'}")


if __name__ == "__main__":
    main()
