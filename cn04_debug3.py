# -*- coding: utf-8 -*-
"""CN04 调试 - 修复后的匹配检测"""
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


def apply_action(env, action):
    """应用动作"""
    if action[0] == 'click':
        return env.step(GameAction.ACTION6, data={'x': action[1], 'y': action[2]})
    elif action[0] == 'move':
        return env.step(GameAction.from_id(action[1]), data={})
    elif action[0] == 'rotate':
        return env.step(GameAction.ACTION5, data={})
    return None


def get_specials_from_pixels(env):
    """从 sprite.pixels 获取特殊像素（显示当前状态，包括匹配的）"""
    g = env._game
    specials = []
    
    for s in g.current_level.get_sprites():
        if not s.is_visible:
            continue
        
        rendered = s.render()
        if rendered is None:
            continue
        
        for y in range(rendered.shape[0]):
            for x in range(rendered.shape[1]):
                if rendered[y, x] in (8, 13):
                    gx, gy = int(s.x + x), int(s.y + y)
                    specials.append((s.name, gx, gy, '8' if rendered[y, x] == 8 else '13'))
                elif rendered[y, x] == 3:
                    # 匹配的像素
                    gx, gy = int(s.x + x), int(s.y + y)
                    specials.append((s.name, gx, gy, '3 (matched)'))
    
    return specials


def get_specials_from_original(env):
    """从 hlxyvcmpk 获取特殊像素（显示原始状态，不包括匹配的）"""
    g = env._game
    specials = []
    
    for s in g.current_level.get_sprites():
        if not s.is_visible:
            continue
        
        orig = g.hlxyvcmpk.get(s.name)
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
                    specials.append((s.name, gx, gy, '8' if orig[y, x] == 8 else '13'))
    
    return specials


def count_matched(specials):
    """计算匹配的对数"""
    from collections import Counter
    # 只统计8和13
    specials_filtered = [s for s in specials if s[3] in ('8', '13')]
    pos_count = Counter([s[1:3] for s in specials_filtered])
    return sum(1 for s in specials_filtered if pos_count[s[1:3]] >= 2) // 2


def debug_state(env, label):
    """调试游戏状态"""
    g = env._game
    
    print(f"\n  [{label}] Level {g.level_index}, State: {g._state.name}")
    
    # 从 pixels 获取（显示当前状态，包括匹配的）
    specials_pixels = get_specials_from_pixels(env)
    matched = count_matched(specials_pixels)
    
    print(f"    From pixels (current state):")
    print(f"      Matched: {matched}")
    for s in specials_pixels:
        print(f"      {s[0]}: ({s[1]},{s[2]}) {s[3]}")
    
    # 从原始数据获取（显示原始状态）
    specials_orig = get_specials_from_original(env)
    print(f"    From original (no matches):")
    for s in specials_orig:
        print(f"      {s[0]}: ({s[1]},{s[2]}) {s[3]}")


def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("cn04")
    
    if env is None:
        print("无法创建 cn04 环境")
        return
    
    print("=== CN04 调试 - 修复后的匹配检测 ===")
    
    # 初始状态
    frame = env.reset()
    debug_state(env, "INIT")
    
    # 找到 Sprite 1
    g = env._game
    sprite1 = None
    for s in g.current_level.get_sprites():
        if s.name == "0000pusfglvcns":
            sprite1 = s
            break
    
    # 点击 Sprite 1
    rendered = sprite1.render()
    non_tp = np.argwhere(rendered >= 0)
    y, x = non_tp[len(non_tp) // 2]
    click_pos = (int(sprite1.x + x), int(sprite1.y + y))
    
    print(f"\nClick Sprite 1 at {click_pos}")
    apply_action(env, ('click', click_pos[0], click_pos[1]))
    debug_state(env, "AFTER CLICK")
    
    # 旋转 3 次到 0°
    for i in range(3):
        print(f"\nRotate {i+1}")
        apply_action(env, ('rotate',))
        debug_state(env, f"AFTER ROTATE {i+1}")
    
    # 移动 (4, 7)
    print(f"\nMove right 4")
    for i in range(4):
        apply_action(env, ('move', 4))
    debug_state(env, "AFTER MOVE RIGHT 4")
    
    print(f"\nMove down 7")
    for i in range(7):
        apply_action(env, ('move', 2))
    debug_state(env, "AFTER MOVE DOWN 7")
    
    # 检查最终状态
    print(f"\n=== FINAL ===")
    g = env._game
    print(f"Level: {g.level_index}")
    print(f"State: {g._state.name}")
    print(f"Levels completed: {g.level_index}")


if __name__ == "__main__":
    main()
