# -*- coding: utf-8 -*-
"""CN04 调试 - 检查移动后状态"""
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


def get_specials(game):
    """获取所有特殊像素"""
    specials = []
    for s in game.current_level.get_sprites():
        if s.is_visible:
            orig = game.hlxyvcmpk.get(s.name)
            if orig is not None:
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
    pos_count = Counter([s[1:3] for s in specials])
    return sum(1 for s in specials if pos_count[s[1:3]] >= 2) // 2


def debug_state(env, label):
    """调试游戏状态"""
    g = env._game
    specials = get_specials(g)
    matched = count_matched(specials)
    print(f"\n  [{label}] Level {g.level_index}, State: {g._state.name}")
    print(f"    Matched: {matched}")
    for s in specials:
        print(f"      {s[0]}: ({s[1]},{s[2]}) {s[3]}")


def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("cn04")
    
    if env is None:
        print("无法创建 cn04 环境")
        return
    
    print("=== CN04 调试 - 移动测试 ===")
    
    # 重置
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
    click_pos = find_click_pos(sprite1)
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
