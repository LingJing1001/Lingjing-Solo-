# -*- coding: utf-8 -*-
"""运行 cn04 获取真实 episode 数据 - 直接使用引擎"""
import sys
import os
import time
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KAGGLE = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"
sys.path.insert(0, str(KAGGLE))
os.environ.setdefault("LINGJING_SRC", str(ROOT / "lingjing_solo"))
sys.path.insert(0, str(ROOT))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, ActionInput, GameState
import numpy as np

def make_env(gid="cn04"):
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    return arc.make(gid)

def apply_action(env, action):
    """应用动作"""
    if isinstance(action, int):
        return env.step(GameAction.from_id(action), data={})
    elif isinstance(action, (list, tuple)):
        if len(action) == 1:
            # 单参数动作: (5,) 表示旋转
            return env.step(GameAction.from_id(action[0]), data={})
        elif len(action) == 3 and action[0] == 6:
            # 点击动作: (6, x, y)
            return env.step(GameAction.ACTION6, data={"x": action[1], "y": action[2]})
    raise ValueError(f"Unknown action: {action}")

def replay(env, actions, level):
    frame = env.reset()
    for a in actions:
        frame = apply_action(env, a)
        if frame.levels_completed > level or frame.state.name == "WIN":
            break
    return frame

def get_special_pixels(sprite, game):
    """获取 sprite 的特殊像素位置"""
    orig = game.hlxyvcmpk.get(sprite.name)
    if orig is None:
        return []
    h, w = orig.shape
    pts_local = []
    for y in range(h):
        for x in range(w):
            if orig[y, x] in (8, 13):
                pts_local.append((x, y))
    if not pts_local:
        return []
    rot = sprite.rotation
    pts = []
    for x, y in pts_local:
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
        pts.append((int(sprite.x + rx), int(sprite.y + ry)))
    return pts

def find_click_pos(game, sprite):
    """找能选中 sprite 的坐标"""
    rendered = sprite.render()
    if rendered is None:
        return None
    non_tp = np.argwhere(rendered >= 0)
    if len(non_tp) == 0:
        return None
    y, x = non_tp[len(non_tp) // 2]
    return (int(sprite.x + x), int(sprite.y + y))

def solve_level_direct(env, level):
    """直接分析法求解"""
    frame = env.reset()
    g = env._game
    
    # 获取所有可见 sprite
    sprites = [s for s in g.current_level.get_sprites() if s.is_visible]
    if len(sprites) < 2:
        return None
    
    # 获取特殊像素
    specials = []
    for s in sprites:
        sp = get_special_pixels(s, g)
        if sp:
            specials.append((s, sp))
    
    if len(specials) < 2:
        return None
    
    s1, sp1 = specials[0]
    s2, sp2 = specials[1]
    
    click1 = find_click_pos(g, s1)
    click2 = find_click_pos(g, s2)
    
    if not click1 or not click2:
        return None
    
    # 尝试旋转 + 移动
    for rot_count in range(4):
        trial = []
        trial.append((6, click1[0], click1[1]))
        for _ in range(rot_count):
            trial.append((5,))
        
        # 回放查看状态
        frame = replay(env, trial, level)
        g = env._game
        
        # 找到 s1 当前位置
        s1_cur = None
        for s in g.current_level.get_sprites():
            if s.name == s1.name and s.is_visible:
                s1_cur = s
                break
        if not s1_cur:
            continue
        
        sp1_cur = get_special_pixels(s1_cur, g)
        if len(sp1_cur) != len(sp2):
            continue
        
        # 计算偏移
        dx = sp2[0][0] - sp1_cur[0][0]
        dy = sp2[0][1] - sp1_cur[0][1]
        
        if len(sp1_cur) > 1 and len(sp2) > 1:
            if sp1_cur[1][0] + dx == sp2[1][0] and sp1_cur[1][1] + dy == sp2[1][1]:
                moves = []
                for _ in range(abs(dx)):
                    moves.append((4 if dx > 0 else 3,))
                for _ in range(abs(dy)):
                    moves.append((2 if dy > 0 else 1,))
                full = trial + moves
                frame = replay(env, full, level)
                if frame.levels_completed > level or frame.state.name == "WIN":
                    return full
    
    return None

def main():
    env = make_env("cn04")
    if env is None:
        print("无法创建 cn04 环境")
        return
    
    print(f"开始求解 cn04...")
    
    total_actions = 0
    all_ok = True
    all_actions = []
    
    for lvl in range(8):
        t0 = time.time()
        
        sol = solve_level_direct(env, lvl)
        
        dt = time.time() - t0
        
        if sol is None:
            print(f"L{lvl+1}: NO SOLUTION  {dt:.1f}s")
            all_ok = False
            break
        
        frame = replay(env, sol, lvl)
        ok = frame.levels_completed > lvl or frame.state.name == "WIN"
        total_actions += len(sol)
        all_actions.extend(sol)
        
        print(f"L{lvl+1}: {len(sol)} actions -> levels={frame.levels_completed}  "
              f"{'OK' if ok else 'FAIL'}  {dt:.1f}s")
        
        if not ok:
            all_ok = False
            break
    
    print(f"\nTotal: actions={total_actions}")
    print("ALL_PASS" if all_ok else "FAIL")
    
    if all_ok:
        print(f"\n动作序列: {all_actions}")

if __name__ == "__main__":
    main()
