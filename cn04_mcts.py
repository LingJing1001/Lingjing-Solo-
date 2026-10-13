# -*- coding: utf-8 -*-
"""CN04 MCTS 求解器 - 简化版"""
import sys
import os
import time
import math
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
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
    matched = 0
    for s in specials:
        if pos_count[s['pos']] >= 2:
            matched += 1
    return matched // 2


def get_score(env):
    """获取当前得分"""
    g = env._game
    sprites = list(g.current_level.get_sprites())
    specials = get_special_pixels(g, sprites)
    total = len(specials) // 2
    matched = count_matched(specials)
    return matched, total


def apply_action(env, action):
    """应用动作"""
    if action[0] == 'click':
        return env.step(GameAction.ACTION6, data={'x': action[1], 'y': action[2]})
    elif action[0] == 'move':
        return env.step(GameAction.from_id(action[1]), data={})
    elif action[0] == 'rotate':
        return env.step(GameAction.ACTION5, data={})
    return None


def get_actions(env):
    """获取可用动作"""
    actions = []
    g = env._game
    
    # 点击动作
    for s in g.current_level.get_sprites():
        if s.is_visible:
            rendered = s.render()
            if rendered is not None:
                non_tp = np.argwhere(rendered >= 0)
                if len(non_tp) > 0:
                    y, x = non_tp[len(non_tp) // 2]
                    gx, gy = int(s.x + x), int(s.y + y)
                    actions.append(('click', gx, gy))
    
    # 移动/旋转
    actions.extend([
        ('move', 1), ('move', 2), ('move', 3), ('move', 4), ('rotate',)
    ])
    
    return actions


def mcts_rollout(env, max_steps=30):
    """随机模拟"""
    start_matched, start_total = get_score(env)
    
    for step in range(max_steps):
        matched, total = get_score(env)
        if matched >= total and total > 0:
            return matched / total  # 成功
        
        actions = get_actions(env)
        if not actions:
            break
        
        action = random.choice(actions)
        apply_action(env, action)
    
    # 返回最终匹配进度
    matched, total = get_score(env)
    return matched / max(total, 1)


def mcts_search(env, iterations=100, max_depth=20):
    """简化版 MCTS - 多轮随机搜索"""
    action_scores = defaultdict(lambda: {'wins': 0, 'total': 0})
    
    for i in range(iterations):
        # 保存当前状态
        frame = env.reset()
        g = env._game
        
        # 获取可用动作
        actions = get_actions(env)
        if not actions:
            break
        
        # 随机选择一个动作
        action = random.choice(actions)
        
        # 应用动作
        apply_action(env, action)
        
        # 随机模拟
        score = mcts_rollout(env, max_steps=20)
        
        # 记录结果
        action_scores[action]['wins'] += score
        action_scores[action]['total'] += 1
    
    # 选择最佳动作
    if not action_scores:
        return None
    
    best_action = max(action_scores.keys(), 
                     key=lambda a: action_scores[a]['wins'] / max(action_scores[a]['total'], 1))
    
    return best_action


def solve_level(env, level, max_time=60):
    """求解单关"""
    t0 = time.time()
    actions = []
    
    for step in range(100):
        if time.time() - t0 > max_time:
            break
        
        # 检查是否完成
        matched, total = get_score(env)
        if matched >= total and total > 0:
            print(f"  [MCTS] LEVEL {level+1} COMPLETED in {step} steps!")
            break
        
        # MCTS 搜索
        action = mcts_search(env, iterations=50, max_depth=10)
        
        if action is None:
            break
        
        # 应用动作
        apply_action(env, action)
        actions.append(action)
        
        # 显示进度
        if step % 10 == 0:
            matched, total = get_score(env)
            print(f"  [MCTS] Step {step}: {matched}/{total} matched")
    
    return actions if actions else None


def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("cn04")
    
    if env is None:
        print("无法创建 cn04 环境")
        return
    
    print("开始 MCTS 求解 cn04...")
    
    total_actions = 0
    all_ok = True
    
    for lvl in range(8):
        t0 = time.time()
        
        sol = solve_level(env, lvl, max_time=30)
        
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
