# -*- coding: utf-8 -*-
"""CN04 启发式求解器 - 基于特殊像素位置"""
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
    matched = 0
    for s in specials:
        if pos_count[s['pos']] >= 2:
            matched += 1
    return matched // 2


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


def solve_level_heuristic(env, level, max_time=60):
    """启发式求解 - 基于特殊像素配对"""
    t0 = time.time()
    actions = []
    
    # 获取初始状态
    frame = env.reset()
    g = env._game
    
    # 获取所有 sprite 和特殊像素
    sprites = list(g.current_level.get_sprites())
    specials = get_special_pixels(g, sprites)
    
    print(f"  [Heuristic] Level {level+1}: {len(sprites)} sprites, {len(specials)} special pixels")
    
    # 分组：8 型和 13 型
    type_8 = [s for s in specials if s['type'] == '8']
    type_13 = [s for s in specials if s['type'] == '13']
    
    print(f"  [Heuristic] 8-type: {len(type_8)}, 13-type: {len(type_13)}")
    
    # 策略：找到未配对的 8 和 13，尝试让它们重叠
    # 对于每个 sprite，找到它的 8 和 13 位置
    sprite_specials = defaultdict(list)
    for s in specials:
        sprite_specials[s['sprite']].append(s)
    
    # 对于每个 sprite，尝试旋转让 8 和 13 对齐到其他 sprite 的 13 和 8
    for sprite_name, sp_list in sprite_specials.items():
        if time.time() - t0 > max_time:
            break
        
        # 找到这个 sprite
        sprite = None
        for s in sprites:
            if s.name == sprite_name and s.is_visible:
                sprite = s
                break
        
        if sprite is None:
            continue
        
        # 找点击位置
        click_pos = find_click_pos(sprite)
        if click_pos is None:
            continue
        
        # 找到这个 sprite 的 8 和 13
        sp_8 = [s for s in sp_list if s['type'] == '8']
        sp_13 = [s for s in sp_list if s['type'] == '13']
        
        if not sp_8 or not sp_13:
            continue
        
        # 对于每个 8，找一个未配对的 13
        for s8 in sp_8:
            for s13 in sp_13:
                # 计算需要移动的偏移
                dx = s13['pos'][0] - s8['pos'][0]
                dy = s13['pos'][1] - s8['pos'][1]
                
                # 如果已经在同一位置，跳过
                if dx == 0 and dy == 0:
                    continue
                
                # 尝试：点击 sprite -> 旋转 -> 移动
                # 先点击
                actions.append(('click', click_pos[0], click_pos[1]))
                apply_action(env, ('click', click_pos[0], click_pos[1]))
                
                # 尝试不同旋转
                for rot in range(4):
                    if rot > 0:
                        actions.append(('rotate',))
                        apply_action(env, ('rotate',))
                    
                    # 移动
                    for _ in range(abs(dx)):
                        move_action = ('move', 4 if dx > 0 else 3)
                        actions.append(move_action)
                        apply_action(env, move_action)
                    
                    for _ in range(abs(dy)):
                        move_action = ('move', 2 if dy > 0 else 1)
                        actions.append(move_action)
                        apply_action(env, move_action)
                    
                    # 检查是否匹配
                    matched = count_matched(get_special_pixels(env._game, list(env._game.current_level.get_sprites())))
                    if matched > 0:
                        print(f"  [Heuristic] Found match! {matched}")
                        break
                
                # 检查是否完成
                matched = count_matched(get_special_pixels(env._game, list(env._game.current_level.get_sprites())))
                if matched >= 1:
                    print(f"  [Heuristic] LEVEL COMPLETED!")
                    return actions
    
    # 如果启发式方法没找到解，尝试 BFS
    print(f"  [Heuristic] Heuristic failed, trying BFS...")
    bfs_actions = solve_level_bfs(env, max_time=max_time - (time.time() - t0))
    
    if bfs_actions:
        return bfs_actions
    
    return actions if actions else None


def solve_level_bfs(env, max_time=30):
    """BFS 搜索"""
    from collections import deque
    
    t0 = time.time()
    visited = set()
    queue = deque([([], env.reset())])
    
    max_depth = 20
    
    while queue and time.time() - t0 < max_time:
        actions, frame = queue.popleft()
        
        # 检查是否完成
        matched = count_matched(get_special_pixels(env._game, list(env._game.current_level.get_sprites())))
        if matched >= 1:
            print(f"  [BFS] Found solution in {len(actions)} steps")
            return actions
        
        if len(actions) >= max_depth:
            continue
        
        # 生成候选动作
        g = env._game
        candidates = []
        
        # 点击动作
        for s in g.current_level.get_sprites():
            if s.is_visible:
                pos = find_click_pos(s)
                if pos:
                    candidates.append(('click', pos[0], pos[1]))
        
        # 移动/旋转
        candidates.extend([('move', 1), ('move', 2), ('move', 3), ('move', 4), ('rotate',)])
        
        for action in candidates:
            # 应用动作
            if action[0] == 'click':
                new_frame = env.step(GameAction.ACTION6, data={'x': action[1], 'y': action[2]})
            elif action[0] == 'move':
                new_frame = env.step(GameAction.from_id(action[1]), data={})
            elif action[0] == 'rotate':
                new_frame = env.step(GameAction.ACTION5, data={})
            
            # 状态哈希
            state_hash = hash(str([s.name for s in env._game.current_level.get_sprites()]))
            
            if state_hash not in visited:
                visited.add(state_hash)
                queue.append((actions + [action], new_frame))
    
    return None


def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("cn04")
    
    if env is None:
        print("无法创建 cn04 环境")
        return
    
    print("开始启发式求解 cn04...")
    
    total_actions = 0
    all_ok = True
    
    for lvl in range(8):
        t0 = time.time()
        
        sol = solve_level_heuristic(env, lvl, max_time=30)
        
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
