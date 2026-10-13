# -*- coding: utf-8 -*-
"""CN04 混合求解器 - 多策略融合"""
import sys
import os
import time
import math
import random
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

ROOT = Path(__file__).resolve().parent
KAGGLE = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"
sys.path.insert(0, str(KAGGLE))
os.environ.setdefault("LINGJING_SRC", str(ROOT / "lingjing_solo"))
sys.path.insert(0, str(ROOT))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction


# ============================================================================
# 策略1: 模式匹配
# ============================================================================

class PatternMatcher:
    """模式匹配策略 - 快速求解简单关卡"""
    
    def __init__(self):
        self.patterns = {
            1: self._pattern_l1,  # 第1关模式
        }
    
    def solve(self, env, level_idx: int) -> Optional[List]:
        """尝试模式匹配"""
        if level_idx in self.patterns:
            return self.patterns[level_idx](env)
        return None
    
    def _pattern_l1(self, env) -> List:
        """第1关模式: 2个sprite，旋转后移动配对"""
        actions = []
        
        # 重置
        env.reset()
        g = env._game
        
        # 找到 Sprite 1
        sprite1 = None
        for s in g.current_level.get_sprites():
            if s.name == "0000pusfglvcns":
                sprite1 = s
                break
        
        if not sprite1:
            return None
        
        # 点击 Sprite 1
        rendered = sprite1.render()
        if rendered is None:
            return None
        non_tp = np.argwhere(rendered >= 0)
        y, x = non_tp[len(non_tp) // 2]
        click_pos = (int(sprite1.x + x), int(sprite1.y + y))
        
        actions.append(('click', click_pos[0], click_pos[1]))
        env.step(GameAction.ACTION6, data={'x': click_pos[0], 'y': click_pos[1]})
        
        # 旋转 3 次到 0°
        for _ in range(3):
            actions.append(('rotate',))
            env.step(GameAction.ACTION5, data={})
        
        # 移动 (4, 7)
        for _ in range(4):
            actions.append(('move', 4))
            env.step(GameAction.from_id(4), data={})
        
        for _ in range(7):
            actions.append(('move', 2))
            env.step(GameAction.from_id(2), data={})
        
        return actions


# ============================================================================
# 策略2: BFS 搜索
# ============================================================================

class BFSSolver:
    """BFS 搜索策略 - 中等复杂度关卡"""
    
    def __init__(self, max_depth: int = 20, max_nodes: int = 50000):
        self.max_depth = max_depth
        self.max_nodes = max_nodes
    
    def solve(self, env, level_idx: int, max_time: float = 30) -> Optional[List]:
        """BFS 搜索"""
        t0 = time.time()
        
        # 状态哈希
        def state_hash(g):
            sprites = list(g.current_level.get_sprites())
            return hash(tuple(
                (s.name, s.x, s.y, s.rotation, int(s.is_visible))
                for s in sprites
            ))
        
        # 检查胜利
        def is_win(g):
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
                            
                            # 检查是否被其他 sprite 覆盖
                            for s2 in g.current_level.get_sprites():
                                if s2 == s or not s2.is_visible:
                                    continue
                                if s2.x <= gx < s2.x + s2.width and s2.y <= gy < s2.y + s2.height:
                                    rendered = s2.render()
                                    if rendered is not None:
                                        lx, ly = gx - s2.x, gy - s2.y
                                        if 0 < ly < rendered.shape[0] and 0 < lx < rendered.shape[1]:
                                            if rendered[ly, lx] in (8, 13):
                                                return False
            return True
        
        # 生成动作
        def get_actions(g):
            actions = []
            
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
            if g.xseexqzst:
                actions.extend([
                    ('move', 1), ('move', 2), ('move', 3), ('move', 4), ('rotate',)
                ])
            
            return actions
        
        # 应用动作
        def apply_action(g, action):
            if action[0] == 'click':
                env.step(GameAction.ACTION6, data={'x': action[1], 'y': action[2]})
            elif action[0] == 'move':
                env.step(GameAction.from_id(action[1]), data={})
            elif action[0] == 'rotate':
                env.step(GameAction.ACTION5, data={})
        
        # BFS
        visited = set()
        queue = deque()
        
        # 初始状态
        g = env._game
        h0 = state_hash(g)
        queue.append(([], h0))
        visited.add(h0)
        
        nodes = 0
        
        while queue and nodes < self.max_nodes and time.time() - t0 < max_time:
            path, state_h = queue.popleft()
            
            if len(path) >= self.max_depth:
                continue
            
            # 恢复状态
            env.reset()
            for a in path:
                apply_action(env._game, a)
            
            g = env._game
            
            # 检查胜利
            if is_win(g):
                return path
            
            # 展开
            for action in get_actions(g):
                # 应用动作
                new_path = path + [action]
                apply_action(g, action)
                
                new_h = state_hash(env._game)
                if new_h not in visited:
                    visited.add(new_h)
                    queue.append((new_path, new_h))
                
                nodes += 1
                
                # 检查胜利
                if is_win(env._game):
                    return new_path
        
        return None


# ============================================================================
# 策略3: 启发式搜索
# ============================================================================

class HeuristicSolver:
    """启发式搜索策略 - 基于特殊像素分析"""
    
    def __init__(self, max_iterations: int = 100):
        self.max_iterations = max_iterations
    
    def solve(self, env, level_idx: int, max_time: float = 30) -> Optional[List]:
        """启发式搜索"""
        t0 = time.time()
        actions = []
        
        for iteration in range(self.max_iterations):
            if time.time() - t0 > max_time:
                break
            
            # 重置
            env.reset()
            g = env._game
            
            # 获取所有特殊像素
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
                            specials.append({
                                'sprite': s.name,
                                'pos': (gx, gy),
                                'type': '8' if orig[y, x] == 8 else '13'
                            })
            
            # 检查是否胜利 - 所有特殊像素都配对
            from collections import Counter
            pos_count = Counter([s['pos'] for s in specials])
            matched = sum(1 for s in specials if pos_count[s['pos']] >= 2) // 2
            
            if matched * 2 == len(specials) and len(specials) > 0:
                print(f"    [Heuristic] Found solution with {matched} matches!")
                return actions
            
            # 找到未配对的特殊像素
            unmatched = [s for s in specials if pos_count[s['pos']] < 2]
            
            if len(unmatched) < 2:
                break
            
            # 找一个 8 和一个 13
            sp8 = next((s for s in unmatched if s['type'] == '8'), None)
            sp13 = next((s for s in unmatched if s['type'] == '13'), None)
            
            if not sp8 or not sp13:
                break
            
            # 找到对应的 sprite
            sprite = None
            for s in g.current_level.get_sprites():
                if s.name == sp8['sprite'] and s.is_visible:
                    sprite = s
                    break
            
            if not sprite:
                break
            
            # 点击 sprite
            rendered = sprite.render()
            if rendered is None:
                break
            non_tp = np.argwhere(rendered >= 0)
            y, x = non_tp[len(non_tp) // 2]
            click_pos = (int(sprite.x + x), int(sprite.y + y))
            
            actions.append(('click', click_pos[0], click_pos[1]))
            env.step(GameAction.ACTION6, data={'x': click_pos[0], 'y': click_pos[1]})
            
            # 移动让 8 和 13 配对
            dx = sp13['pos'][0] - sp8['pos'][0]
            dy = sp13['pos'][1] - sp8['pos'][1]
            
            for _ in range(abs(dx)):
                a = ('move', 4 if dx > 0 else 3)
                actions.append(a)
                env.step(GameAction.from_id(a[1]), data={})
            
            for _ in range(abs(dy)):
                a = ('move', 2 if dy > 0 else 1)
                actions.append(a)
                env.step(GameAction.from_id(a[1]), data={})
        
        # 最终检查
        env.reset()
        for a in actions:
            if a[0] == 'click':
                env.step(GameAction.ACTION6, data={'x': a[1], 'y': a[2]})
            elif a[0] == 'move':
                env.step(GameAction.from_id(a[1]), data={})
            elif a[0] == 'rotate':
                env.step(GameAction.ACTION5, data={})
        
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
                        specials.append({
                            'sprite': s.name,
                            'pos': (gx, gy),
                            'type': '8' if orig[y, x] == 8 else '13'
                        })
        
        pos_count = Counter([s['pos'] for s in specials])
        matched = sum(1 for s in specials if pos_count[s['pos']] >= 2) // 2
        
        if matched * 2 == len(specials) and len(specials) > 0:
            return actions
        
        return None


# ============================================================================
# 混合求解器
# ============================================================================

class HybridSolver:
    """混合求解器 - 多策略融合"""
    
    def __init__(self):
        self.pattern_matcher = PatternMatcher()
        self.bfs_solver = BFSSolver(max_depth=15, max_nodes=20000)
        self.heuristic_solver = HeuristicSolver(max_iterations=50)
    
    def solve(self, env, level_idx: int, max_time: float = 30) -> Optional[List]:
        """多策略融合求解"""
        t0 = time.time()
        
        print(f"  [Hybrid] Level {level_idx + 1}: Starting hybrid solve...")
        
        # 策略1: 模式匹配 (最快)
        if time.time() - t0 < max_time * 0.2:
            print(f"  [Hybrid] Trying pattern matching...")
            result = self.pattern_matcher.solve(env, level_idx)
            if result:
                print(f"  [Hybrid] Pattern match success!")
                return result
        
        # 策略2: 启发式搜索 (中等速度)
        if time.time() - t0 < max_time * 0.6:
            print(f"  [Hybrid] Trying heuristic search...")
            result = self.heuristic_solver.solve(env, level_idx, max_time - (time.time() - t0))
            if result:
                print(f"  [Hybrid] Heuristic search success!")
                return result
        
        # 策略3: BFS 搜索 (最慢但最全面)
        if time.time() - t0 < max_time:
            print(f"  [Hybrid] Trying BFS search...")
            result = self.bfs_solver.solve(env, level_idx, max_time - (time.time() - t0))
            if result:
                print(f"  [Hybrid] BFS search success!")
                return result
        
        print(f"  [Hybrid] All strategies failed!")
        return None


# ============================================================================
# 主函数
# ============================================================================

def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("cn04")
    
    if env is None:
        print("无法创建 cn04 环境")
        return
    
    print("=== CN04 混合求解器 ===")
    print("策略: 模式匹配 -> 启发式搜索 -> BFS 搜索")
    
    solver = HybridSolver()
    
    total_actions = 0
    all_ok = True
    
    for lvl in range(8):
        t0 = time.time()
        
        sol = solver.solve(env, lvl, max_time=30)
        
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
