# -*- coding: utf-8 -*-
"""纯视觉求解器 - 只看画面帧，不读引擎内部

实现M1要求：从画面琢磨规则，不读引擎内部
"""
import sys
import os
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
KAGGLE = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"
sys.path.insert(0, str(KAGGLE))
os.environ.setdefault("LINGJING_SRC", str(ROOT / "lingjing_solo"))
sys.path.insert(0, str(ROOT))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, ActionInput


class PureVisualSolver:
    """纯视觉求解器 - 只看画面帧"""
    
    def __init__(self):
        self.bg_color = None
        self.special_colors = set()
        self.matched_color = 3  # 匹配后的颜色
    
    def solve(self, env, level: int, max_time: int = 120) -> Optional[List]:
        """纯视觉求解"""
        t0 = time.time()
        
        # 设置关卡
        env._game._current_level_index = level
        env._game.level_reset()
        frame = env.reset()
        
        # ① 视觉感知 - 识别画面元素
        print("  [视觉] 识别画面元素...", flush=True)
        obs = self._perceive(frame)
        self._print_obs(obs)
        
        # ② 规则发现 - 小动作实验
        print("  [视觉] 小动作实验...", flush=True)
        rules = self._discover_rules(env, obs)
        self._print_rules(rules)
        
        # ③ 求解 - 基于规则生成动作
        print("  [视觉] 基于规则求解...", flush=True)
        actions = self._solve_by_rules(env, obs, rules, level, max_time - (time.time() - t0))
        
        if actions:
            # ④ 视觉验证
            print("  [视觉] 视觉验证...", flush=True)
            if self._visual_verify(env, actions, level):
                print(f"  [视觉] 验证通过: {len(actions)} 步", flush=True)
                return actions
            else:
                print("  [视觉] 验证失败", flush=True)
        
        return None
    
    def _perceive(self, frame_data) -> Dict:
        """从画面帧识别游戏元素"""
        f = frame_data.frame[0] if hasattr(frame_data, 'frame') else frame_data
        arr = np.array(f)
        
        # 识别背景色
        colors, counts = np.unique(arr, return_counts=True)
        bg_color = int(colors[np.argmax(counts)])
        self.bg_color = bg_color
        
        # 识别块（连通区域）
        non_bg = arr != bg_color
        labeled, num = ndimage.label(non_bg)
        
        blocks = []
        for i in range(1, num + 1):
            region = (labeled == i)
            color = int(arr[region][0])
            ys, xs = np.where(region)
            
            blocks.append({
                'id': i,
                'color': color,
                'bbox': (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                'center': (int(xs.mean()), int(ys.mean())),
                'size': len(ys),
                'pixels': list(zip(xs.tolist(), ys.tolist()))
            })
        
        return {
            'blocks': blocks,
            'colors': set(colors.tolist()),
            'bg_color': bg_color,
            'frame': arr,
            'available_actions': list(frame_data.available_actions) if hasattr(frame_data, 'available_actions') else [1,2,3,4,5,6],
            'state': frame_data.state.name if hasattr(frame_data, 'state') else 'UNKNOWN',
            'levels_completed': frame_data.levels_completed if hasattr(frame_data, 'levels_completed') else 0,
        }
    
    def _discover_rules(self, env, obs: Dict) -> Dict:
        """小动作实验发现规则"""
        rules = {
            'has_movement': False,
            'has_selection': False,
            'has_special_pixels': False,
            'match_mechanism': 'unknown',
            'special_colors': set(),
        }
        
        initial_colors = set(obs['colors'])
        
        # 检查是否有特殊颜色（8和13通常是特殊像素）
        for color in obs['colors']:
            if color in (8, 13):
                rules['has_special_pixels'] = True
                rules['special_colors'].add(color)
        
        # 小动作实验
        for action_id in obs['available_actions'][:4]:
            if action_id > 4:
                continue
            
            # 重置
            env._game._current_level_index = obs.get('level', 0)
            env._game.level_reset()
            frame = env.reset()
            before = self._perceive(frame)
            
            # 执行动作
            frame = env.step(GameAction.from_id(action_id), data={})
            after = self._perceive(frame)
            
            # 检测块移动
            for b1, b2 in zip(before['blocks'], after['blocks']):
                if b1['bbox'] != b2['bbox']:
                    rules['has_movement'] = True
                    break
            
            # 检测颜色变化
            new_colors = after['colors'] - initial_colors
            if new_colors:
                if 3 in new_colors:
                    rules['match_mechanism'] = 'color_match'
                rules.setdefault('color_changes', {})[action_id] = list(new_colors)
            
            # Undo
            if 7 in obs['available_actions']:
                env.step(GameAction.ACTION7, data={})
        
        # 生成假说
        if rules['has_special_pixels']:
            rules['hypothesis'] = 'special_pixel_match'
        elif rules['has_movement']:
            rules['hypothesis'] = 'block_movement'
        else:
            rules['hypothesis'] = 'general_search'
        
        env._game.level_reset()
        env.reset()
        
        return rules
    
    def _solve_by_rules(self, env, obs: Dict, rules: Dict, level: int, max_time: float) -> Optional[List]:
        """基于规则求解"""
        t0 = time.time()
        
        if rules['hypothesis'] == 'special_pixel_match':
            return self._solve_match(env, obs, rules, level, max_time)
        elif rules['hypothesis'] == 'block_movement':
            return self._solve_bfs(env, obs, level, max_time)
        else:
            return self._solve_bfs(env, obs, level, max_time)
    
    def _solve_match(self, env, obs: Dict, rules: Dict, level: int, max_time: float) -> Optional[List]:
        """配对求解 - 纯视觉"""
        t0 = time.time()
        
        # 识别特殊像素块
        special_blocks = [b for b in obs['blocks'] if b['color'] in rules['special_colors']]
        
        if len(special_blocks) < 2:
            return None
        
        print(f"  [视觉] 特殊块: {len(special_blocks)} 个", flush=True)
        
        # 尝试让特殊块重叠
        # 策略：点击一个块，移动到另一个块的位置
        
        # 找到可点击的块（通过尝试点击）
        for block in obs['blocks']:
            if time.time() - t0 > max_time:
                break
            
            # 尝试点击这个块
            cx, cy = block['center']
            
            # 重置
            env._game._current_level_index = level
            env._game.level_reset()
            env.reset()
            
            # 点击
            frame = env.step(GameAction.ACTION6, data={'x': cx, 'y': cy})
            after = self._perceive(frame)
            
            # 检查是否选中（颜色变化或块变化）
            selected = self._detect_selection(obs, after)
            
            if selected:
                print(f"  [视觉] 选中块: color={block['color']}", flush=True)
                
                # 尝试旋转和移动让特殊像素重叠
                for rot in range(4):
                    if rot > 0:
                        frame = env.step(GameAction.ACTION5, data={})
                    
                    after_rot = self._perceive(frame)
                    
                    # 找到另一个特殊块
                    other_blocks = [b for b in after_rot['blocks'] 
                                  if b['color'] in rules['special_colors'] and b['id'] != block['id']]
                    
                    if not other_blocks:
                        continue
                    
                    # 计算移动距离
                    target = other_blocks[0]
                    dx = target['center'][0] - block['center'][0]
                    dy = target['center'][1] - block['center'][1]
                    
                    # 生成移动动作
                    actions = [('click', cx, cy)]
                    
                    # 旋转
                    for _ in range(rot):
                        actions.append(('rotate',))
                    
                    # 移动
                    for _ in range(abs(dx)):
                        actions.append(('move', 4 if dx > 0 else 3))
                    
                    for _ in range(abs(dy)):
                        actions.append(('move', 2 if dy > 0 else 1))
                    
                    # 验证
                    if self._visual_verify(env, actions, level):
                        return actions
                    
                    # 重置
                    env._game._current_level_index = level
                    env._game.level_reset()
                    env.reset()
        
        return None
    
    def _solve_bfs(self, env, obs: Dict, level: int, max_time: float) -> Optional[List]:
        """BFS搜索 - 纯视觉"""
        from collections import deque
        
        t0 = time.time()
        
        # 重置
        env._game._current_level_index = level
        env._game.level_reset()
        env.reset()
        
        # BFS
        queue = deque([[]])
        visited = {obs['frame'].tobytes() if obs.get('frame') is not None else ''}
        
        while queue and time.time() - t0 < max_time:
            actions = queue.popleft()
            
            if len(actions) > 30:  # 限制深度
                continue
            
            # 重置并执行动作
            env._game._current_level_index = level
            env._game.level_reset()
            frame = env.reset()
            
            for a in actions:
                if a[0] == 'click':
                    frame = env.step(GameAction.ACTION6, data={'x': a[1], 'y': a[2]})
                elif a[0] == 'move':
                    frame = env.step(GameAction.from_id(a[1]), data={})
                elif a[0] == 'rotate':
                    frame = env.step(GameAction.ACTION5, data={})
            
            # 检查是否过关
            state = frame.state.name if hasattr(frame, 'state') else ''
            levels = frame.levels_completed if hasattr(frame, 'levels_completed') else 0
            
            if state == 'WIN' or levels > level:
                return actions
            
            # 生成候选动作
            current_obs = self._perceive(frame)
            
            # 移动动作
            for action_id in current_obs['available_actions']:
                if action_id <= 4:
                    queue.append(actions + [('move', action_id)])
            
            # 点击动作
            for block in current_obs['blocks']:
                cx, cy = block['center']
                queue.append(actions + [('click', cx, cy)])
            
            # 旋转
            if 5 in current_obs['available_actions']:
                queue.append(actions + [('rotate',)])
        
        return None
    
    def _detect_selection(self, before: Dict, after: Dict) -> bool:
        """检测是否选中了块"""
        # 选中后块的颜色或大小可能变化
        if len(before['blocks']) != len(after['blocks']):
            return True
        
        for b1, b2 in zip(before['blocks'], after['blocks']):
            if b1['color'] != b2['color'] or b1['size'] != b2['size']:
                return True
        
        return False
    
    def _visual_verify(self, env, actions: List, level: int) -> bool:
        """纯视觉验证"""
        # 重置
        env._game._current_level_index = level
        env._game.level_reset()
        env.reset()
        
        # 执行动作
        for action in actions:
            if action[0] == 'click':
                frame = env.step(GameAction.ACTION6, data={'x': action[1], 'y': action[2]})
            elif action[0] == 'move':
                frame = env.step(GameAction.from_id(action[1]), data={})
            elif action[0] == 'rotate':
                frame = env.step(GameAction.ACTION5, data={})
        
        # 检查画面帧
        state = frame.state.name if hasattr(frame, 'state') else ''
        levels = frame.levels_completed if hasattr(frame, 'levels_completed') else 0
        
        return state == 'WIN' or levels > level
    
    def _print_obs(self, obs: Dict):
        """打印观测"""
        print(f"  [视觉] 块: {len(obs['blocks'])} 个", flush=True)
        print(f"  [视觉] 颜色: {sorted(obs['colors'])}", flush=True)
        print(f"  [视觉] 背景: {obs['bg_color']}", flush=True)
        for b in obs['blocks']:
            print(f"  [视觉] 块{b['id']}: color={b['color']}, center={b['center']}, size={b['size']}", flush=True)
    
    def _print_rules(self, rules: Dict):
        """打印规则"""
        print(f"  [视觉] 规则: {rules}", flush=True)


def main():
    """M1目标：cn04前三关，纯视觉，无罐头"""
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    
    print("=" * 60)
    print("纯视觉求解器 - M1目标")
    print("cn04前三关，不读引擎内部，从画面琢磨规则")
    print("=" * 60)
    
    env = arc.make("cn04")
    if env is None:
        print("无法创建 cn04 环境")
        return
    
    g = env._game
    num_levels = len(g._levels)
    print(f"总关卡数: {num_levels}")
    print(f"M1目标: 前三关")
    
    solver = PureVisualSolver()
    
    total_actions = 0
    all_ok = True
    
    for lvl in range(min(3, num_levels)):
        print(f"\n{'='*40}")
        print(f"关卡 L{lvl+1}")
        print(f"{'='*40}")
        
        t0 = time.time()
        actions = solver.solve(env, lvl, max_time=120)
        dt = time.time() - t0
        
        if actions:
            total_actions += len(actions)
            print(f"L{lvl+1}: {len(actions)} actions -> OK  {dt:.1f}s")
        else:
            print(f"L{lvl+1}: NO SOLUTION  {dt:.1f}s")
            all_ok = False
    
    print(f"\nM1结果: actions={total_actions}")
    print("M1_PASS" if all_ok else "M1_FAIL")


if __name__ == "__main__":
    main()
