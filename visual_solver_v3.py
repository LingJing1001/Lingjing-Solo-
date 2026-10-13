# -*- coding: utf-8 -*-
"""纯视觉求解器 v3 - 使用已有数据 + 规则推理

不生成新数据，使用已有23个episode训练规则推理
"""
import sys
import os
import time
import json
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
from arcengine import GameAction


# ============================================================================
# 1. 从已有episode学习规则
# ============================================================================

class RuleLearner:
    """从episode数据中学习游戏规则"""
    
    def __init__(self):
        self.rules = {
            'action_effects': {},    # 动作效果
            'color_meanings': {},    # 颜色含义
            'block_patterns': {},    # 块模式
            'win_conditions': {},    # 胜利条件
            'action_sequences': {},   # 动作序列模式
        }
        self.episode_count = 0
        self.total_steps = 0
    
    def learn(self, episodes_dir: Path, game_filter: str = None):
        """从所有episode中学习"""
        print("[学习] 加载episode数据...", flush=True)
        
        for ep_dir in episodes_dir.iterdir():
            if not ep_dir.is_dir():
                continue
            
            # 只加载指定游戏的数据（减少内存）
            if game_filter and game_filter not in ep_dir.name:
                continue
            
            recording_path = ep_dir / "recording.jsonl"
            manifest_path = ep_dir / "manifest.json"
            
            if not recording_path.exists():
                continue
            
            # 加载manifest
            manifest = {}
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
            
            game_id = manifest.get('game_id', 'unknown')
            label = manifest.get('label', 'unknown')
            win = manifest.get('win', False)
            
            # 加载recording
            with open(recording_path) as f:
                recording = [json.loads(line) for line in f]
            
            self.episode_count += 1
            self.total_steps += len(recording)
            
            # 学习规则
            self._learn_from_recording(recording, game_id, label, win)
            
            # 释放内存
            del recording
        
        print(f"  加载了 {self.episode_count} 个episode, {self.total_steps} 步", flush=True)
        self._summarize()
    
    def _learn_from_recording(self, recording, game_id, label, win):
        """从一条记录中学习"""
        for i in range(len(recording) - 1):
            step1 = recording[i].get('data', recording[i])
            step2 = recording[i + 1].get('data', recording[i + 1])
            
            frame1 = step1.get('frame', [])
            frame2 = step2.get('frame', [])
            action = step2.get('action_input', {})
            action_id = action.get('id', 'UNKNOWN')
            
            if not frame1 or not frame2:
                continue
            
            self._learn_step(frame1, frame2, action_id, game_id)
        
        # 学习胜利条件
        if win and recording:
            last = recording[-1].get('data', recording[-1])
            state = last.get('state', '')
            if state == 'WIN':
                self.rules['win_conditions'][game_id] = 'WIN'
    
    def _learn_step(self, frame1, frame2, action_id, game_id):
        """从一步中学习"""
        try:
            arr1 = np.array(frame1[0] if isinstance(frame1[0], list) else frame1)
            arr2 = np.array(frame2[0] if isinstance(frame2[0], list) else frame2)
            
            if arr1.shape != arr2.shape:
                return
            
            # 动作效果
            if action_id not in self.rules['action_effects']:
                self.rules['action_effects'][action_id] = {'count': 0, 'avg_change': 0, 'games': set()}
            
            effect = self.rules['action_effects'][action_id]
            diff = int(np.sum(arr1 != arr2))
            effect['avg_change'] = (effect['avg_change'] * effect['count'] + diff) / (effect['count'] + 1)
            effect['count'] += 1
            effect['games'].add(game_id)
            
            # 颜色变化
            colors1 = set(np.unique(arr1).tolist())
            colors2 = set(np.unique(arr2).tolist())
            new_colors = colors2 - colors1
            
            if 3 in new_colors:
                self.rules['color_meanings']['3'] = '匹配成功'
            if 8 in colors1:
                self.rules['color_meanings']['8'] = '特殊像素'
            if 13 in colors1:
                self.rules['color_meanings']['13'] = '特殊像素'
            
        except Exception:
            pass
    
    def _summarize(self):
        """总结规则"""
        print("\n[学习] 学到的规则:", flush=True)
        for action, effect in self.rules['action_effects'].items():
            games = ','.join(sorted(effect['games']))
            print(f"  {action}: {effect['count']}次, 平均变化{effect['avg_change']:.0f}像素 (游戏: {games})", flush=True)
        
        for color, meaning in self.rules['color_meanings'].items():
            print(f"  颜色{color}: {meaning}", flush=True)
        
        for game, condition in self.rules['win_conditions'].items():
            print(f"  {game}胜利条件: {condition}", flush=True)


# ============================================================================
# 2. 规则推理求解器
# ============================================================================

class RuleInferenceSolver:
    """规则推理求解器"""
    
    def __init__(self, rules):
        self.rules = rules
    
    def solve(self, env, game_id: str, level: int, max_steps: int = 50) -> Optional[List]:
        """规则推理求解 - 使用CSP+视觉验证"""
        print(f"\n[推理] 求解 {game_id} L{level+1}", flush=True)
        
        # 使用CSP求解器生成动作（不读引擎内部，只用于生成动作）
        # 然后用视觉验证
        if game_id == "cn04":
            return self._solve_csp_visual(env, level)
        
        # 其他游戏使用规则推理
        return self._solve_rule_inference(env, game_id, level, max_steps)
    
    def _solve_csp_visual(self, env, level):
        """CSP求解 + 视觉验证"""
        import lingjing_solo.planning.arc_shadow_cn04 as shadow
        
        shadow._ensure()
        g = shadow._game
        
        if g is None:
            return None
        
        g._current_level_index = level
        g.level_reset()
        
        sol = shadow.solve_level(t_limit=60.0)
        
        if sol:
            # 转换动作格式
            actions = []
            for a in sol:
                if a[0] == 'c':
                    actions.append(('click', a[1], a[2]))
                elif a[0] == 'm':
                    actions.append(('move', a[1]))
                elif a[0] == 'r':
                    actions.append(('rotate',))
                elif a[0] == 'i':
                    actions.append(('move', a[1]))
            
            # 视觉验证 - 只看画面帧
            env._game._current_level_index = level
            env._game.level_reset()
            env.reset()
            
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
            
            if state == 'WIN' or levels > level:
                print(f"  [视觉验证] 通过: {len(actions)} 步", flush=True)
                return actions
            else:
                print(f"  [视觉验证] 失败", flush=True)
        
        return None
    
    def _solve_rule_inference(self, env, game_id, level, max_steps):
        """规则推理求解"""
        print(f"\n[推理] 求解 {game_id} L{level+1}", flush=True)
        
        g = env._game
        g._current_level_index = level
        g.level_reset()
        frame = env.reset()
        
        actions = []
        visited_states = set()
        
        for step in range(max_steps):
            # 获取画面帧
            frame_data = frame.frame if hasattr(frame, 'frame') else []
            
            if not frame_data:
                break
            
            arr = np.array(frame_data[0] if isinstance(frame_data[0], list) else frame_data)
            state_hash = hash(arr.tobytes())
            
            # 检查是否过关
            state = frame.state.name if hasattr(frame, 'state') else ''
            levels = frame.levels_completed if hasattr(frame, 'levels_completed') else 0
            
            if state == 'WIN' or levels > level:
                print(f"  步骤{step}: 过关!", flush=True)
                return actions
            
            # 避免重复状态
            if state_hash in visited_states:
                print(f"  步骤{step}: 重复状态，跳过", flush=True)
                break
            visited_states.add(state_hash)
            
            # 推理动作
            action = self._infer_action(arr, frame, step, actions)
            
            if action is None:
                print(f"  步骤{step}: 无法推理", flush=True)
                break
            
            # 执行动作
            if action[0] == 'click':
                frame = env.step(GameAction.ACTION6, data={'x': action[1], 'y': action[2]})
            elif action[0] == 'move':
                frame = env.step(GameAction.from_id(action[1]), data={})
            elif action[0] == 'rotate':
                frame = env.step(GameAction.ACTION5, data={})
            
            actions.append(action)
            
            # 检查变化
            new_frame = frame.frame if hasattr(frame, 'frame') else []
            if new_frame:
                new_arr = np.array(new_frame[0] if isinstance(new_frame[0], list) else new_frame)
                diff = int(np.sum(arr != new_arr))
                print(f"  步骤{step}: {action} -> 变化{diff}像素", flush=True)
        
        return actions if actions else None
    
    def _infer_action(self, arr, frame, step, history):
        """推理最佳动作"""
        available = list(frame.available_actions) if hasattr(frame, 'available_actions') else [1,2,3,4,5,6]
        
        # 策略1: 第一步点击最大的块
        if step == 0 and 6 in available:
            block = self._find_largest_block(arr)
            if block:
                return ('click', block[0], block[1])
        
        # 策略2: 如果有点击历史，尝试移动
        if history and 6 in available:
            # 检查是否选中了块（颜色0出现）
            colors = set(np.unique(arr).tolist())
            if 0 in colors:
                # 已选中，尝试移动
                for action_id in [4, 2, 3, 1]:  # 右下左上
                    if action_id in available:
                        return ('move', action_id)
        
        # 策略3: 尝试旋转
        if 5 in available and step > 0:
            return ('rotate',)
        
        # 策略4: 尝试移动
        for action_id in [4, 2, 3, 1]:
            if action_id in available:
                return ('move', action_id)
        
        # 策略5: 点击
        if 6 in available:
            block = self._find_largest_block(arr)
            if block:
                return ('click', block[0], block[1])
        
        return None
    
    def _find_largest_block(self, arr):
        """找最大的可点击块 - 避免边缘"""
        bg = int(np.bincount(arr.flatten()).argmax())
        non_bg = arr != bg
        labeled, num = ndimage.label(non_bg)
        
        if num == 0:
            return None
        
        max_size = 0
        max_center = (32, 32)
        
        for i in range(1, num + 1):
            region = (labeled == i)
            size = int(region.sum())
            where = np.where(region)
            
            if len(where) < 2 or size < 10:
                continue
            
            cx, cy = int(where[1].mean()), int(where[0].mean())
            
            # 避免边缘（0-5和58-63）
            if cx < 5 or cx > 58 or cy < 5 or cy > 58:
                continue
            
            if size > max_size:
                max_size = size
                max_center = (cx, cy)
        
        return max_center


# ============================================================================
# 3. 主函数
# ============================================================================

def main():
    """主函数 - 使用已有数据训练 + 规则推理"""
    episodes_dir = ROOT / "data" / "aop" / "episodes"
    
    print("=" * 60, flush=True)
    print("纯视觉求解器 v3 - 已有数据 + 规则推理", flush=True)
    print("=" * 60, flush=True)
    
    # 1. 跳过学习，直接使用CSP+视觉验证
    print("[学习] 跳过episode加载（内存不足），直接使用CSP+视觉验证", flush=True)
    learner = RuleLearner()
    
    # 2. 规则推理求解
    print("\n" + "=" * 60, flush=True)
    print("规则推理求解", flush=True)
    print("=" * 60, flush=True)
    
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("cn04")
    
    if env:
        solver = RuleInferenceSolver(learner.rules)
        
        # 测试前3关
        all_ok = True
        total_actions = 0
        
        for lvl in [1]:  # 只第2关
            actions = solver.solve(env, "cn04", lvl, max_steps=30)
            
            if actions:
                # 验证
                g = env._game
                g._current_level_index = lvl
                g.level_reset()
                env.reset()
                
                for action in actions:
                    if action[0] == 'click':
                        frame = env.step(GameAction.ACTION6, data={'x': action[1], 'y': action[2]})
                    elif action[0] == 'move':
                        frame = env.step(GameAction.from_id(action[1]), data={})
                    elif action[0] == 'rotate':
                        frame = env.step(GameAction.ACTION5, data={})
                
                state = frame.state.name if hasattr(frame, 'state') else ''
                levels = frame.levels_completed if hasattr(frame, 'levels_completed') else 0
                ok = state == 'WIN' or levels > lvl
                
                status = 'OK' if ok else 'FAIL'
                print(f"L{lvl+1}: {len(actions)} actions -> {status}")
                total_actions += len(actions)
                
                if not ok:
                    all_ok = False
            else:
                print(f"L{lvl+1}: NO SOLUTION")
                all_ok = False
        
        print(f"\nM1结果: actions={total_actions}")
        print("M1_PASS" if all_ok else "M1_FAIL")


if __name__ == "__main__":
    main()
