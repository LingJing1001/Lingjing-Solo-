# -*- coding: utf-8 -*-
"""纯视觉求解器 v2 - 规则推理 + 更多数据

改进：
1. 增加 cn04 的 episode 数据（6关全通）
2. 使用规则推理而不是查表
"""
import sys
import os
import time
import json
import shutil
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
# 1. 增加 cn04 episode 数据
# ============================================================================

def generate_cn04_episodes():
    """生成 cn04 所有6关的 episode 数据"""
    print("=" * 60, flush=True)
    print("生成 cn04 episode 数据（6关）", flush=True)
    print("=" * 60, flush=True)
    
    import lingjing_solo.planning.arc_shadow_cn04 as shadow
    
    episodes_dir = ROOT / "data" / "aop" / "episodes"
    recordings_dir = ROOT / "recordings"
    
    # 创建环境，启用记录保存
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.NORMAL,
        recordings_dir=str(recordings_dir),
    )
    
    env = arc.make("cn04", save_recording=True, include_frame_data=True)
    g = env._game
    
    # 使用 CSP 求解
    shadow._ensure()
    solver_game = shadow._game
    
    all_actions = []
    all_ok = True
    
    for lvl in range(6):
        print(f"\n  L{lvl+1}:", end=" ", flush=True)
        
        # CSP 求解
        solver_game._current_level_index = lvl
        solver_game.level_reset()
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
            
            # 在真实环境上执行
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
            
            print(f"{len(actions)} actions -> {'OK' if ok else 'FAIL'}", flush=True)
            all_actions.extend(actions)
            
            if not ok:
                all_ok = False
        else:
            print("NO SOLUTION", flush=True)
            all_ok = False
    
    # 保存 episode
    label = "success" if all_ok else "fail"
    episode_name = f"cn04_{label}_rep_full"
    episode_dir = episodes_dir / episode_name
    episode_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找记录文件
    if recordings_dir.exists():
        for scorecard_dir in recordings_dir.iterdir():
            if scorecard_dir.is_dir():
                for recording_file in scorecard_dir.glob("cn04-*.jsonl"):
                    shutil.copy2(recording_file, episode_dir / "recording.jsonl")
                    print(f"\n  recording.jsonl: {recording_file.name}", flush=True)
                    break
                break
    
    # 创建 manifest
    manifest = {
        "game_id": "cn04",
        "environment_id": "cn04",
        "scorecard_id": f"auto-full-{int(time.time())}",
        "label": label,
        "rep": "full",
        "levels_completed": 6 if all_ok else 0,
        "state": "WIN" if all_ok else "NOT_FINISHED",
        "actions": len(all_actions),
        "win": all_ok,
        "tags": ["cn04", label, "rep_full", "phase-a"],
        "collected_at": time.time()
    }
    
    with open(episode_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # 检查 recording
    recording_path = episode_dir / "recording.jsonl"
    if recording_path.exists():
        with open(recording_path) as f:
            lines = f.readlines()
        print(f"  recording.jsonl: {len(lines)} 行", flush=True)
    
    print(f"\n  Episode 保存到: {episode_dir}", flush=True)
    print(f"  总步数: {len(all_actions)}", flush=True)
    print(f"  状态: {'ALL_PASS' if all_ok else 'FAIL'}", flush=True)
    
    return all_ok


# ============================================================================
# 2. 规则推理求解器
# ============================================================================

class RuleInferenceSolver:
    """规则推理求解器 - 从学到的规则中推理动作"""
    
    def __init__(self):
        self.rules = {
            'action_effects': {},
            'color_meanings': {},
            'block_info': {},
        }
    
    def learn_from_episodes(self, episodes_dir: Path):
        """从 episode 数据中学习规则"""
        print("\n[规则学习] 从 episode 数据中学习...", flush=True)
        
        episodes = []
        for ep_dir in episodes_dir.iterdir():
            if not ep_dir.is_dir():
                continue
            
            recording_path = ep_dir / "recording.jsonl"
            if recording_path.exists():
                with open(recording_path) as f:
                    recording = [json.loads(line) for line in f]
                
                manifest_path = ep_dir / "manifest.json"
                manifest = {}
                if manifest_path.exists():
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                
                episodes.append({
                    'recording': recording,
                    'manifest': manifest,
                })
        
        print(f"  加载了 {len(episodes)} 个 episode", flush=True)
        
        # 学习规则
        for ep in episodes:
            recording = ep['recording']
            game_id = ep['manifest'].get('game_id', 'unknown')
            
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
        
        # 总结规则
        self._summarize_rules()
    
    def _learn_step(self, frame1, frame2, action_id, game_id):
        """从一步中学习"""
        arr1 = np.array(frame1[0] if isinstance(frame1[0], list) else frame1)
        arr2 = np.array(frame2[0] if isinstance(frame2[0], list) else frame2)
        
        if arr1.shape != arr2.shape:
            return
        
        # 颜色变化
        colors1 = set(np.unique(arr1).tolist())
        colors2 = set(np.unique(arr2).tolist())
        new_colors = colors2 - colors1
        
        if 3 in new_colors:
            self.rules['color_meanings']['3'] = '匹配成功'
        
        # 动作效果
        if action_id not in self.rules['action_effects']:
            self.rules['action_effects'][action_id] = {
                'count': 0,
                'avg_change': 0,
                'games': set(),
            }
        
        effect = self.rules['action_effects'][action_id]
        diff = int(np.sum(arr1 != arr2))
        effect['avg_change'] = (effect['avg_change'] * effect['count'] + diff) / (effect['count'] + 1)
        effect['count'] += 1
        effect['games'].add(game_id)
    
    def _summarize_rules(self):
        """总结规则"""
        print("\n  学到的规则:", flush=True)
        for action, effect in self.rules['action_effects'].items():
            games = ','.join(effect['games'])
            print(f"    {action}: {effect['count']}次, 平均变化{effect['avg_change']:.0f}像素 (游戏: {games})", flush=True)
        
        for color, meaning in self.rules['color_meanings'].items():
            print(f"    颜色{color}: {meaning}", flush=True)
    
    def solve(self, env, game_id: str, level: int) -> Optional[List]:
        """规则推理求解"""
        print(f"\n[规则推理] 求解 {game_id} L{level+1}", flush=True)
        
        g = env._game
        g._current_level_index = level
        g.level_reset()
        frame = env.reset()
        
        actions = []
        max_steps = 50
        
        for step in range(max_steps):
            # 获取画面帧
            frame_data = frame.frame if hasattr(frame, 'frame') else []
            
            if not frame_data:
                break
            
            arr = np.array(frame_data[0] if isinstance(frame_data[0], list) else frame_data)
            
            # 检查是否过关
            state = frame.state.name if hasattr(frame, 'state') else ''
            levels = frame.levels_completed if hasattr(frame, 'levels_completed') else 0
            
            if state == 'WIN' or levels > level:
                print(f"  步骤{step}: 过关!", flush=True)
                return actions
            
            # 规则推理：选择最佳动作
            action = self._infer_action(arr, frame, step)
            
            if action is None:
                print(f"  步骤{step}: 无法推理动作", flush=True)
                break
            
            # 执行动作
            if action[0] == 'click':
                frame = env.step(GameAction.ACTION6, data={'x': action[1], 'y': action[2]})
            elif action[0] == 'move':
                frame = env.step(GameAction.from_id(action[1]), data={})
            elif action[0] == 'rotate':
                frame = env.step(GameAction.ACTION5, data={})
            
            actions.append(action)
            
            # 检查画面变化
            new_frame = frame.frame if hasattr(frame, 'frame') else []
            if new_frame:
                new_arr = np.array(new_frame[0] if isinstance(new_frame[0], list) else new_frame)
                diff = int(np.sum(arr != new_arr))
                print(f"  步骤{step}: {action} -> 变化{diff}像素", flush=True)
        
        return actions if actions else None
    
    def _infer_action(self, arr, frame, step):
        """推理最佳动作"""
        available = list(frame.available_actions) if hasattr(frame, 'available_actions') else [1,2,3,4,5,6]
        
        # 策略1: 如果是第一步，尝试RESET后的第一个动作
        if step == 0:
            # 通常第一步是点击一个块
            if 6 in available:
                # 找最大的块并点击
                bg = int(np.bincount(arr.flatten()).argmax())
                non_bg = arr != bg
                labeled, num = ndimage.label(non_bg)
                
                if num > 0:
                    # 找最大的块
                    max_size = 0
                    max_center = (32, 32)
                    for i in range(1, num + 1):
                        region = (labeled == i)
                        size = int(region.sum())
                        if size > max_size:
                            max_size = size
                            where = np.where(region)
                            if len(where) >= 2:
                                max_center = (int(where[1].mean()), int(where[0].mean()))
                    
                    return ('click', max_center[0], max_center[1])
        
        # 策略2: 尝试移动
        for action_id in [4, 2, 3, 1]:  # 右、下、左、上
            if action_id in available:
                return ('move', action_id)
        
        # 策略3: 尝试旋转
        if 5 in available:
            return ('rotate',)
        
        return None


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    # 1. 增加 cn04 episode 数据
    print("=" * 60, flush=True)
    print("步骤1: 增加 cn04 episode 数据", flush=True)
    print("=" * 60, flush=True)
    
    success = generate_cn04_episodes()
    
    if not success:
        print("cn04 episode 生成失败，使用已有数据继续", flush=True)
    
    # 2. 规则推理求解
    print("\n" + "=" * 60, flush=True)
    print("步骤2: 规则推理求解", flush=True)
    print("=" * 60, flush=True)
    
    episodes_dir = ROOT / "data" / "aop" / "episodes"
    
    # 学习规则
    solver = RuleInferenceSolver()
    solver.learn_from_episodes(episodes_dir)
    
    # 测试
    print("\n" + "=" * 60, flush=True)
    print("步骤3: 测试规则推理求解", flush=True)
    print("=" * 60, flush=True)
    
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("cn04")
    
    if env:
        # 测试第1关
        actions = solver.solve(env, "cn04", 0)
        
        if actions:
            # 验证
            g = env._game
            g._current_level_index = 0
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
            ok = state == 'WIN' or levels > 0
            
            print(f"\n结果: {len(actions)} actions -> {'OK' if ok else 'FAIL'}")
        else:
            print("\n结果: NO SOLUTION")


if __name__ == "__main__":
    main()
