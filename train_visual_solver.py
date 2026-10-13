# -*- coding: utf-8 -*-
"""从 episode 数据训练纯视觉求解器

方法：
1. 规则提取 - 从画面帧变化中学习游戏规则
2. 状态-动作映射 - 建立画面帧到动作的映射
3. 模仿学习 - 从成功记录中学习策略
"""
import sys
import os
import json
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
from arcengine import GameAction


# ============================================================================
# 1. Episode 数据加载
# ============================================================================

class EpisodeLoader:
    """加载 episode 数据"""
    
    @staticmethod
    def load(episode_dir: Path) -> Dict:
        """加载一个 episode"""
        data = {
            'manifest': None,
            'scorecard': None,
            'recording': []
        }
        
        # 加载 manifest
        manifest_path = episode_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                data['manifest'] = json.load(f)
        
        # 加载 scorecard
        scorecard_path = episode_dir / "scorecard.json"
        if scorecard_path.exists():
            with open(scorecard_path) as f:
                data['scorecard'] = json.load(f)
        
        # 加载 recording
        recording_path = episode_dir / "recording.jsonl"
        if recording_path.exists():
            with open(recording_path) as f:
                for line in f:
                    data['recording'].append(json.loads(line))
        
        return data
    
    @staticmethod
    def load_all(episodes_dir: Path) -> List[Dict]:
        """加载所有 episode"""
        episodes = []
        
        for episode_dir in episodes_dir.iterdir():
            if not episode_dir.is_dir():
                continue
            
            data = EpisodeLoader.load(episode_dir)
            if data['recording']:
                episodes.append(data)
        
        return episodes


# ============================================================================
# 2. 规则提取 - 从画面帧变化中学习
# ============================================================================

class RuleExtractor:
    """从 episode 数据中提取游戏规则"""
    
    def __init__(self):
        self.rules = {
            'action_effects': {},  # 动作效果
            'color_changes': {},   # 颜色变化
            'block_movements': {}, # 块移动
            'match_rules': {},     # 匹配规则
        }
    
    def extract(self, episodes: List[Dict]) -> Dict:
        """从 episode 数据中提取规则"""
        print("  [训练] 提取规则...", flush=True)
        
        for ep in episodes:
            recording = ep['recording']
            
            for i in range(len(recording) - 1):
                step1 = recording[i].get('data', recording[i])
                step2 = recording[i + 1].get('data', recording[i + 1])
                
                # 获取画面帧
                frame1 = step1.get('frame', [])
                frame2 = step2.get('frame', [])
                
                # 获取动作
                action = step2.get('action_input', {})
                action_id = action.get('id', 'UNKNOWN')
                
                if not frame1 or not frame2:
                    continue
                
                # 分析画面变化
                self._analyze_change(frame1, frame2, action_id, i)
        
        # 总结规则
        self._summarize_rules()
        
        return self.rules
    
    def _analyze_change(self, frame1, frame2, action_id, step):
        """分析画面变化"""
        arr1 = np.array(frame1[0] if isinstance(frame1[0], list) else frame1)
        arr2 = np.array(frame2[0] if isinstance(frame2[0], list) else frame2)
        
        if arr1.shape != arr2.shape:
            return
        
        # 颜色变化
        colors1 = set(np.unique(arr1).tolist())
        colors2 = set(np.unique(arr2).tolist())
        new_colors = colors2 - colors1
        disappeared_colors = colors1 - colors2
        
        if new_colors or disappeared_colors:
            key = f"step{step}_{action_id}"
            self.rules['color_changes'][key] = {
                'action': action_id,
                'new_colors': list(new_colors),
                'disappeared_colors': list(disappeared_colors),
            }
        
        # 块移动
        bg1 = int(np.bincount(arr1.flatten()).argmax())
        non_bg1 = arr1 != bg1
        labeled1, num1 = ndimage.label(non_bg1)
        
        bg2 = int(np.bincount(arr2.flatten()).argmax())
        non_bg2 = arr2 != bg2
        labeled2, num2 = ndimage.label(non_bg2)
        
        if num1 != num2:
            key = f"step{step}_{action_id}"
            self.rules['block_movements'][key] = {
                'action': action_id,
                'blocks_before': num1,
                'blocks_after': num2,
            }
        
        # 动作效果
        if action_id not in self.rules['action_effects']:
            diff = np.sum(arr1 != arr2)
            self.rules['action_effects'][action_id] = {
                'avg_diff': diff,
                'count': 1,
            }
        else:
            effect = self.rules['action_effects'][action_id]
            effect['avg_diff'] = (effect['avg_diff'] * effect['count'] + np.sum(arr1 != arr2)) / (effect['count'] + 1)
            effect['count'] += 1
    
    def _summarize_rules(self):
        """总结规则"""
        # 总结动作效果
        for action, effect in self.rules['action_effects'].items():
            print(f"  [规则] {action}: 平均变化 {effect['avg_diff']:.0f} 像素 ({effect['count']} 次)", flush=True)
        
        # 总结颜色变化
        color_3_appeared = any(
            3 in change.get('new_colors', [])
            for change in self.rules['color_changes'].values()
        )
        
        if color_3_appeared:
            self.rules['match_rules']['color_3_match'] = "颜色3出现表示匹配成功"
            print("  [规则] 发现匹配规则: 颜色3 = 匹配成功", flush=True)
        
        # 总结块移动
        block_count_changed = any(
            change['blocks_before'] != change['blocks_after']
            for change in self.rules['block_movements'].values()
        )
        
        if block_count_changed:
            self.rules['match_rules']['block_count_change'] = "块数量变化表示游戏状态变化"
            print("  [规则] 发现块变化规则", flush=True)


# ============================================================================
# 3. 状态-动作映射 - 建立画面帧到动作的映射
# ============================================================================

class StateActionMapper:
    """状态-动作映射 - 使用特征匹配"""
    
    def __init__(self):
        self.features = []  # 特征列表 [(特征, 动作)]
    
    def _extract_features(self, frame_data) -> np.ndarray:
        """提取画面特征 - 颜色分布 + 块位置"""
        if not frame_data:
            return np.zeros(20)
        
        arr = np.array(frame_data[0] if isinstance(frame_data[0], list) else frame_data)
        
        # 颜色分布
        colors, counts = np.unique(arr, return_counts=True)
        color_dist = np.zeros(16)
        for c, cnt in zip(colors, counts):
            if 0 <= int(c) < 16:
                color_dist[int(c)] = cnt
        
        # 归一化
        total = color_dist.sum()
        if total > 0:
            color_dist = color_dist / total
        
        # 块数量
        bg = int(colors[np.argmax(counts)]) if len(colors) > 0 else 0
        non_bg = arr != bg
        labeled, num_blocks = ndimage.label(non_bg)
        
        # 块的中心位置（取前3个块）
        block_centers = []
        for i in range(1, min(num_blocks + 1, 4)):
            region = (labeled == i)
            where_result = np.where(region)
            if len(where_result) >= 2:
                ys, xs = where_result[0], where_result[1]
                if len(ys) > 0:
                    block_centers.extend([float(xs.mean()) / 64, float(ys.mean()) / 64])
        
        # 补齐到3个块
        while len(block_centers) < 6:
            block_centers.append(0)
        
        return np.concatenate([color_dist, [num_blocks / 10], block_centers[:6]])
    
    def train(self, episodes: List[Dict]):
        """从 episode 数据中学习状态-动作映射"""
        print("  [训练] 学习状态-动作映射...", flush=True)
        
        for ep in episodes:
            recording = ep['recording']
            
            for i in range(len(recording) - 1):
                step1 = recording[i].get('data', recording[i])
                step2 = recording[i + 1].get('data', recording[i + 1])
                
                frame = step1.get('frame', [])
                action = step2.get('action_input', {})
                
                if not frame:
                    continue
                
                features = self._extract_features(frame)
                action_id = action.get('id', 'UNKNOWN')
                action_data = action.get('data', {})
                
                self.features.append((features, {
                    'action': action_id,
                    'data': action_data,
                }))
        
        print(f"  [训练] 学会 {len(self.features)} 个状态-动作映射", flush=True)
    
    def predict(self, frame) -> Optional[Dict]:
        """预测动作 - 使用特征相似度匹配"""
        if not self.features:
            return None
        
        features = self._extract_features(frame)
        
        # 找最相似的状态
        best_match = None
        best_score = float('inf')
        
        for stored_features, action in self.features:
            # 欧氏距离
            dist = np.sum((features - stored_features) ** 2)
            
            if dist < best_score:
                best_score = dist
                best_match = action
        
        # 只在足够相似时才返回
        if best_score < 0.1:  # 阈值
            return best_match
        
        return None


# ============================================================================
# 4. 模仿学习 - 从成功记录中学习策略
# ============================================================================

class ImitationLearner:
    """模仿学习"""
    
    def __init__(self):
        self.state_action_mapper = StateActionMapper()
        self.rule_extractor = RuleExtractor()
        self.trained = False
    
    def train(self, episodes: List[Dict]):
        """训练"""
        print("=" * 60, flush=True)
        print("开始训练纯视觉求解器", flush=True)
        print("=" * 60, flush=True)
        
        # 1. 提取规则
        print("\n[步骤1] 规则提取", flush=True)
        rules = self.rule_extractor.extract(episodes)
        
        # 2. 学习状态-动作映射
        print("\n[步骤2] 状态-动作映射", flush=True)
        self.state_action_mapper.train(episodes)
        
        self.trained = True
        print(f"\n训练完成!", flush=True)
        print(f"  规则: {len(rules)} 条", flush=True)
        print(f"  映射: {len(self.state_action_mapper.features)} 个", flush=True)
    
    def solve(self, env, game_id: str, level: int) -> Optional[List]:
        """使用学到的知识求解"""
        if not self.trained:
            print("未训练!", flush=True)
            return None
        
        print(f"\n[求解] {game_id} L{level+1}", flush=True)
        
        # 设置关卡
        g = env._game
        g._current_level_index = level
        g.level_reset()
        frame = env.reset()
        
        actions = []
        max_steps = 50
        
        for step in range(max_steps):
            # 获取画面帧
            frame_data = frame.frame if hasattr(frame, 'frame') else []
            
            # 预测动作
            prediction = self.state_action_mapper.predict(frame_data)
            
            if prediction is None:
                print(f"  步骤{step}: 未知状态，无法预测", flush=True)
                break
            
            action_id = prediction['action']
            action_data = prediction['data']
            
            # 执行动作
            if action_id == 'ACTION1':
                frame = env.step(GameAction.ACTION1, data={})
                actions.append(('move', 1))
            elif action_id == 'ACTION2':
                frame = env.step(GameAction.ACTION2, data={})
                actions.append(('move', 2))
            elif action_id == 'ACTION3':
                frame = env.step(GameAction.ACTION3, data={})
                actions.append(('move', 3))
            elif action_id == 'ACTION4':
                frame = env.step(GameAction.ACTION4, data={})
                actions.append(('move', 4))
            elif action_id == 'ACTION5':
                frame = env.step(GameAction.ACTION5, data={})
                actions.append(('rotate',))
            elif action_id == 'ACTION6':
                x = action_data.get('x', 0)
                y = action_data.get('y', 0)
                frame = env.step(GameAction.ACTION6, data={'x': x, 'y': y})
                actions.append(('click', x, y))
            elif action_id == 'RESET':
                frame = env.reset()
                continue
            else:
                print(f"  步骤{step}: 未知动作 {action_id}", flush=True)
                break
            
            # 检查是否过关
            state = frame.state.name if hasattr(frame, 'state') else ''
            levels = frame.levels_completed if hasattr(frame, 'levels_completed') else 0
            
            if state == 'WIN' or levels > level:
                print(f"  步骤{step}: 过关!", flush=True)
                return actions
        
        return actions if actions else None


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数 - 训练和测试"""
    episodes_dir = ROOT / "data" / "aop" / "episodes"
    
    print("=" * 60)
    print("纯视觉求解器训练")
    print("=" * 60)
    
    # 1. 加载 episode 数据
    print("\n[1] 加载 episode 数据...")
    episodes = EpisodeLoader.load_all(episodes_dir)
    print(f"  加载了 {len(episodes)} 个 episode")
    
    # 统计数据
    total_steps = sum(len(ep['recording']) for ep in episodes)
    print(f"  总步数: {total_steps}")
    
    # 2. 训练
    learner = ImitationLearner()
    learner.train(episodes)
    
    # 3. 测试
    print("\n" + "=" * 60)
    print("[3] 测试训练结果")
    print("=" * 60)
    
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("cn04")
    
    if env:
        # 测试第1关
        actions = learner.solve(env, "cn04", 0)
        
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
