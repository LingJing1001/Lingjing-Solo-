# -*- coding: utf-8 -*-
"""视觉感知模块 - 从画面帧琢磨规则，不读引擎内部

实现M1公约要求：从画面帧认出块/目标/选中态
"""
import sys
import os
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
KAGGLE = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"
sys.path.insert(0, str(KAGGLE))
os.environ.setdefault("LINGJING_SRC", str(ROOT / "lingjing_solo"))
sys.path.insert(0, str(ROOT))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, ActionInput


@dataclass
class VisualBlock:
    """视觉识别的块"""
    block_id: int
    color: int
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    size: int
    center: Tuple[int, int]
    pixels: List[Tuple[int, int]] = field(default_factory=list)


@dataclass 
class VisualObservation:
    """视觉观测状态"""
    blocks: List[VisualBlock] = field(default_factory=list)
    special_pixels: List[Dict] = field(default_factory=list)  # 特殊像素
    selected_block: Optional[int] = None  # 选中的块ID
    background_color: int = 0
    frame_hash: int = 0
    
    def hash(self) -> int:
        """状态哈希"""
        state = sorted([(b.block_id, b.bbox, b.color) for b in self.blocks])
        state.append(self.selected_block)
        return hash(tuple(state))


class VisualPerception:
    """视觉感知器 - 从画面帧识别游戏元素"""
    
    def __init__(self):
        self.history = []  # 历史帧
        self.special_colors = set()  # 已知的特殊颜色
    
    def perceive(self, frame_data) -> VisualObservation:
        """从画面帧识别游戏元素"""
        # 获取画面帧
        if hasattr(frame_data, 'frame'):
            f = frame_data.frame[0]
        else:
            f = frame_data
        
        arr = np.array(f)
        
        # 识别背景色（出现最多的颜色）
        colors, counts = np.unique(arr, return_counts=True)
        bg_color = colors[np.argmax(counts)]
        
        # 识别块（非背景色的连通区域）
        non_bg = arr != bg_color
        labeled, num_features = ndimage.label(non_bg)
        
        blocks = []
        special_pixels = []
        
        for i in range(1, num_features + 1):
            region = (labeled == i)
            color = int(arr[region][0])
            ys, xs = np.where(region)
            
            block = VisualBlock(
                block_id=i,
                color=color,
                bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                size=len(ys),
                center=(int(xs.mean()), int(ys.mean())),
                pixels=list(zip(xs.tolist(), ys.tolist()))
            )
            blocks.append(block)
            
            # 识别特殊像素（颜色8和13）
            if color in (8, 13):
                special_pixels.append({
                    'block_id': i,
                    'color': color,
                    'position': block.center,
                    'type': '8' if color == 8 else '13'
                })
        
        # 识别选中态（颜色0通常是选中态）
        selected = None
        for block in blocks:
            if block.color == 0:
                selected = block.block_id
                break
        
        obs = VisualObservation(
            blocks=blocks,
            special_pixels=special_pixels,
            selected_block=selected,
            background_color=int(bg_color),
            frame_hash=hash(arr.tobytes())
        )
        
        self.history.append(obs)
        return obs
    
    def detect_changes(self, before: VisualObservation, after: VisualObservation) -> Dict:
        """检测两个观测之间的变化"""
        changes = {
            'blocks_moved': [],
            'blocks_appeared': [],
            'blocks_disappeared': [],
            'special_pixels_changed': False,
            'selected_changed': before.selected_block != after.selected_block,
        }
        
        # 检测块移动
        before_blocks = {b.block_id: b for b in before.blocks}
        after_blocks = {b.block_id: b for b in after.blocks}
        
        for bid, b_block in before_blocks.items():
            if bid in after_blocks:
                a_block = after_blocks[bid]
                if b_block.bbox != a_block.bbox:
                    changes['blocks_moved'].append({
                        'block_id': bid,
                        'from': b_block.bbox,
                        'to': a_block.bbox
                    })
            else:
                changes['blocks_disappeared'].append(bid)
        
        for bid in after_blocks:
            if bid not in before_blocks:
                changes['blocks_appeared'].append(bid)
        
        # 检测特殊像素变化
        if len(before.special_pixels) != len(after.special_pixels):
            changes['special_pixels_changed'] = True
        
        return changes
    
    def discover_rules(self, env, max_experiments: int = 10) -> Dict:
        """通过小动作实验发现游戏规则"""
        print("  [视觉感知] 开始小动作实验...", flush=True)
        
        rules = {
            'actions': [],
            'block_movement': False,
            'has_selection': False,
            'has_special_pixels': False,
            'match_mechanism': 'unknown',
            'color_changes': {},  # 颜色变化记录
        }
        
        # 获取可用动作
        frame = env.reset()
        available = list(frame.available_actions) if hasattr(frame, 'available_actions') else [1,2,3,4,6]
        rules['actions'] = available
        
        # 初始观测
        before = self.perceive(frame)
        print(f"  [视觉感知] 初始: {len(before.blocks)} 块, {len(before.special_pixels)} 特殊像素", flush=True)
        
        # 获取初始帧的颜色分布
        if hasattr(frame, 'frame'):
            initial_arr = np.array(frame.frame[0])
            initial_colors = set(np.unique(initial_arr).tolist())
            print(f"  [视觉感知] 初始颜色: {sorted(initial_colors)}", flush=True)
        
        # 检查是否有特殊像素 - 通过颜色变化检测
        # 执行动作后，观察是否有新颜色出现
        for action_id in available[:4]:
            if action_id > 4:
                continue
            
            # 重置
            frame = env.reset()
            before = self.perceive(frame)
            
            # 执行动作
            try:
                frame = env.step(GameAction.from_id(action_id), data={})
                after = self.perceive(frame)
                
                # 检测变化
                changes = self.detect_changes(before, after)
                
                if changes['blocks_moved']:
                    rules['block_movement'] = True
                    print(f"  [视觉感知] 动作{action_id}: 块移动 {len(changes['blocks_moved'])} 个", flush=True)
                
                # 检测颜色变化
                if hasattr(frame, 'frame'):
                    after_arr = np.array(frame.frame[0])
                    after_colors = set(np.unique(after_arr).tolist())
                    new_colors = after_colors - initial_colors
                    if new_colors:
                        rules['color_changes'][action_id] = list(new_colors)
                        print(f"  [视觉感知] 动作{action_id}: 新颜色 {new_colors}", flush=True)
                        
                        # 检查是否有特殊颜色（3通常是匹配后的颜色）
                        if 3 in new_colors:
                            rules['has_special_pixels'] = True
                            rules['match_mechanism'] = 'special_pixel_match'
                
                # 如果有Undo，回退
                if 7 in available:
                    env.step(GameAction.ACTION7, data={})
                    
            except Exception as e:
                print(f"  [视觉感知] 动作{action_id} 失败: {e}", flush=True)
        
        # 检查选中态 - 通过颜色0检测
        if before.selected_block is not None:
            rules['has_selection'] = True
        
        # 如果没有发现特殊像素，但有多于2个块，可能是配对游戏
        if not rules['has_special_pixels'] and len(before.blocks) >= 2:
            rules['match_mechanism'] = 'block_match'
        
        # 重置
        env.reset()
        
        print(f"  [视觉感知] 规则发现: {rules}", flush=True)
        return rules


def main():
    """测试视觉感知模块"""
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    
    print("=" * 60)
    print("视觉感知模块测试 - CN04")
    print("=" * 60)
    
    env = arc.make("cn04")
    if env:
        perception = VisualPerception()
        rules = perception.discover_rules(env)
        
        print("\n发现的规则:")
        for key, value in rules.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("视觉感知模块测试 - SK48")
    print("=" * 60)
    
    env = arc.make("sk48")
    if env:
        perception = VisualPerception()
        rules = perception.discover_rules(env)
        
        print("\n发现的规则:")
        for key, value in rules.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
