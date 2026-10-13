# -*- coding: utf-8 -*-
"""分析 cn04 游戏状态 - 感知方块、障碍、目的"""
import sys
import os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KAGGLE = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"
sys.path.insert(0, str(KAGGLE))
os.environ.setdefault("LINGJING_SRC", str(ROOT / "lingjing_solo"))
sys.path.insert(0, str(ROOT))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

def make_env(gid="cn04"):
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    return arc.make(gid)

def analyze_game(env):
    """分析游戏状态"""
    frame = env.reset()
    g = env._game
    
    print("=" * 60)
    print("CN04 游戏分析")
    print("=" * 60)
    
    # 关卡信息
    print(f"\n当前关卡: {g.level_index + 1}")
    print(f"总关卡数: {8}")
    
    # 获取所有 sprite
    sprites = list(g.current_level.get_sprites())
    print(f"\nSprite 数量: {len(sprites)}")
    
    # 分析每个 sprite
    print("\n" + "-" * 60)
    print("Sprite 详情:")
    print("-" * 60)
    
    special_pixels_all = []
    
    for i, sprite in enumerate(sprites):
        if not sprite.is_visible:
            continue
            
        print(f"\n[Sprite {i+1}] {sprite.name}")
        print(f"  位置: ({sprite.x}, {sprite.y})")
        print(f"  旋转: {sprite.rotation}°")
        print(f"  尺寸: {sprite.width} x {sprite.height}")
        
        # 获取原始像素数据
        orig = g.hlxyvcmpk.get(sprite.name)
        if orig is not None:
            h, w = orig.shape
            
            # 找特殊像素 (8 和 13)
            specials = []
            for y in range(h):
                for x in range(w):
                    if orig[y, x] in (8, 13):
                        # 计算旋转后的全局坐标
                        rot = sprite.rotation
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
                        
                        gx, gy = int(sprite.x + rx), int(sprite.y + ry)
                        specials.append({
                            'local': (x, y),
                            'global': (gx, gy),
                            'type': '8' if orig[y, x] == 8 else '13'
                        })
            
            print(f"  特殊像素: {len(specials)} 个")
            for sp in specials:
                print(f"    - 类型 {sp['type']}: 局部({sp['local'][0]}, {sp['local'][1]}) -> 全局({sp['global'][0]}, {sp['global'][1]})")
                special_pixels_all.append({
                    'sprite': sprite.name,
                    'pos': sp['global'],
                    'type': sp['type']
                })
    
    # 分析匹配状态
    print("\n" + "-" * 60)
    print("匹配分析:")
    print("-" * 60)
    
    # 找重叠的特殊像素
    from collections import Counter
    pos_count = Counter([sp['pos'] for sp in special_pixels_all])
    
    matched = []
    unmatched = []
    
    for sp in special_pixels_all:
        if pos_count[sp['pos']] >= 2:
            matched.append(sp)
        else:
            unmatched.append(sp)
    
    print(f"已匹配的特殊像素: {len(matched)} 个")
    print(f"未匹配的特殊像素: {len(unmatched)} 个")
    
    # 找未匹配的 8 和 13
    unmatched_8 = [sp for sp in unmatched if sp['type'] == '8']
    unmatched_13 = [sp for sp in unmatched if sp['type'] == '13']
    
    print(f"\n未匹配的 8 型: {len(unmatched_8)} 个")
    for sp in unmatched_8[:5]:
        print(f"  - {sp['sprite']} @ {sp['pos']}")
    
    print(f"\n未匹配的 13 型: {len(unmatched_13)} 个")
    for sp in unmatched_13[:5]:
        print(f"  - {sp['sprite']} @ {sp['pos']}")
    
    # 计算匹配进度
    total_special = len(special_pixels_all)
    matched_pairs = len(matched) // 2  # 每对2个
    progress = matched_pairs / (total_special // 2) * 100 if total_special > 0 else 0
    
    print(f"\n匹配进度: {progress:.1f}% ({matched_pairs}/{total_special // 2} 对)")
    
    return {
        'sprites': sprites,
        'special_pixels': special_pixels_all,
        'matched': matched,
        'unmatched': unmatched,
        'progress': progress
    }

def main():
    env = make_env("cn04")
    if env is None:
        print("无法创建 cn04 环境")
        return
    
    # 分析第1关
    print("\n" + "=" * 60)
    print("第 1 关分析")
    print("=" * 60)
    analyze_game(env)
    
    # 分析第2关
    print("\n" + "=" * 60)
    print("第 2 关分析")
    print("=" * 60)
    env.reset()
    g = env._game
    g.next_level()  # 进入下一关
    analyze_game(env)

if __name__ == "__main__":
    main()
