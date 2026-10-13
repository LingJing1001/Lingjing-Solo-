# -*- coding: utf-8 -*-
"""ARC游戏公共接口 - 封装常用操作，避免直接操作内部变量

使用方法:
    from arc_game_api import GameAPI
    
    api = GameAPI("re86")
    api.goto_level(3)           # 切换到L4
    api.list_sprites()          # 列出所有sprite
    api.find_by_tag("0007dtbisvazhv")  # 按tag查找
    api.check_win()             # 检查是否过关
    api.move(4)                # 移动
    api.click(32, 32)         # 点击
    api.reset()                # 重置当前关卡
"""
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent
KAGGLE = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"
sys.path.insert(0, str(KAGGLE))
os.environ.setdefault("LINGJING_SRC", str(ROOT / "lingjing_solo"))
sys.path.insert(0, str(ROOT))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, ActionInput, GameState
import numpy as np


class GameAPI:
    """ARC游戏公共接口"""
    
    def __init__(self, game_id: str):
        """初始化游戏"""
        self.game_id = game_id
        self.arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
        self.env = self.arc.make(game_id)
        self.game = self.env._game
        self.num_levels = len(self.game._levels)
    
    # ========================================================================
    # 关卡管理
    # ========================================================================
    
    def goto_level(self, level: int) -> None:
        """切换到指定关卡（0-based）
        
        注意：不会调用env.reset()，避免覆盖关卡设置
        """
        if level < 0 or level >= self.num_levels:
            raise ValueError(f"关卡{level}超出范围(0-{self.num_levels-1})")
        self.game._current_level_index = level
        self.game.level_reset()
    
    def current_level(self) -> int:
        """获取当前关卡索引"""
        return self.game._current_level_index
    
    def level_count(self) -> int:
        """获取总关卡数"""
        return self.num_levels
    
    def reset_level(self) -> None:
        """重置当前关卡"""
        self.game.level_reset()
    
    def next_level(self) -> bool:
        """进入下一关，返回是否成功"""
        if self.game._current_level_index + 1 < self.num_levels:
            self.game.next_level()
            return True
        return False
    
    # ========================================================================
    # Sprite查询
    # ========================================================================
    
    def list_sprites(self, visible_only: bool = False) -> List[Dict]:
        """列出所有sprite
        
        Args:
            visible_only: 是否只返回可见的sprite
        """
        sprites = self.game.current_level.get_sprites()
        result = []
        for s in sprites:
            if visible_only and not s.is_visible:
                continue
            result.append({
                'name': s.name,
                'x': s.x,
                'y': s.y,
                'width': s.width,
                'height': s.height,
                'rotation': s.rotation,
                'visible': s.is_visible,
                'tags': list(s.tags) if hasattr(s, 'tags') else [],
            })
        return result
    
    def find_by_tag(self, tag: str) -> List[Dict]:
        """按tag查找sprite"""
        sprites = self.game.current_level.get_sprites_by_tag(tag)
        return [
            {
                'name': s.name,
                'x': s.x,
                'y': s.y,
                'visible': s.is_visible,
                'tags': list(s.tags) if hasattr(s, 'tags') else [],
            }
            for s in sprites
        ]
    
    def get_sprite(self, name: str) -> Optional[Dict]:
        """按名称获取sprite"""
        for s in self.game.current_level.get_sprites():
            if s.name == name:
                return {
                    'name': s.name,
                    'x': s.x,
                    'y': s.y,
                    'width': s.width,
                    'height': s.height,
                    'rotation': s.rotation,
                    'visible': s.is_visible,
                    'tags': list(s.tags) if hasattr(s, 'tags') else [],
                    'pixels': s.pixels,
                }
        return None
    
    def get_blocks(self) -> List[Dict]:
        """获取可移动块（tag=0031cppcuvqlbi）"""
        return self.find_by_tag("0031cppcuvqlbi")
    
    def get_targets(self) -> List[Dict]:
        """获取目标（tag=0054xnsuqceejm）"""
        return self.find_by_tag("0054xnsuqceejm")
    
    def get_collision_objects(self) -> List[Dict]:
        """获取碰撞对象（tag=0007dtbisvazhv）"""
        return self.find_by_tag("0007dtbisvazhv")
    
    def get_selected_block(self) -> Optional[Dict]:
        """获取当前选中的块"""
        for s in self.game.current_level.get_sprites_by_tag("0031cppcuvqlbi"):
            if s.pixels[s.height//2, s.width//2] == 0:
                return {
                    'name': s.name,
                    'x': s.x,
                    'y': s.y,
                }
        return None
    
    # ========================================================================
    # 动作执行
    # ========================================================================
    
    def move(self, direction: int) -> None:
        """移动（1=上, 2=下, 3=左, 4=右）"""
        self.game.perform_action(
            ActionInput(id=GameAction.from_id(direction), data={}, reasoning=None),
            raw=True
        )
    
    def click(self, x: int, y: int) -> None:
        """点击坐标"""
        self.game.perform_action(
            ActionInput(id=GameAction.ACTION6, data={'x': x, 'y': y}, reasoning=None),
            raw=True
        )
    
    def rotate(self) -> None:
        """旋转/切换"""
        self.game.perform_action(
            ActionInput(id=GameAction.ACTION5, data={}, reasoning=None),
            raw=True
        )
    
    def undo(self) -> None:
        """撤销"""
        self.game.perform_action(
            ActionInput(id=GameAction.ACTION7, data={}, reasoning=None),
            raw=True
        )
    
    def execute(self, actions: List[Tuple]) -> None:
        """执行动作序列
        
        Args:
            actions: 动作列表，如 [('click', 32, 32), ('move', 4), ('rotate',)]
        """
        for action in actions:
            if action[0] == 'click':
                self.click(action[1], action[2])
            elif action[0] == 'move':
                self.move(action[1])
            elif action[0] == 'rotate':
                self.rotate()
            elif action[0] == 'undo':
                self.undo()
    
    # ========================================================================
    # 状态检查
    # ========================================================================
    
    def check_win(self) -> bool:
        """检查是否过关"""
        return self.game._state == GameState.WIN or \
               int(self.game._current_level_index) > self._check_level
    
    def get_state(self) -> str:
        """获取游戏状态"""
        return self.game._state.name if hasattr(self.game._state, 'name') else str(self.game._state)
    
    def get_levels_completed(self) -> int:
        """获取已完成关卡数"""
        return self.game._current_level_index
    
    def get_frame(self) -> np.ndarray:
        """获取当前画面帧"""
        frame = self.env.reset()
        if hasattr(frame, 'frame'):
            return np.array(frame.frame[0])
        return np.array([])
    
    def get_available_actions(self) -> List[int]:
        """获取可用动作"""
        return list(self.game._available_actions) if hasattr(self.game, '_available_actions') else []
    
    def get_step_count(self) -> int:
        """获取剩余步数"""
        if hasattr(self.game, 'xikvflgqgp'):
            return self.game.xikvflgqgp.current_steps
        return -1
    
    # ========================================================================
    # 便捷方法
    # ========================================================================
    
    def print_level_info(self, level: Optional[int] = None) -> None:
        """打印关卡信息"""
        if level is not None:
            self.goto_level(level)
        
        print(f"=== {self.game_id} L{self.current_level()+1} ===")
        print(f"状态: {self.get_state()}")
        print(f"步数: {self.get_step_count()}")
        print(f"可用动作: {self.get_available_actions()}")
        
        sprites = self.list_sprites()
        print(f"Sprite数: {len(sprites)}")
        for s in sprites:
            print(f"  {s['name']}: pos=({s['x']},{s['y']}), visible={s['visible']}, tags={s['tags']}")
        
        blocks = self.get_blocks()
        targets = self.get_targets()
        collisions = self.get_collision_objects()
        print(f"块: {len(blocks)}, 目标: {len(targets)}, 碰撞对象: {len(collisions)}")
        
        selected = self.get_selected_block()
        if selected:
            print(f"选中: {selected['name']} at ({selected['x']},{selected['y']})")
    
    def solve_and_verify(self, actions: List[Tuple], level: int) -> bool:
        """求解并验证
        
        Args:
            actions: 动作序列
            level: 关卡索引
        """
        self.goto_level(level)
        self.execute(actions)
        return self.check_win()


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 示例：检查re86所有关卡
    api = GameAPI("re86")
    print(f"总关卡数: {api.level_count()}")
    
    for lvl in range(api.level_count()):
        api.goto_level(lvl)
        
        sprites = api.list_sprites()
        blocks = api.get_blocks()
        targets = api.get_targets()
        collisions = api.get_collision_objects()
        
        print(f"L{lvl+1}: {len(sprites)} sprites, "
              f"blocks={len(blocks)}, targets={len(targets)}, collisions={len(collisions)}")
        
        if collisions:
            for c in collisions:
                print(f"  碰撞对象: {c['name']} at ({c['x']},{c['y']})")
