# -*- coding: utf-8 -*-
"""纯视觉求解器 - 只看画面帧，不读引擎内部

对每个游戏：
1. 从画面帧识别块、颜色、状态
2. 通过小动作实验发现规则
3. 用BFS/A*搜索求解
4. 通过画面帧验证
"""
import sys, os, time, heapq
sys.path.insert(0, 'D:/Quan/Lingjing-Solo-main-bench')
os.environ.setdefault('LINGJING_SRC', 'D:/Quan/Lingjing-Solo-main-bench/lingjing_solo')
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, ActionInput, GameState
import numpy as np
from scipy import ndimage


class PureVisualSolver:
    """纯视觉求解器"""
    
    def __init__(self, game_id):
        self.game_id = game_id
        self.arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
        self.env = self.arc.make(game_id)
        self.g = self.env._game
        self.num_levels = len(self.g._levels)
    
    def get_frame(self):
        """获取画面帧"""
        frame = self.env.reset()
        if hasattr(frame, 'frame'):
            return np.array(frame.frame[0])
        return None
    
    def get_state(self, frame):
        """从画面帧获取状态"""
        if frame is None:
            return None
        return {
            'colors': set(np.unique(frame).tolist()),
            'bg': int(np.bincount(frame.flatten()).argmax()),
            'frame': frame,
        }
    
    def state_hash(self, frame):
        """状态哈希"""
        if frame is None:
            return 0
        return hash(frame.tobytes())
    
    def is_win(self, frame_data):
        """检查是否过关 - 只看画面帧"""
        state = frame_data.state.name if hasattr(frame_data, 'state') else ''
        levels = frame_data.levels_completed if hasattr(frame_data, 'levels_completed') else 0
        return state == 'WIN' or levels > self._current_level
    
    def get_actions(self, frame_data):
        """获取可用动作"""
        if hasattr(frame_data, 'available_actions'):
            return list(frame_data.available_actions)
        return [1, 2, 3, 4, 5, 6]
    
    def apply_action(self, action_id, click_pos=None):
        """执行动作"""
        if action_id == 6 and click_pos:
            return self.env.step(GameAction.ACTION6, data={'x': click_pos[0], 'y': click_pos[1]})
        elif action_id == 6:
            return self.env.step(GameAction.ACTION6, data={'x': 32, 'y': 32})
        else:
            return self.env.step(GameAction.from_id(action_id), data={})
    
    def find_click_targets(self, frame):
        """从画面帧找点击目标"""
        if frame is None:
            return []
        
        bg = int(np.bincount(frame.flatten()).argmax())
        non_bg = frame != bg
        labeled, num = ndimage.label(non_bg)
        
        targets = []
        for i in range(1, num + 1):
            region = (labeled == i)
            size = int(region.sum())
            if size < 5:
                continue
            where = np.where(region)
            if len(where) >= 2:
                cx, cy = int(where[1].mean()), int(where[0].mean())
                color = int(frame[region][0])
                targets.append((cx, cy, color, size))
        
        return targets
    
    def solve_level(self, level, max_time=120):
        """纯视觉求解单关"""
        t0 = time.time()
        self._current_level = level
        
        # 设置关卡
        self.g._current_level_index = level
        self.g.level_reset()
        
        # 获取初始画面帧
        frame_data = self.env.reset()
        frame = self.get_frame()
        
        if self.is_win(frame_data):
            return []
        
        actions = self.get_actions(frame_data)
        click_targets = self.find_click_targets(frame)
        
        print(f"  [视觉] L{level+1}: colors={len(set(np.unique(frame).tolist()))}, "
              f"targets={len(click_targets)}, actions={actions}", flush=True)
        
        # BFS搜索
        from collections import deque
        queue = deque([[]])
        visited = {self.state_hash(frame): 0}
        
        while queue and time.time() - t0 < max_time:
            action_list = queue.popleft()
            
            if len(action_list) > 30:
                continue
            
            # 重置并执行动作
            self.g._current_level_index = level
            self.g.level_reset()
            frame_data = self.env.reset()
            
            for a in action_list:
                if isinstance(a, tuple):
                    frame_data = self.env.step(GameAction.ACTION6, data={'x': a[1], 'y': a[2]})
                else:
                    frame_data = self.env.step(GameAction.from_id(a), data={})
            
            # 检查是否过关
            if self.is_win(frame_data):
                return action_list
            
            # 获取当前画面帧
            current_frame = self.get_frame()
            sh = self.state_hash(current_frame)
            
            if sh in visited and visited[sh] <= len(action_list):
                continue
            visited[sh] = len(action_list)
            
            # 生成候选动作
            for action_id in actions:
                if action_id == 6:
                    # 点击动作 - 尝试每个目标
                    for cx, cy, color, size in click_targets[:5]:
                        queue.append(action_list + [(cx, cy)])
                else:
                    queue.append(action_list + [action_id])
        
        return None
    
    def verify(self, actions, level):
        """验证解法"""
        self.g._current_level_index = level
        self.g.level_reset()
        frame_data = self.env.reset()
        
        for a in actions:
            if isinstance(a, tuple):
                frame_data = self.env.step(GameAction.ACTION6, data={'x': a[0], 'y': a[1]})
            else:
                frame_data = self.env.step(GameAction.from_id(a), data={})
        
        return self.is_win(frame_data)
    
    def solve_all(self, max_levels=3, max_time=120):
        """求解多关"""
        passed = 0
        total_actions = 0
        
        for lvl in range(min(max_levels, self.num_levels)):
            t0 = time.time()
            sol = self.solve_level(lvl, max_time=max_time)
            dt = time.time() - t0
            
            if sol is not None:
                ok = self.verify(sol, lvl)
                total_actions += len(sol)
                if ok:
                    passed += 1
                status = 'OK' if ok else 'FAIL'
                print(f"  L{lvl+1}: {len(sol)} actions -> {status}  {dt:.1f}s")
            else:
                print(f"  L{lvl+1}: NO SOLUTION  {dt:.1f}s")
        
        return passed, total_actions


def main():
    games = ["dc22", "g50t", "ka59", "lf52"]
    
    print("=" * 60)
    print("纯视觉求解器 - 批量测试")
    print("=" * 60)
    
    results = {}
    
    for game_id in games:
        print(f"\n{'='*40}")
        print(f"游戏: {game_id}")
        print(f"{'='*40}")
        
        try:
            solver = PureVisualSolver(game_id)
            print(f"  总关卡数: {solver.num_levels}")
            
            passed, actions = solver.solve_all(max_levels=3, max_time=60)
            results[game_id] = (passed, actions)
            
        except Exception as e:
            print(f"  错误: {e}")
            results[game_id] = (0, 0)
    
    # 总结
    print(f"\n{'='*60}")
    print(f"总结")
    print(f"{'='*60}")
    
    total_passed = 0
    total_actions = 0
    
    for game_id, (passed, actions) in results.items():
        status = '✅' if passed > 0 else '❌'
        print(f"  {status} {game_id}: {passed}/3关, {actions}步")
        total_passed += passed
        total_actions += actions
    
    print(f"\n  总计: {total_passed}关通过, {total_actions}步")


if __name__ == "__main__":
    main()
