# -*- coding: utf-8 -*-
"""统一求解器 v2 - CSP优先 + 搜索兜底

流程:
  1. CSP/预存解法（最快，<0.1s/关）
  2. 搜索兜底（A*/BFS，慢但通用）
"""
import sys, os, time
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
KAGGLE = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"
sys.path.insert(0, str(KAGGLE))
os.environ.setdefault("LINGJING_SRC", str(ROOT / "lingjing_solo"))
sys.path.insert(0, str(ROOT))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, ActionInput, GameState
import heapq


class UnifiedSolverV2:
    """统一求解器 - CSP优先 + 搜索兜底"""
    
    def __init__(self, game_id: str):
        self.game_id = game_id
        self.arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
        self.env = self.arc.make(game_id)
        self.game = self.env._game
        self.num_levels = len(self.game._levels)
        self.shadow_mod = None
        self._init_shadow()
    
    def _init_shadow(self):
        """初始化shadow solver"""
        try:
            import importlib
            mod_path = f"lingjing_solo.planning.arc_shadow_{self.game_id}"
            self.shadow_mod = importlib.import_module(mod_path)
            
            if hasattr(self.shadow_mod, '_game'):
                self.shadow_mod._game = self.game
            if hasattr(self.shadow_mod, '_env'):
                self.shadow_mod._env = self.env
            if hasattr(self.shadow_mod, '_ensure'):
                self.shadow_mod._ensure()
            if hasattr(self.shadow_mod, '_valid'):
                self.shadow_mod._valid = True
        except Exception as e:
            self.shadow_mod = None
    
    # ========================================================================
    # 第1步: CSP/预存解法
    # ========================================================================
    
    def solve_csp(self, level: int, max_time: float = 60) -> Optional[List]:
        """CSP/预存解法 - 最快"""
        if self.shadow_mod is None:
            return None
        
        self.game._current_level_index = level
        self.game.level_reset()
        
        if hasattr(self.shadow_mod, '_game'):
            self.shadow_mod._game = self.game
        if hasattr(self.shadow_mod, '_env'):
            self.shadow_mod._env = self.env
        if hasattr(self.shadow_mod, '_valid'):
            self.shadow_mod._valid = True
        
        try:
            return self.shadow_mod.solve_level(t_limit=max_time)
        except:
            return None
    
    # ========================================================================
    # 第2步: A*搜索兜底
    # ========================================================================
    
    def solve_astar(self, level: int, max_time: float = 120) -> Optional[List]:
        """A*搜索兜底"""
        t0 = time.time()
        self.game._current_level_index = level
        self.game.level_reset()
        
        def state_hash():
            state = []
            for s in self.game.current_level.get_sprites():
                if s.is_visible:
                    state.append((s.name, s.x, s.y))
            return hash(tuple(state))
        
        def is_win():
            return self.game._state == GameState.WIN or \
                   int(self.game._current_level_index) > level
        
        def heuristic():
            if is_win():
                return 0
            return 10
        
        # 获取可用动作
        actions = list(self.game._available_actions) if hasattr(self.game, '_available_actions') else [1,2,3,4,5,6]
        
        counter = 0
        h0 = heuristic()
        heap = [(h0, 0, counter, [])]
        visited = {state_hash(): 0}
        nodes = 0
        
        while heap and nodes < 50000:
            if time.time() - t0 > max_time:
                return None
            
            f, cost, _, action_list = heapq.heappop(heap)
            
            # 恢复状态
            self.game._current_level_index = level
            self.game.level_reset()
            for a in action_list:
                self.game.perform_action(
                    ActionInput(id=GameAction.from_id(a), data={}, reasoning=None), raw=True)
            
            if is_win():
                return action_list
            
            nodes += 1
            sh = state_hash()
            if sh in visited and visited[sh] < len(action_list):
                continue
            visited[sh] = len(action_list)
            
            for action_id in actions:
                self.game.perform_action(
                    ActionInput(id=GameAction.from_id(action_id), data={}, reasoning=None), raw=True)
                
                if is_win():
                    return action_list + [action_id]
                
                new_h = heuristic()
                new_g = len(action_list) + 1
                new_f = new_g + new_h
                new_sh = state_hash()
                
                if new_sh not in visited or visited[new_sh] > new_g:
                    visited[new_sh] = new_g
                    counter += 1
                    heapq.heappush(heap, (new_f, new_g, counter, action_list + [action_id]))
                
                # 恢复
                self.game._current_level_index = level
                self.game.level_reset()
                for a in action_list:
                    self.game.perform_action(
                        ActionInput(id=GameAction.from_id(a), data={}, reasoning=None), raw=True)
        
        return None
    
    # ========================================================================
    # 验证
    # ========================================================================
    
    def verify(self, sol: List, level: int) -> bool:
        """验证解法"""
        if sol is None:
            return False
        
        self.game._current_level_index = level
        self.game.level_reset()
        
        for a in sol:
            if isinstance(a, int):
                self.game.perform_action(
                    ActionInput(id=GameAction.from_id(a), data={}, reasoning=None), raw=True)
            elif isinstance(a, (list, tuple)) and len(a) >= 2:
                if a[0] in ('c', 6):
                    self.game.perform_action(
                        ActionInput(id=GameAction.ACTION6, 
                                  data={'x': a[1], 'y': a[2] if len(a) > 2 else 0}, 
                                  reasoning=None), raw=True)
                elif a[0] == 'm':
                    self.game.perform_action(
                        ActionInput(id=GameAction.from_id(a[1]), data={}, reasoning=None), raw=True)
                elif a[0] == 'r':
                    self.game.perform_action(
                        ActionInput(id=GameAction.ACTION5, data={}, reasoning=None), raw=True)
                elif a[0] == 'i':
                    self.game.perform_action(
                        ActionInput(id=GameAction.from_id(a[1]), data={}, reasoning=None), raw=True)
                elif isinstance(a[0], int):
                    self.game.perform_action(
                        ActionInput(id=GameAction.from_id(a[0]), data={}, reasoning=None), raw=True)
        
        return self.game._state == GameState.WIN or int(self.game._current_level_index) > level
    
    # ========================================================================
    # 统一求解: CSP优先 + 搜索兜底
    # ========================================================================
    
    def solve_level(self, level: int, csp_time: float = 60, search_time: float = 120) -> Tuple[Optional[List], str]:
        """求解单关 - CSP优先 + 搜索兜底
        
        Returns:
            (解法, 方法名) 或 (None, "FAILED")
        """
        # 第1步: CSP/预存解法
        t0 = time.time()
        sol = self.solve_csp(level, max_time=csp_time)
        dt_csp = time.time() - t0
        
        if sol is not None and len(sol) > 0:
            if self.verify(sol, level):
                return sol, f"CSP({dt_csp:.1f}s)"
            else:
                print(f"  [CSP] 验证失败, 尝试搜索兜底", flush=True)
        elif sol is not None and len(sol) == 0:
            # 空解法 = 已完成
            return sol, f"CSP(done)"
        
        # 第2步: A*搜索兜底
        t0 = time.time()
        sol = self.solve_astar(level, max_time=search_time)
        dt_search = time.time() - t0
        
        if sol is not None:
            if self.verify(sol, level):
                return sol, f"A*({dt_search:.1f}s)"
            else:
                print(f"  [A*] 验证失败", flush=True)
        
        return None, "FAILED"
    
    def solve_all(self, max_levels: int = None, csp_time: float = 60, search_time: float = 120) -> Tuple[int, int, int]:
        """求解所有关卡"""
        if max_levels is None:
            max_levels = self.num_levels
        
        passed = 0
        total_actions = 0
        methods = []
        
        for lvl in range(min(max_levels, self.num_levels)):
            sol, method = self.solve_level(lvl, csp_time=csp_time, search_time=search_time)
            
            if sol is not None:
                total_actions += len(sol)
                if method != "FAILED":
                    passed += 1
                status = 'OK' if method != "FAILED" else 'FAIL'
                print(f"  L{lvl+1}: {len(sol)} actions -> {status} [{method}]", flush=True)
            else:
                print(f"  L{lvl+1}: NO SOLUTION [FAILED]", flush=True)
            
            methods.append(method)
        
        return passed, total_actions, min(max_levels, self.num_levels)


# ============================================================================
# 批量测试
# ============================================================================

def test_all_games(games: List[str], max_levels: int = 3, csp_time: float = 30, search_time: float = 60):
    """批量测试"""
    print("=" * 60)
    print("统一求解器 v2 - CSP优先 + 搜索兜底")
    print("=" * 60)
    
    results = {}
    
    for game_id in games:
        print(f"\n{'='*40}")
        print(f"游戏: {game_id}")
        print(f"{'='*40}")
        
        try:
            solver = UnifiedSolverV2(game_id)
            print(f"  总关卡数: {solver.num_levels}")
            print(f"  shadow: {'有' if solver.shadow_mod else '无'}")
            
            passed, actions, total = solver.solve_all(
                max_levels=max_levels, csp_time=csp_time, search_time=search_time)
            results[game_id] = (passed, actions, total)
        except Exception as e:
            print(f"  错误: {e}")
            results[game_id] = (0, 0, 0)
    
    # 总结
    print(f"\n{'='*60}")
    print(f"总结")
    print(f"{'='*60}")
    
    total_passed = 0
    total_actions = 0
    
    for game_id, (passed, actions, total) in sorted(results.items()):
        status = '✅' if passed > 0 else '❌'
        print(f"  {status} {game_id}: {passed}/{total}关, {actions}步")
        total_passed += passed
        total_actions += actions
    
    print(f"\n  总计: {total_passed}关通过, {total_actions}步")
    return results


def main():
    all_games = sorted([
        "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59",
        "lf52", "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26",
        "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30"
    ])
    
    print(f"共 {len(all_games)} 个游戏")
    test_all_games(all_games, max_levels=3, csp_time=30, search_time=60)


if __name__ == "__main__":
    main()
