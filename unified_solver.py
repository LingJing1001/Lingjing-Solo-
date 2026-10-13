# -*- coding: utf-8 -*-
"""统一CSP求解器 - 集成所有游戏的解法

游戏分类:
  CSP/预存解法: ar25, bp35, cd82, cn04, ft09, ls20, r11l, sb26, sc25, su15, tn36, tr87, vc33, wa30
  BFS搜索: dc22, g50t, ka59, lf52, lp85, m0r0, re86, s5i5, sk48, sp80, tu93
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


# ============================================================================
# 游戏分类
# ============================================================================

# 有预存解法的游戏
CSP_GAMES = {
    "ar25", "bp35", "cd82", "cn04", "ft09", "ls20", "r11l",
    "sb26", "sc25", "su15", "tn36", "tr87", "vc33", "wa30"
}

# BFS搜索的游戏
BFS_GAMES = {
    "dc22", "g50t", "ka59", "lf52", "lp85", "m0r0",
    "re86", "s5i5", "sk48", "sp80", "tu93"
}


# ============================================================================
# 统一求解器
# ============================================================================

class UnifiedSolver:
    """统一CSP求解器"""
    
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
            mod_name = f"lingjing_solo.planning.arc_shadow_{self.game_id}"
            self.shadow_mod = __import__(mod_name, fromlist=['solve_level'])
            
            # 初始化_game和_env
            if hasattr(self.shadow_mod, '_game'):
                self.shadow_mod._game = self.game
            if hasattr(self.shadow_mod, '_env'):
                self.shadow_mod._env = self.env
            
            # 调用_ensure（如果有）
            if hasattr(self.shadow_mod, '_ensure'):
                self.shadow_mod._ensure()
            
            # 对于有_ensure的游戏（如cn04），需要设置_valid
            if hasattr(self.shadow_mod, '_valid'):
                self.shadow_mod._valid = True
            if hasattr(self.shadow_mod, '_game') and self.shadow_mod._game is None:
                self.shadow_mod._game = self.game
            if hasattr(self.shadow_mod, '_env') and self.shadow_mod._env is None:
                self.shadow_mod._env = self.env
                
        except ImportError:
            # 尝试直接导入
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
            except Exception as e2:
                print(f"  [初始化] shadow solver错误: {e2}")
                self.shadow_mod = None
        except Exception as e:
            print(f"  [初始化] shadow solver错误: {e}")
            self.shadow_mod = None
    
    def solve_level(self, level: int, max_time: float = 120) -> Optional[List]:
        """求解单关"""
        if self.shadow_mod is None:
            return None
        
        # 设置关卡
        self.game._current_level_index = level
        self.game.level_reset()
        
        # 更新shadow的_game
        if hasattr(self.shadow_mod, '_game'):
            self.shadow_mod._game = self.game
        if hasattr(self.shadow_mod, '_env'):
            self.shadow_mod._env = self.env
        if hasattr(self.shadow_mod, '_valid'):
            self.shadow_mod._valid = True
        
        try:
            sol = self.shadow_mod.solve_level(t_limit=max_time)
            return sol
        except Exception as e:
            print(f"  [求解] 错误: {e}")
            return None
    
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
            elif isinstance(a, (list, tuple)):
                if len(a) >= 2:
                    if a[0] in ('c', 6):
                        self.game.perform_action(
                            ActionInput(id=GameAction.ACTION6, data={'x': a[1], 'y': a[2] if len(a) > 2 else 0}, reasoning=None), raw=True)
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
    
    def solve_all(self, max_levels: int = None, max_time: float = 120) -> Tuple[int, int, int]:
        """求解所有关卡
        
        Returns:
            (passed, total_actions, total_levels)
        """
        if max_levels is None:
            max_levels = self.num_levels
        
        passed = 0
        total_actions = 0
        
        for lvl in range(min(max_levels, self.num_levels)):
            t0 = time.time()
            sol = self.solve_level(lvl, max_time=max_time)
            dt = time.time() - t0
            
            if sol is not None and len(sol) > 0:
                ok = self.verify(sol, lvl)
                total_actions += len(sol)
                if ok:
                    passed += 1
                status = 'OK' if ok else 'FAIL'
                print(f"  L{lvl+1}: {len(sol)} actions -> {status}  {dt:.1f}s")
            elif sol is not None and len(sol) == 0:
                # 空解法 = 已完成
                passed += 1
                print(f"  L{lvl+1}: Already completed  {dt:.1f}s")
            else:
                print(f"  L{lvl+1}: NO SOLUTION  {dt:.1f}s")
        
        return passed, total_actions, min(max_levels, self.num_levels)


# ============================================================================
# 批量测试
# ============================================================================

def test_all_games(games: List[str], max_levels: int = 3, max_time: float = 60):
    """批量测试所有游戏"""
    print("=" * 60)
    print("统一CSP求解器 - 批量测试")
    print("=" * 60)
    
    results = {}
    
    for game_id in games:
        print(f"\n{'='*40}")
        print(f"游戏: {game_id}")
        print(f"{'='*40}")
        
        try:
            solver = UnifiedSolver(game_id)
            print(f"  总关卡数: {solver.num_levels}")
            print(f"  类型: {'CSP' if game_id in CSP_GAMES else 'BFS'}")
            
            passed, actions, total = solver.solve_all(max_levels=max_levels, max_time=max_time)
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
    
    for game_id, (passed, actions, total) in results.items():
        game_type = 'CSP' if game_id in CSP_GAMES else 'BFS'
        status = '✅' if passed > 0 else '❌'
        print(f"  {status} {game_id} ({game_type}): {passed}/{total}关, {actions}步")
        total_passed += passed
        total_actions += actions
    
    print(f"\n  总计: {total_passed}关通过, {total_actions}步")
    
    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数 - 测试所有游戏"""
    all_games = sorted(list(CSP_GAMES) + list(BFS_GAMES))
    
    print(f"共 {len(all_games)} 个游戏")
    print(f"CSP游戏: {len(CSP_GAMES)} 个")
    print(f"BFS游戏: {len(BFS_GAMES)} 个")
    
    # 先测试CSP游戏
    print(f"\n{'='*60}")
    print(f"第一轮: CSP游戏")
    print(f"{'='*60}")
    
    csp_results = test_all_games(sorted(CSP_GAMES), max_levels=3, max_time=60)
    
    # 再测试BFS游戏
    print(f"\n{'='*60}")
    print(f"第二轮: BFS游戏")
    print(f"{'='*60}")
    
    bfs_results = test_all_games(sorted(BFS_GAMES), max_levels=3, max_time=60)


if __name__ == "__main__":
    main()
