# -*- coding: utf-8 -*-
"""批量测试多个游戏"""
import sys, os, time
sys.path.insert(0, 'D:/Quan/Lingjing-Solo-main-bench')
os.environ.setdefault('LINGJING_SRC', 'D:/Quan/Lingjing-Solo-main-bench/lingjing_solo')
from arc_game_api import GameAPI
from arcengine import GameAction, ActionInput, GameState
import numpy as np

def test_game(game_id, max_levels=3, max_time=60):
    """测试一个游戏"""
    print(f"\n{'='*60}")
    print(f"游戏: {game_id}")
    print(f"{'='*60}")
    
    try:
        api = GameAPI(game_id)
    except Exception as e:
        print(f"  无法创建: {e}")
        return game_id, 0, "ERROR"
    
    print(f"  总关卡数: {api.level_count()}")
    
    # 检查每关的sprite
    for lvl in range(min(api.level_count(), max_levels)):
        api.goto_level(lvl)
        blocks = api.get_blocks()
        targets = api.get_targets()
        collisions = api.get_collision_objects()
        actions = api.get_available_actions()
        print(f"  L{lvl+1}: blocks={len(blocks)}, targets={len(targets)}, collisions={len(collisions)}, actions={actions}")
    
    # 尝试用shadow solver
    total_actions = 0
    passed = 0
    
    for lvl in range(min(api.level_count(), max_levels)):
        t0 = time.time()
        sol = None
        
        # 尝试shadow solver
        try:
            module_name = f"lingjing_solo.planning.arc_shadow_{game_id}"
            mod = __import__(module_name, fromlist=['solve_level'])
            
            # 初始化
            if hasattr(mod, '_game') and hasattr(mod, '_env'):
                mod._game = api.game
                mod._env = api.env
            
            api.goto_level(lvl)
            sol = mod.solve_level(t_limit=max_time)
        except Exception as e:
            print(f"  L{lvl+1}: shadow solver错误: {e}")
        
        dt = time.time() - t0
        
        if sol:
            # 验证
            api.goto_level(lvl)
            for a in sol:
                if isinstance(a, int):
                    api.game.perform_action(
                        ActionInput(id=GameAction.from_id(a), data={}, reasoning=None), raw=True)
                elif isinstance(a, tuple) and a[0] == 'c':
                    api.game.perform_action(
                        ActionInput(id=GameAction.ACTION6, data={'x': a[1], 'y': a[2]}, reasoning=None), raw=True)
            
            ok = api.game._state == GameState.WIN or int(api.game._current_level_index) > lvl
            status = 'OK' if ok else 'FAIL'
            total_actions += len(sol)
            if ok:
                passed += 1
            print(f"  L{lvl+1}: {len(sol)} actions -> {status}  {dt:.1f}s")
        else:
            print(f"  L{lvl+1}: NO SOLUTION  {dt:.1f}s")
    
    result = "PASS" if passed > 0 else "FAIL"
    print(f"\n  {game_id}: {passed}/{min(api.level_count(), max_levels)} passed, {total_actions} actions")
    return game_id, passed, result


def main():
    games = ["dc22", "g50t", "ka59", "lf52"]
    results = []
    
    for game_id in games:
        game, passed, result = test_game(game_id, max_levels=3, max_time=60)
        results.append((game, passed, result))
    
    print(f"\n{'='*60}")
    print(f"总结")
    print(f"{'='*60}")
    for game, passed, result in results:
        status = '✅' if passed > 0 else '❌'
        print(f"  {status} {game}: {passed}关通过 ({result})")


if __name__ == "__main__":
    main()
