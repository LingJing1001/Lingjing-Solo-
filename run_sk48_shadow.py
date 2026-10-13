# -*- coding: utf-8 -*-
"""使用 arc_shadow_sk48.py 求解 sk48"""
import sys
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KAGGLE = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"
sys.path.insert(0, str(KAGGLE))
os.environ.setdefault("LINGJING_SRC", str(ROOT / "lingjing_solo"))
sys.path.insert(0, str(ROOT))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
import lingjing_solo.planning.arc_shadow_sk48 as shadow


def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("sk48")
    
    if env is None:
        print("无法创建 sk48 环境")
        return
    
    print("=== 使用 arc_shadow_sk48.py 求解 sk48 ===")
    
    # 初始化 shadow solver
    shadow._game = env._game
    shadow._env = env
    
    total_actions = 0
    all_ok = True
    
    # 检查关卡数
    g = env._game
    num_levels = len(g._levels)
    print(f"总关卡数: {num_levels}")
    
    for lvl in range(num_levels):
        t0 = time.time()
        
        # 设置关卡
        g._current_level_index = lvl
        g.level_reset()
        
        # 使用 shadow solver 求解
        sol = shadow.solve_level(t_limit=300.0)
        
        dt = time.time() - t0
        
        if sol is None:
            print(f"L{lvl+1}: NO SOLUTION  {dt:.1f}s")
            all_ok = False
            break
        
        if len(sol) == 0:
            print(f"L{lvl+1}: Already completed  {dt:.1f}s")
            continue
        
        # 验证
        g._current_level_index = lvl
        g.level_reset()
        
        for action_id in sol:
            g.perform_action(
                __import__('arcengine').ActionInput(
                    id=getattr(GameAction, f"ACTION{action_id}"),
                    data={},
                    reasoning=None
                ),
                raw=True
            )
        
        ok = int(g._current_level_index) > lvl or g._state.name == "WIN"
        total_actions += len(sol)
        
        print(f"L{lvl+1}: {len(sol)} actions -> levels={g._current_level_index}  "
              f"{'OK' if ok else 'FAIL'}  {dt:.1f}s")
        
        if not ok:
            all_ok = False
            break
    
    print(f"\nTotal: actions={total_actions}")
    print("ALL_PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
