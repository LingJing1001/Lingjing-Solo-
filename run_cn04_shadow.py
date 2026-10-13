# -*- coding: utf-8 -*-
"""使用 arc_shadow_cn04.py 求解 cn04 - 使用 shadow._game 验证"""
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
import lingjing_solo.planning.arc_shadow_cn04 as shadow


def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("cn04")
    
    if env is None:
        print("无法创建 cn04 环境")
        return
    
    print("=== 使用 arc_shadow_cn04.py 求解 cn04 ===")
    
    # 初始化 shadow solver
    shadow._ensure()
    
    total_actions = 0
    all_ok = True
    
    for lvl in range(6):
        t0 = time.time()
        
        # 设置关卡
        g = shadow._game
        if g:
            g._current_level_index = lvl
            g.level_reset()
        
        # 使用 shadow solver 求解
        sol = shadow.solve_level(t_limit=30.0)
        
        dt = time.time() - t0
        
        if sol is None:
            print(f"L{lvl+1}: NO SOLUTION  {dt:.1f}s")
            all_ok = False
            break
        
        if len(sol) == 0:
            print(f"L{lvl+1}: Already completed  {dt:.1f}s")
            continue
        
        # 使用 shadow._game 验证
        g = shadow._game
        g._current_level_index = lvl
        g.level_reset()
        
        for a in sol:
            if a[0] == 'c':
                # 点击动作
                g.perform_action(shadow._ActionInput(id=shadow._ACTION[6], 
                                                     data={"x": int(a[1]), "y": int(a[2])}, 
                                                     reasoning=None), raw=True)
            elif a[0] == 'm':
                # 移动动作
                g.perform_action(shadow._ActionInput(id=shadow._ACTION[a[1]], 
                                                     data={}, 
                                                     reasoning=None), raw=True)
            elif a[0] == 'r':
                # 旋转动作
                g.perform_action(shadow._ActionInput(id=shadow._ACTION[5], 
                                                     data={}, 
                                                     reasoning=None), raw=True)
            elif a[0] == 'i':
                # 键盘动作 (i, n) -> ACTIONn
                g.perform_action(shadow._ActionInput(id=shadow._ACTION[a[1]], 
                                                     data={}, 
                                                     reasoning=None), raw=True)
        
        ok = int(g._current_level_index) > lvl or g._state == shadow._GameState.WIN
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
