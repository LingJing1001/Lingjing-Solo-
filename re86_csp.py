# -*- coding: utf-8 -*-
"""re86 CSP 求解器 - 理解游戏机制后求解

游戏机制：
- 每次移动3格（ognwltlfta=3）
- ACTION1-4: 移动选中的块（上下左右各3格）
- ACTION5: 切换选中的块
- 边界检查：中心必须在画面内
- 碰撞：碰到障碍物时改变形状
- 目标：将块对齐到目标位置
"""
import sys, os, time
sys.path.insert(0, 'D:/Quan/Lingjing-Solo-main-bench')
os.environ.setdefault('LINGJING_SRC', 'D:/Quan/Lingjing-Solo-main-bench/lingjing_solo')
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, ActionInput, GameState
import numpy as np

STEP = 3  # 每次移动3格

def get_blocks(g):
    """获取可移动块"""
    return g.current_level.get_sprites_by_tag("0031cppcuvqlbi")

def get_targets(g):
    """获取目标"""
    return g.current_level.get_sprites_by_tag("0054xnsuqceejm")

def get_selected_block(g):
    """获取当前选中的块"""
    blocks = get_blocks(g)
    for s in blocks:
        center = s.pixels[s.height//2, s.width//2]
        if center == 0:  # ducxxbjjgd = 0
            return s
    return None

def get_block_center(s):
    """获取块的中心坐标"""
    return (s.x + s.width // 2, s.y + s.height // 2)

def apply_action(g, action_id):
    """应用动作"""
    g.perform_action(ActionInput(
        id=GameAction.from_id(action_id), data={}, reasoning=None), raw=True)

def is_win(g, level):
    """检查是否过关"""
    return g._state == GameState.WIN or int(g._current_level_index) > level

def solve_level_csp(g, level, max_time=120):
    """CSP求解 - 基于游戏规则"""
    t0 = time.time()
    
    g._current_level_index = level
    g.level_reset()
    
    # 分析初始状态
    blocks = get_blocks(g)
    targets = get_targets(g)
    selected = get_selected_block(g)
    
    print(f"  [CSP] 块: {len(blocks)}, 目标: {len(targets)}", flush=True)
    print(f"  [CSP] 选中: {selected.name if selected else 'None'}", flush=True)
    
    for s in blocks:
        cx, cy = get_block_center(s)
        print(f"  [CSP] 块{s.name}: pos=({s.x},{s.y}), center=({cx},{cy}), size=({s.width}x{s.height})", flush=True)
    
    for t in targets:
        print(f"  [CSP] 目标{t.name}: pos=({t.x},{t.y}), size=({t.width}x{t.height})", flush=True)
    
    # 策略：尝试所有可能的动作序列
    # 1. 切换到每个块
    # 2. 移动每个块到不同位置
    # 3. 检查是否过关
    
    # 尝试简单的动作序列
    # 策略1: 只移动选中的块
    for direction in [1, 2, 3, 4]:  # 上下左右
        for steps in range(1, 20):
            if time.time() - t0 > max_time:
                break
            
            g._current_level_index = level
            g.level_reset()
            
            actions = []
            for _ in range(steps):
                apply_action(g, direction)
                actions.append(direction)
            
            if is_win(g, level):
                print(f"  [CSP] FOUND: {len(actions)} steps (方向{direction} x{steps})", flush=True)
                return actions
    
    # 策略2: 切换+移动
    for switch_count in range(1, len(blocks)):
        for direction in [1, 2, 3, 4]:
            for steps in range(1, 15):
                if time.time() - t0 > max_time:
                    break
                
                g._current_level_index = level
                g.level_reset()
                
                actions = []
                # 切换
                for _ in range(switch_count):
                    apply_action(g, 5)
                    actions.append(5)
                
                # 移动
                for _ in range(steps):
                    apply_action(g, direction)
                    actions.append(direction)
                
                if is_win(g, level):
                    print(f"  [CSP] FOUND: {len(actions)} steps (切换{switch_count}+方向{direction} x{steps})", flush=True)
                    return actions
    
    # 策略3: 多方向组合
    for d1 in [1, 2, 3, 4]:
        for d2 in [1, 2, 3, 4]:
            for s1 in range(1, 10):
                for s2 in range(1, 10):
                    if time.time() - t0 > max_time:
                        break
                    
                    g._current_level_index = level
                    g.level_reset()
                    
                    actions = []
                    for _ in range(s1):
                        apply_action(g, d1)
                        actions.append(d1)
                    for _ in range(s2):
                        apply_action(g, d2)
                        actions.append(d2)
                    
                    if is_win(g, level):
                        print(f"  [CSP] FOUND: {len(actions)} steps ({d1}x{s1}+{d2}x{s2})", flush=True)
                        return actions
    
    # 策略4: 切换+多方向
    for switch in range(len(blocks)):
        for d1 in [1, 2, 3, 4]:
            for d2 in [1, 2, 3, 4]:
                for s1 in range(1, 8):
                    for s2 in range(1, 8):
                        if time.time() - t0 > max_time:
                            break
                        
                        g._current_level_index = level
                        g.level_reset()
                        
                        actions = []
                        for _ in range(switch):
                            apply_action(g, 5)
                            actions.append(5)
                        for _ in range(s1):
                            apply_action(g, d1)
                            actions.append(d1)
                        for _ in range(s2):
                            apply_action(g, d2)
                            actions.append(d2)
                        
                        if is_win(g, level):
                            print(f"  [CSP] FOUND: {len(actions)} steps (切换{switch}+{d1}x{s1}+{d2}x{s2})", flush=True)
                            return actions
    
    print(f"  [CSP] NO SOLUTION", flush=True)
    return None

def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("re86")
    g = env._game
    
    num_levels = len(g._levels)
    print(f"re86: {num_levels} 关")
    
    total_actions = 0
    all_ok = True
    
    for lvl in range(min(3, num_levels)):
        print(f"\n--- L{lvl+1} ---")
        t0 = time.time()
        
        sol = solve_level_csp(g, lvl, max_time=120)
        dt = time.time() - t0
        
        if sol:
            # 验证
            g._current_level_index = lvl
            g.level_reset()
            for a in sol:
                apply_action(g, a)
            
            ok = is_win(g, lvl)
            status = 'OK' if ok else 'FAIL'
            total_actions += len(sol)
            print(f"L{lvl+1}: {len(sol)} actions -> {status}  {dt:.1f}s")
            
            if not ok:
                all_ok = False
        else:
            print(f"L{lvl+1}: NO SOLUTION  {dt:.1f}s")
            all_ok = False
    
    print(f"\nTotal: actions={total_actions}")
    print("ALL_PASS" if all_ok else "FAIL")

if __name__ == "__main__":
    main()
