"""诊断 LS20 L2: 执行罐头解后状态。"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"F:/pro/ARC-AGI-3-Agents")
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

_LEVEL_ACTIONS = {
    0: [3,3,3,1,1,1,1,4,4,4,1,1,1],
    1: [1,4,1,1,1,1,1,4,4,2,4,2,2,2,2,2,2,1,2,2,3,3,4,1,4,1,1,1,1,1,1,1,3,3,3,3,3,3,2,3,2,2,2,2,2],
    2: [1,1,1,1,1,1,1,1,3,2,2,2,2,2,2,2,2,1,1,1,3,3,1,4,4,4,4,4,4,4,1,1,1,3,1,2,1,4,2],
    3: [3,3,3,2,2,2,3,2,2,3,3,1,2,1,2,1,2,1,1,3,3,1,2,3,3,1,1,1,2,2,4,1,1,1,1,4,1,4,1,1,3,3,3],
    4: [1,4,1,1,3,4,3,3,3,4,3,4,3,4,4,2,2,3,3,3,1,3,3,3,4,4,2,2,2,2,2,4,4,2,4,4,4,1,4,4,2,2,2,1],
    5: [1,3,1,3,3,1,1,1,4,4,4,4,4,4,1,4,1,4,1,1,4,2,2,1,1,3,1,2,3,3,4,3,3,3,3,3,2,2,2,2,4,4,1,3,4,3,3,1,1,1,1,1,1,1,2,4,4,4,4,4,4,2,4,4,1,1,4,2,2,2,2,2],
    6: [1,1,2,2,3,3,2,2,2,2,2,1,2,4,2,1,4,1,2,1,2,1,2,1,2,3,3,1,1,1,4,4,4,4,1,4,4,1,4,4,1,1,4,2,2,3,3,3,1,2,2,2,2,2],
}

arcade = Arcade(environments_dir=r"F:/pro/ARC-AGI-3-Agents/environment_files", operation_mode=OperationMode.OFFLINE)
gid = [e.game_id for e in arcade.get_environments() if e.game_id.startswith("ls20")][0]
env = arcade.make(gid)

# 测 L2
level = 1
frame = env.reset()
g = env._game
g.set_level(level)
print(f"  set_level后: level={g._current_level_index}")
# 用 env.step 刷新
frame = env.step(GameAction.ACTION1, data=None)
print(f"  ACTION1后: 玩家=({g.gudziatsk.x},{g.gudziatsk.y}) state={g._state} level={g._current_level_index}")

# 用 env.step 执行
print(f"\n用 env.step 执行 {len(_LEVEL_ACTIONS[1])} 步:")
for i, a in enumerate(_LEVEL_ACTIONS[1]):
    try:
        frame = env.step(GameAction["ACTION%d"%a], data=None)
    except Exception as e:
        print(f"  步{i+1} ACTION{a} ERROR: {e}")
        break
    p = g.gudziatsk
    if i < 10 or i % 10 == 0:
        print(f"  步{i+1}: ACTION{a} 玩家=({p.x},{p.y}) state={g._state} level={g._current_level_index}")
print(f"\n最终: 玩家=({g.gudziatsk.x},{g.gudziatsk.y}) state={g._state} level={g._current_level_index}")
try:
    print(f"  pbznecvnfr() = {g.pbznecvnfr()}")
except: pass
