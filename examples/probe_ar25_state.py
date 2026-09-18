"""探查 AR25 env._game 内部结构，确认状态快照所需属性名。"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"F:/pro/Lingjing-Solo-/arc_adaptor")
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, ActionInput, GameState

arcade = Arcade(environments_dir=r"F:/pro/Lingjing-Solo-/environment_files",
                operation_mode=OperationMode.OFFLINE)
gid = [e.game_id for e in arcade.get_environments() if e.game_id.startswith("ar25")][0]
env = arcade.make(gid)
frame = env.reset()
g = env._game

# arc_shadow.py 用的属性名
attrs = ['ayyvxqrhnzw','yvifanjrcyu','lelsvjlwneo','ovoizfolxfq','ouurgkpbbjj',
         'hsiusrsrdkswnt','qehjebksqcm','hujpxmlafgh','xukxeewuexo','xjwpeqpcxav',
         '_state','_current_level_index','fswikrcrdmx','jtkyjqznbnp','naxbskjmlg','vplrhaovhr']
print("arc_shadow 属性名检查:")
for a in attrs:
    ok = hasattr(g, a)
    val = getattr(g, a, None)
    desc = type(val).__name__
    if ok and hasattr(val, '__len__'):
        desc += f" len={len(val)}"
    print(f"  {'✓' if ok else '✗'} g.{a}: {desc}")

# 测试 perform_action + 胜利检测
print(f"\n初始: level_idx={g._current_level_index} state={g._state}")
acts = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5]
won = g.vplrhaovhr()
print(f"vplrhaovhr() = {won}")

# 测试一次动作
g.perform_action(ActionInput(id=GameAction.ACTION1, data={}, reasoning=None), raw=True)
print(f"ACTION1后: level_idx={g._current_level_index} state={g._state}")
print(f"  current_steps={g.lelsvjlwneo.current_steps} ilqnjlrnkk={g.lelsvjlwneo.ilqnjlrnkk}")

# 目标格
targets = [(int(t.x), int(t.y)) for t in g.fswikrcrdmx]
print(f"\n目标格数: {len(targets)}")
print(f"目标格示例: {targets[:5]}")
