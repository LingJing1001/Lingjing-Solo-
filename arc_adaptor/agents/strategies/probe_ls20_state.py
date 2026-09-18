"""探查 LS20 内部状态结构, 为 R2+R3+R4 适配做准备。"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"F:/pro/Lingjing-Solo-/arc_adaptor")
sys.path.insert(0, r"F:/pro/Lingjing-Solo-backup-20260916-125530/arc_adaptor")
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, ActionInput, GameState

arcade = Arcade(environments_dir=r"F:/pro/ARC-AGI-3-Agents/environment_files", operation_mode=OperationMode.OFFLINE)
gid = [e.game_id for e in arcade.get_environments() if e.game_id.startswith("ls20")][0]
env = arcade.make(gid)
env.reset()
g = env._game

print(f"type(g) = {type(g).__name__}")
print(f"level_idx = {g._current_level_index}, state = {g._state}")

# LS20 关键属性 (从 ls20.py 源码)
for attr in ['gudziatsk', 'gisrhqpee', 'tbwnoxqgc', 'htkmubhry', 'htkmubhry_2',
             'tsynhckng', 'srgbthxut', 'plrpelhym', 'lvrnuajbl', 'fwckfzsyc',
             'hiaauhahz', 'cklxociuu', 'aqygnziho', 'ebfuxzbvn', 'akoadfsur',
             'ltwrkifkx', 'zyoimjaei', 'wsoslqeku', 'hasivfwip', 'euemavvxz',
             'ofoahudlo', 'byotxmvkt', 'alsxlhizr']:
    try:
        val = getattr(g, attr, None)
        if val is not None:
            if hasattr(val, 'x') and hasattr(val, 'y'):
                print(f"  g.{attr}: Sprite at ({val.x},{val.y})")
            elif hasattr(val, '__len__'):
                print(f"  g.{attr}: {type(val).__name__} len={len(val)}")
            else:
                print(f"  g.{attr}: {type(val).__name__} = {val}")
    except Exception as e:
        print(f"  g.{attr}: ERROR")

# 玩家位置
try:
    p = g.gudziatsk
    print(f"\n玩家(gudziatsk): ({p.x},{p.y}) size=({p.width},{p.height})")
except: print("\n无 gudziatsk")

# 目标格
try:
    targets = g.current_level.get_sprites_by_tag("npxgalaybz")
    print(f"目标格(npxgalaybz): {len(targets)}个 {[(int(t.x),int(t.y)) for t in targets]}")
except: pass

# 步数
try:
    print(f"步数UI: {g._step_counter_ui.current_steps}/{g._step_counter_ui.osgviligwp}")
except: pass

# 胜利检测方法
for m in ['vplrhaovhr', 'cgj', 'pbznecvnfr']:
    try:
        val = getattr(g, m)()
        print(f"{m}() = {val}")
    except: pass

# 测试动作
print("\n测试 ACTION1 (UP):")
snap = [(s.x, s.y, s.pixels.copy()) for s in g.current_level._sprites]
g.perform_action(ActionInput(id=GameAction.ACTION1, data={}, reasoning=None), raw=True)
try:
    p = g.gudziatsk
    print(f"  玩家: ({p.x},{p.y}) state={g._state} level={g._current_level_index}")
except: pass
# 恢复
for s, (x, y, px) in zip(g.current_level._sprites, snap):
    s.set_position(x, y); s.pixels = px.copy()
