"""诊断：课题探索 agent 是否加载正确、动作是否生效。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEAM = ROOT.parent / "灵境战队AGI课题探索"
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(ROOT))

import arc_agi
from arc_agi import OperationMode

spec = importlib.util.spec_from_file_location("ta", ROOT / "agent" / "lingjing_team_agent.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import lingjing_solo

print("=== 加载路径 ===")
print("lingjing_solo:", lingjing_solo.__file__)
print("课题探索包:", TEAM / "lingjing_solo")
ls20 = Path(lingjing_solo.__file__).parent / "planning" / "ls20_solver.py"
print("有 ls20_solver?", ls20.exists(), "(课题探索版应无)")

from agent.lingjing_team_agent import _extract_grid, _valid_to_solo

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make("ls20")
ag = mod.MyAgent(
    card_id="diag",
    game_id="ls20",
    agent_name="diag",
    ROOT_URL="x",
    record=False,
    arc_env=env,
    tags=[],
)
ag.MAX_ACTIONS = 40

print("\n=== 前 15 步动作 ===")
acts = []
for i in range(15):
    f = ag.frames[-1]
    g = _extract_grid(f)
    a = ag.choose_action(ag.frames, f)
    reason = getattr(a, "reasoning", None)
    print(
        f"{i:02d} state={f.state.name if hasattr(f.state,'name') else f.state} "
        f"L={f.levels_completed} grid={'OK '+str(g.shape) if g is not None else 'NONE'} "
        f"solo_valid={_valid_to_solo(f)} -> {a.name}"
    )
    acts.append(a.name)
    frame = ag.take_action(a)
    ag.append_frame(frame)
    ag.action_counter += 1

print("\n动作序列:", acts)
print("最终 levels:", ag.frames[-1].levels_completed, "state:", ag.frames[-1].state)

# 对比 Starter：清掉课题探索缓存后再加载
print("\n=== 对比 Starter my_agent 前 25 步 ===")
for k in list(sys.modules):
    if k == "lingjing_solo" or k.startswith("lingjing_solo."):
        del sys.modules[k]
# 让 starter 的 lingjing_solo 优先
sys.path = [p for p in sys.path if "灵境战队AGI课题探索" not in p]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

spec2 = importlib.util.spec_from_file_location("ma", ROOT / "agent" / "my_agent.py")
mod2 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(mod2)
import lingjing_solo as ls2

print("Starter lingjing_solo:", ls2.__file__)
print("Starter 有 ls20_solver?", (Path(ls2.__file__).parent / "planning" / "ls20_solver.py").exists())

arc2 = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env2 = arc2.make("ls20")
ag2 = mod2.MyAgent(
    card_id="diag2",
    game_id="ls20",
    agent_name="diag2",
    ROOT_URL="x",
    record=False,
    arc_env=env2,
    tags=[],
)
ag2.MAX_ACTIONS = 40
acts2 = []
for i in range(25):
    f = ag2.frames[-1]
    if ag2.is_done(ag2.frames, f):
        break
    a = ag2.choose_action(ag2.frames, f)
    acts2.append(a.name)
    frame = ag2.take_action(a)
    ag2.append_frame(frame)
    ag2.action_counter += 1
print("Starter 动作:", acts2)
print("Starter levels:", ag2.frames[-1].levels_completed, "state:", ag2.frames[-1].state)
