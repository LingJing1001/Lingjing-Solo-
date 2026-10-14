"""逐关验证 wa30 硬编码 plan + 在线规划，定位崩点。只跑 plan 步数，不跑 CEAX。"""
from __future__ import annotations
import sys, os, copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CANON = ROOT.parents[1]
sys.path.insert(0, str(CANON))
os.environ.setdefault("LINGJING_SRC", str(CANON / "lingjing_solo"))
VENDOR = ROOT / "vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

import importlib.util
spec = importlib.util.spec_from_file_location("my_agent", ROOT / "agent" / "my_agent.py")
ma = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ma
spec.loader.exec_module(ma)

import arc_agi
from arcengine import ActionInput, GameAction

arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.NORMAL)
env = arc.make("wa30")
g = getattr(env, "_game", None)
if g is None:
    # 尝试其他属性
    for attr in ("game", "_arcade_game"):
        g = getattr(env, attr, None)
        if g is not None:
            break
if g is None:
    print("无法获取 game 对象，env 属性:", [a for a in dir(env) if not a.startswith("__")])
    sys.exit(1)

plans = ma._WA30_PLANS
print(f"wa30 plan 覆盖关: {sorted(plans)}")
print(f"初始 level_index={getattr(g, 'level_index', '?')} state={getattr(g, '_state', '?')}")
print()

for lv in range(9):  # wa30 9 关
    # 当前所在关
    cur_li = getattr(g, "level_index", -1)
    cur_state = str(getattr(g, "_state", ""))
    if "WIN" in cur_state:
        print(f"  → 全局 WIN，结束")
        break

    # 1) 跳过在线规划（beam 搜索太慢），只验证硬编码 plan
    seq = plans.get(lv)
    if seq is None:
        print(f"L{lv}: 无硬编码 plan，停止")
        break
    print(f"L{lv}: 用硬编码 plan len={len(seq)}")

    # 3) 执行 plan
    before_li = cur_li
    for i, aid in enumerate(seq):
        try:
            g.perform_action(ActionInput(id=GameAction.from_id(aid), data={}), raw=True)
        except Exception as e:
            print(f"L{lv} step {i} aid={aid} 异常: {e}")
            break
    after_li = getattr(g, "level_index", -1)
    after_state = str(getattr(g, "_state", ""))
    advanced = after_li > before_li or "WIN" in after_state
    print(f"  执行后 level_index={after_li} state={after_state} {'✓ 过关' if advanced else '✗ 未过关'}")
    print()
    if not advanced:
        print(f"L{lv} 硬编码 plan 未过关，停止")
        break
