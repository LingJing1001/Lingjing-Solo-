"""通用 plan 逐关验证：wa30/sc25/bp35。只跑 plan 步数，定位崩点。"""
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
import numpy as np

GAME = sys.argv[1] if len(sys.argv) > 1 else "wa30"

arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.NORMAL)
env = arc.make(GAME)
g = getattr(env, "_game", None)
if g is None:
    for attr in ("game", "_arcade_game"):
        g = getattr(env, attr, None)
        if g is not None:
            break
if g is None:
    print("无法获取 game 对象")
    sys.exit(1)

# 取 plan 表
PLANS = {"wa30": ma._WA30_PLANS, "sc25": ma._SC25_PLANS, "bp35": ma._BP35_PLANS}
plans = PLANS.get(GAME)
if plans is None:
    print(f"无 {GAME} plan 表")
    sys.exit(1)

# 总关数
meta_path = list((ROOT / "environment_files" / GAME).rglob("metadata.json"))
total = 0
if meta_path:
    import json
    m = json.loads(meta_path[0].read_text(encoding="utf-8"))
    total = len(m.get("baseline_actions", []))

print(f"{GAME} plan 覆盖: {sorted(plans)}  总关数={total}")
print(f"初始 level_index={getattr(g, 'level_index', '?')}")
print()

def do_action(g, item):
    """执行一个 plan item。wa30=int, sc25/bp35=(aid,x,y)。"""
    if isinstance(item, int):
        g.perform_action(ActionInput(id=GameAction.from_id(item), data={}), raw=True)
    else:
        aid, x, y = item
        if aid == 6:
            act = GameAction.ACTION6
            ai = ActionInput(id=act, data={"x": int(x), "y": int(y)})
        else:
            ai = ActionInput(id=GameAction.from_id(aid), data={})
        g.perform_action(ai, raw=True)

for lv in range(total or len(plans)):
    cur_state = str(getattr(g, "_state", ""))
    if "WIN" in cur_state:
        print(f"  → 全局 WIN")
        break
    seq = plans.get(lv)
    if seq is None:
        print(f"L{lv}: 无 plan，停止")
        break
    before_li = getattr(g, "level_index", -1)
    for i, item in enumerate(seq):
        try:
            do_action(g, item)
        except Exception as e:
            print(f"L{lv} step {i} {item} 异常: {type(e).__name__}: {e}")
            break
    after_li = getattr(g, "level_index", -1)
    after_state = str(getattr(g, "_state", ""))
    advanced = after_li > before_li or "WIN" in after_state
    tag = "✓ 过关" if advanced else "✗ 未过关"
    print(f"L{lv}: plan len={len(seq)} → level_index={after_li} state={after_state} {tag}")
    if not advanced:
        print(f"L{lv} 未过关，停止")
        break
