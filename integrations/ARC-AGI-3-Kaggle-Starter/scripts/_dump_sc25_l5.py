"""dump sc25 指定关的初始状态（grid + sprite tags），为 probe 做准备。"""
from __future__ import annotations
import sys, os
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

arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.NORMAL)
env = arc.make("sc25")
g = getattr(env, "_game", None)
plans = ma._SC25_PLANS

def do_action(g, item):
    aid, x, y = item
    if aid == 6:
        g.perform_action(ActionInput(id=GameAction.ACTION6, data={"x": int(x), "y": int(y)}), raw=True)
    else:
        g.perform_action(ActionInput(id=GameAction.from_id(aid), data={}), raw=True)

# 执行 L0-L4 plan 到达 L5
for lv in range(5):
    seq = plans.get(lv)
    if seq is None:
        break
    for item in seq:
        try:
            do_action(g, item)
        except Exception as e:
            print(f"L{lv} 异常: {e}")
            break

li = getattr(g, "level_index", -1)
print(f"到达 level_index={li} (L5)")
print()

# dump 当前关 grid
lv = g.current_level
grid = None
try:
    grid = lv.render() if hasattr(lv, "render") else None
except Exception:
    pass
if grid is None:
    # 尝试其他方式
    cam = getattr(g, "camera", None)
    if cam is not None:
        try:
            grid = cam.render(lv) if hasattr(cam, "render") else None
        except Exception:
            pass

# dump sprite 信息
print("=== sprites ===")
sps = lv.get_sprites() if hasattr(lv, "get_sprites") else []
for sp in sps:
    tags = getattr(sp, "tags", []) or []
    name = getattr(sp, "name", "?")
    x, y = getattr(sp, "x", -1), getattr(sp, "y", -1)
    w, h = getattr(sp, "width", 0), getattr(sp, "height", 0)
    color = getattr(sp, "color", "?")
    print(f"  {name:20} pos=({x},{y}) size={w}x{h} color={color} tags={list(tags)[:4]}")

print()
print(f"总 sprite 数: {len(sps)}")
# 预算
sc = getattr(getattr(g, "kuncbnslnm", None), "dbdarsgrbj", None)
print(f"步预算: {sc}")
