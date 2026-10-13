"""从 wa30_l5_plan.json 的 prefix 提取 L3/L4 分界。"""
import json, os, sys, copy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["OPERATION_MODE"] = "offline"
os.environ.setdefault("LINGJING_SRC", str(ROOT.parent / "lingjing_solo"))
sys.path.insert(0, str(ROOT.parent))

import arc_agi
from arc_agi import OperationMode
from arcengine import ActionInput as AI, GameAction as GA

arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
env = arc.make("wa30")
g = env._game

# L0-L2 罐头路线
plans = {
    0: [1, 1, 5, 1, 1, 5, 4, 4, 4, 1, 5, 2, 3, 3, 5, 3, 3, 3, 3, 3, 1, 5, 4, 4, 4, 5],
    1: [2, 2, 2, 2, 2, 1, 1, 4, 4, 4, 4, 4, 4, 5, 2, 2, 3, 3, 3, 3, 2, 3, 3, 5,
        1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 5, 3, 4],
    2: [1, 1, 1, 1, 4, 5, 4, 4, 4, 5, 3, 3, 3, 3, 3, 3, 1, 4, 5, 4, 4, 4, 4, 4, 4,
        5, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 2, 4, 5, 4, 4, 4, 4, 4, 5, 3, 4, 3, 4,
        3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3,
        4],
}

# 执行 L0-L2
for lv in range(3):
    for aid in plans[lv]:
        g.perform_action(AI(id=GA.from_id(aid), data={}), raw=True)

print(f"After L0-L2: level={g.level_index} state={g._state}")

# 读 prefix
import glob
for m in glob.glob("C:/newtask-pi/_diag_*/wa30_l5_plan.json", recursive=True):
    data = json.load(open(m, encoding="utf-8"))
    prefix = data.get("prefix", [])
    print(f"prefix total={len(prefix)}")

    # prefix[150:] 是 L3+L4
    remaining = prefix[150:]
    print(f"prefix[150:] = {len(remaining)} steps: {remaining}")

    # 逐手执行，检测换关
    g2 = copy.deepcopy(g)
    l3_plan = []
    l4_plan = []
    current_level = g2.level_index
    current_target = l3_plan

    for i, aid in enumerate(remaining):
        g2.perform_action(AI(id=GA.from_id(aid), data={}), raw=True)
        new_level = g2.level_index
        if new_level != current_level:
            print(f"  step {i}: level {current_level} -> {new_level} (action {aid})")
            current_level = new_level
            if current_level == 4:
                current_target = l4_plan
        current_target.append(aid)

    print(f"\nL3 plan: {len(l3_plan)} steps: {l3_plan}")
    print(f"L4 plan: {len(l4_plan)} steps: {l4_plan}")

    # 验证 L3 plan
    g3 = copy.deepcopy(g)
    for aid in l3_plan:
        g3.perform_action(AI(id=GA.from_id(aid), data={}), raw=True)
    print(f"\nVerify L3: level={g3.level_index} state={g3._state}")

    # 验证 L3+L4
    g4 = copy.deepcopy(g)
    for aid in remaining:
        g4.perform_action(AI(id=GA.from_id(aid), data={}), raw=True)
    print(f"Verify L3+L4: level={g4.level_index} state={g4._state}")
    break
