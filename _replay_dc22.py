"""重放 feat-bfs 的 dc22 通关解法。"""
import os, sys, json
from pathlib import Path
import numpy as np

os.environ["OPERATION_MODE"] = "offline"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter" / "vendor"))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

# 读解法
sol_path = Path(r"D:\Quan\Lingjing-Solo--feat-bfs-depth20-hardbones\state\dc22_full_solution_1007.json")
sol = json.loads(sol_path.read_text(encoding="utf-8"))
print(f"dc22 解法: {sol['steps']} 步, verified={sol['verified']}")
print(f"动作: {sol['actions'][:5]}...")

# 重放
arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
env = arc.make("dc22")
frame = env.reset()
print(f"初始: levels={frame.levels_completed} state={frame.state}")

ACTION_MAP = {"A1": 1, "A2": 2, "A3": 3, "A4": 4, "A5": 5, "A6": 6, "A7": 7,
              "CLICK_A": 6, "CLICK_B": 6, "CLICK_C": 6, "CLICK_D": 6}

for i, action in enumerate(sol["actions"]):
    name, x, y = action
    aid = ACTION_MAP.get(name, 6)
    data = {}
    if x is not None and y is not None:
        data = {"x": int(x), "y": int(y)}
    elif name.startswith("CLICK"):
        data = {"x": 32, "y": 32}  # 兜底

    game_action = GameAction.from_id(aid)
    game_action.reasoning = {"text": f"replay {name}"}
    try:
        frame = env.step(game_action, data=data, reasoning={"text": name})
    except Exception as e:
        print(f"  step {i} err: {e}")
        break

    if i < 3 or frame.levels_completed > 0 or str(frame.state).upper() == "WIN":
        print(f"  step {i}: {name} ({x},{y}) -> levels={frame.levels_completed} state={frame.state}")

    if str(frame.state).upper() == "WIN":
        print(f"  *** WIN at step {i+1}! ***")
        break

print(f"\n最终: levels={frame.levels_completed} state={frame.state} steps={i+1}")
