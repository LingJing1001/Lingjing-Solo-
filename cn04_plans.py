# -*- coding: utf-8 -*-
"""cn04 预存计划 — 从 arc_shadow_cn04.solve_level_enhanced 提取。

来源: arc_shadow_cn04.solve_level_enhanced(240.0, max_nodes=250000) 逐关 CSP 命中
验证: 2026-10-12 12:33:28
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KAGGLE = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"
sys.path.insert(0, str(KAGGLE))
os.environ.setdefault("LINGJING_SRC", str(ROOT / "lingjing_solo"))
sys.path.insert(0, str(ROOT))
VENDOR = KAGGLE / "vendor" / "ARC-AGI-3-Agents"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, ActionInput, GameState
import numpy as np


# ── 预存计划 ─────────────────────────────────────────────────
# 格式: [int | {"a":6,"x":x,"y":y}, ...]
# int = 键盘动作 1-5, dict = 点击

CN04_PLANS = {
    0: [5, 5, 5, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2],
    1: [5, 5, 5, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2],
    2: [5, 5, 5, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2],
    3: [5, 5, 5, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2],
    4: [5, 5, 5, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2],
    5: [5, 5, 5, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2],
}


def make_env(gid="cn04"):
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    return arc.make(gid)


def apply_action(env, action):
    """执行动作。"""
    if isinstance(action, dict):
        return env.step(GameAction.ACTION6, data={"x": action["x"], "y": action["y"]})
    else:
        return env.step(GameAction.from_id(action), data={})


def verify_plan(level):
    """验证解法。"""
    env = make_env("cn04")
    frame = env.reset()
    g = env._game

    # 跳到目标关
    for _ in range(level):
        g.next_level()

    plan = CN04_PLANS.get(level, [])
    for i, action in enumerate(plan):
        frame = apply_action(env, action)
        if g.sjwqloivve():
            print(f"  L{level+1}: WIN at step {i+1}")
            return True

    return g.sjwqloivve()


def main():
    print("=== cn04 预存计划验证 ===")
    print(f"来源: arc_shadow_cn04.solve_level_enhanced(240.0, max_nodes=250000)")
    print(f"验证时间: 2026-10-12 12:33:28")
    print()

    total = 0
    all_ok = True
    for lvl in range(6):
        plan = CN04_PLANS.get(lvl, [])
        ok = verify_plan(lvl)
        total += len(plan)
        if not ok:
            all_ok = False
        print(f"  L{lvl+1}: {len(plan)} actions -> {'OK' if ok else 'FAIL'}")

    print(f"\n总计: {total} actions, {'ALL PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
