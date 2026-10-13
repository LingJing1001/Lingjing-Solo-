"""验证 verified_solutions.py 里的 SU15_SOLUTIONS (L1-L9) 能否真机回放全通关。

不依赖任何外接路径（C:\\newtask-pi\\...），纯用仓库内解法 + arc_agi 引擎回放。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KAGGLE = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"
sys.path.insert(0, str(KAGGLE))
CANONICAL_ROOT = KAGGLE.parents[1]
os.environ.setdefault("LINGJING_SRC", str(CANONICAL_ROOT / "lingjing_solo"))
sys.path.insert(0, str(CANONICAL_ROOT))
VENDOR = KAGGLE / "vendor" / "ARC-AGI-3-Agents"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402

from lingjing_solo.planning.data.verified_solutions import SU15_SOLUTIONS  # noqa: E402


def main() -> None:
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("su15")
    if env is None:
        raise SystemExit("无法创建 su15 环境")

    fr = env.reset()
    print(f"reset: levels_completed={fr.levels_completed} state={fr.state}")

    # SU15_SOLUTIONS keys: "L1".."L9" → 0-indexed levels 0..8
    total_actions = 0
    all_ok = True
    for lvl_idx in range(9):
        key = f"L{lvl_idx + 1}"
        plan = SU15_SOLUTIONS.get(key)
        if not plan:
            print(f"  L{lvl_idx}: 无解法，跳过")
            all_ok = False
            continue
        before = fr.levels_completed
        for step_i, (x, y) in enumerate(plan):
            fr = env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})
            total_actions += 1
            if fr.state.name == "WIN":
                break
        after = fr.levels_completed
        status = "OK" if after > before else "FAIL"
        if after <= before:
            all_ok = False
        print(
            f"  L{lvl_idx} ({key}): {len(plan)} clicks → "
            f"levels {before}→{after}  {status}  state={fr.state.name}"
        )

    print(f"\n总计: levels_completed={fr.levels_completed}/9  actions={total_actions}  state={fr.state.name}")
    print("ALL_PASS" if (all_ok and fr.levels_completed >= 9) else "FAIL")


if __name__ == "__main__":
    main()
