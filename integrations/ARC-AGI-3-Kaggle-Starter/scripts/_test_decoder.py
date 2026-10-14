"""测试公共 plan_decoder：用 m0r0 L0 验证 DeepcopyBFS 策略。"""
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

from lingjing_solo.planning.plan_decoder import PlanDecoder, DeepcopyBFSStrategy

# m0r0: 用注册的 DeepcopyBFS
dec = PlanDecoder("m0r0")
dec.load()
print(f"m0r0 总关数={dec.num_levels}")
plan = dec.solve(0)
if plan is not None:
    print(f"\n✓ m0r0 L0 plan (公共解码器): {plan}")
    print(f"  验证: {dec.verify(0, plan)}")
else:
    print("\n✗ m0r0 L0 求解失败")
