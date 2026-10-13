import json, os, glob

# 1. Search external for L3/L4 plans
print("=== 外部 L3/L4 相关文件 ===")
for pattern in ["**/wa30_l3*", "**/wa30_l4*", "**/wa30_l34*"]:
    matches = glob.glob(f"C:/newtask-pi/_diag_*/{pattern}", recursive=True) + glob.glob(f"C:/newtask-pi/{pattern}", recursive=True)
    for m in sorted(set(matches)):
        print(f"  {os.path.basename(m)} ({os.path.getsize(m)} bytes)")

# 2. Check wa30_l5_plan.json prefix (218 steps to reach L4, so L3 is in there)
print("\n=== wa30_l5_plan.json prefix 分析 ===")
for m in glob.glob("C:/newtask-pi/_diag_*/wa30_l5_plan.json", recursive=True):
    data = json.load(open(m, encoding="utf-8"))
    prefix = data.get("prefix", [])
    plan = data.get("plan", [])
    print(f"  prefix len={len(prefix)}, plan len={len(plan)}")
    print(f"  prefix first 30: {prefix[:30]}")
    # The prefix should contain L0->L1->L2->L3->L4 transitions
    # L0 plan is 26 steps, L1 is 44, L2 is 75 = 145 total for L0-L2
    # So prefix[145:] should be L3+L4 actions
    print(f"  prefix[145:] (L3+L4): {prefix[145:]}")
    print(f"  prefix[145:] len={len(prefix[145:])}")

# 3. Check _WA30_PLANS in my_agent.py
print("\n=== 当前 _WA30_PLANS ===")
import sys
sys.path.insert(0, "integrations/ARC-AGI-3-Kaggle-Starter")
# Just read the plans directly
plans = {
    0: [1, 1, 5, 1, 1, 5, 4, 4, 4, 1, 5, 2, 3, 3, 5, 3, 3, 3, 3, 3, 1, 5, 4, 4, 4, 5],
    1: [2, 2, 2, 2, 2, 1, 1, 4, 4, 4, 4, 4, 4, 5, 2, 2, 3, 3, 3, 3, 2, 3, 3, 5,
        1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 5, 3, 4],
    2: [1, 1, 1, 1, 4, 5, 4, 4, 4, 5, 3, 3, 3, 3, 3, 3, 1, 4, 5, 4, 4, 4, 4, 4, 4,
        5, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 2, 4, 5, 4, 4, 4, 4, 4, 5, 3, 4, 3, 4,
        3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3,
        4],
}
for lv, p in plans.items():
    print(f"  L{lv}: {len(p)} steps")

# 4. Check markov model for L3
print("\n=== markov 转移表 L3 覆盖 ===")
p = "data/markov_wa30_transfers.json"
data = json.load(open(p, encoding="utf-8"))
states = data.get("states", {})
from collections import Counter
levels = Counter()
for k, v in states.items():
    if isinstance(v, dict):
        levels[v.get("level", -1)] += 1
print(f"  states by level: {dict(sorted(levels.items()))}")
print(f"  L3 states: {levels.get(3, 0)} (markov 只到 L2，L3 无覆盖)")

# 5. Check external L456 chain verify for L3 info
print("\n=== L456 chain verify output ===")
for m in glob.glob("C:/newtask-pi/_diag_*/wa30_l456_chain_verify_out.txt", recursive=True):
    with open(m, encoding="utf-8", errors="replace") as f:
        content = f.read()
    # Find L3/L4 mentions
    for line in content.split("\n"):
        if "L3" in line or "L4" in line or "0基3" in line or "0基4" in line or "319" in line or "prefix" in line.lower():
            print(f"  {line.strip()}")
