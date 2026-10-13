import json, os
from pathlib import Path
from collections import Counter

# 1. Check markov transfers coverage
p = Path("data/markov_wa30_transfers.json")
if p.exists():
    data = json.loads(p.read_text(encoding="utf-8"))
    # Check structure
    if isinstance(data, dict):
        print(f"=== markov_wa30_transfers.json ===")
        print(f"keys: {list(data.keys())[:10]}")
        # Try to find level info
        states = data.get("states", {})
        transfers = data.get("transfers", {})
        print(f"states: {len(states)}, transfers: {len(transfers)}")
        # Check if states have level info
        if states:
            sample = list(states.items())[:3]
            for k, v in sample:
                print(f"  sample state: {k[:30]}... -> level={v.get('level', '?') if isinstance(v, dict) else '?'}")
        # Count by level
        levels = Counter()
        for k, v in states.items():
            if isinstance(v, dict):
                levels[v.get("level", -1)] += 1
            elif isinstance(v, (list, tuple)) and len(v) > 0:
                levels[v[0] if isinstance(v[0], int) else -1] += 1
        if levels:
            print(f"  states by level: {dict(sorted(levels.items()))}")
    else:
        print(f"unexpected type: {type(data)}")
else:
    print("transfers file not found")

# 2. Check external diag scripts for L4-L6
print("\n=== External diag scripts for L4-L6 ===")
import glob
for pattern in ["**/wa30_l4*", "**/wa30_l5*", "**/wa30_l6*", "**/wa30_l456*"]:
    matches = glob.glob(f"C:/newtask-pi/{pattern}", recursive=True)
    for m in matches:
        print(f"  {os.path.basename(m)} ({os.path.getsize(m)} bytes)")
