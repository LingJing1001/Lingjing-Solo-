import os, sys
from pathlib import Path
os.environ["OPERATION_MODE"] = "offline"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter" / "vendor"))
from lingjing_solo.engine_solver import EngineSolver

s = EngineSolver(environments_dir="environment_files")
print("has_solution(ar25):", s.has_solution("ar25"))
sol = s._solutions.get("ar25")
print(f"solution: {len(sol) if sol else 0} actions")
if sol:
    print(f"first 5: {sol[:5]}")
