"""测 engine_solver 对 4 个零分游戏的效果。"""
import os, sys
from pathlib import Path
os.environ["OPERATION_MODE"] = "offline"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter" / "vendor"))
from lingjing_solo.engine_solver import EngineSolver

solver = EngineSolver(environments_dir="environment_files")
for gid in ["dc22", "re86", "s5i5", "tu93"]:
    print(f"\n=== {gid} ===")
    has = solver.has_solution(gid)
    sol = solver._solutions.get(gid)
    print(f"  has_solution: {has}, solution: {len(sol) if sol else 0} actions")
    if sol:
        print(f"  first 3: {sol[:3]}")
