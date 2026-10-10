"""调试: engine_solver 单独测 lp85。"""
import os, sys
from pathlib import Path
os.environ["OPERATION_MODE"] = "offline"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter" / "vendor"))
from lingjing_solo.engine_solver import EngineSolver

solver = EngineSolver(environments_dir="environment_files")
print("has_solution(lp85):", solver.has_solution("lp85"))
print("has_solution(lp85-305b61c3):", solver.has_solution("lp85-305b61c3"))
sol = solver._solutions.get("lp85")
print(f"solution: {len(sol) if sol else 'None'} actions")
if sol:
    print(f"first 3: {sol[:3]}")
