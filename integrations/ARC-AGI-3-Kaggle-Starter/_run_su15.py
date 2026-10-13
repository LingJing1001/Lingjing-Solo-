import sys, os, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LINGJING_SRC", str(ROOT.parent / "lingjing_solo"))
sys.path.insert(0, str(ROOT.parent))
os.environ["OPERATION_MODE"] = "offline"

from scripts.benchmark_all_games import run_all

report = run_all(
    max_steps=400,
    agent_path="agent/my_agent.py",
    game_filter={"su15"},
    out_json=Path("ui/static/su15_only.json"),
    agent_label="su15_only",
)
print("=== RESULT ===")
for g in report.get("games", []):
    print(f"  {g['game_id']} L={g.get('levels_completed',0)} score={g.get('game_score',0):.2f} act={g.get('actions',0)} {g.get('state','')} elapsed={g.get('elapsed_sec',0):.1f}s")
