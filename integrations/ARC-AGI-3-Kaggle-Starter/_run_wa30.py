import sys, os, json, time
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
    game_filter={"wa30"},
    out_json=Path("ui/static/wa30_only.json"),
    agent_label="wa30_only",
)
print("=== SUMMARY ===")
print(json.dumps({k: report[k] for k in ("total_games", "games_won", "total_levels_completed", "aggregate_score", "elapsed_sec")}, indent=2))
for g in report.get("games", []):
    print(f"  {g['game_id']} L={g.get('levels_completed',0)} score={g.get('game_score',0):.2f} act={g.get('actions',0)} {g.get('state','')}")
