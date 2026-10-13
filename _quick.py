import json
from pathlib import Path
p = Path("ui/static/games_benchmark_r3wired_offline_400.json")
d = json.loads(p.read_text(encoding="utf-8"))
print(f"running={d.get('running')} games={d.get('total_games')} won={d.get('games_won')} levels={d.get('total_levels_completed')} agg={d.get('aggregate_score'):.2f}")
print(f"{'game':8} {'L':>3} {'score':>7} {'acts':>5} {'state':>14}")
for g in d.get("games", []):
    print(f"{g['game_id']:8} {g.get('levels_completed',0):>3} {g.get('game_score',0):>7.2f} {g.get('actions',0):>5} {g.get('state',''):>14}")
