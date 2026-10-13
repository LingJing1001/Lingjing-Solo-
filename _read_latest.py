import json
from pathlib import Path

p = Path("integrations/ARC-AGI-3-Kaggle-Starter/ui/static/games_benchmark_wa30_l6_offline_400.json")
d = json.loads(p.read_text(encoding="utf-8"))
print(f"running={d.get('running')} games={d.get('total_games')} won={d.get('games_won')} levels={d.get('total_levels_completed')} agg={d.get('aggregate_score'):.2f} elapsed={d.get('elapsed_sec'):.0f}s")
print(f"{'game':8} {'L':>3} {'score':>7} {'acts':>5} {'state':>14}")
for g in d.get("games", []):
    print(f"{g['game_id']:8} {g.get('levels_completed',0):>3} {g.get('game_score',0):>7.2f} {g.get('actions',0):>5} {g.get('state',''):>14}")

# Compare with baseline
p2 = Path("integrations/ARC-AGI-3-Kaggle-Starter/ui/static/games_benchmark_main_0486e5c_offline_400.json")
if p2.exists():
    prev = json.loads(p2.read_text(encoding="utf-8"))
    prev_map = {g["game_id"]: g for g in prev.get("games", [])}
    print(f"\n=== vs baseline ===")
    print(f"  prev: agg={prev.get('aggregate_score'):.2f} won={prev.get('games_won')} levels={prev.get('total_levels_completed')}")
    print(f"  new:  agg={d.get('aggregate_score'):.2f} won={d.get('games_won')} levels={d.get('total_levels_completed')}")
    for g in d.get("games", []):
        pg = prev_map.get(g["game_id"], {})
        if g.get('levels_completed',0) != pg.get('levels_completed',0) or g.get('game_score',0) != pg.get('game_score',0):
            print(f"  !! {g['game_id']:8} L {pg.get('levels_completed',0)}->{g.get('levels_completed',0)}  score {pg.get('game_score',0):.2f}->{g.get('game_score',0):.2f}")
