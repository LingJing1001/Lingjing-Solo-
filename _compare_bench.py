import json
from pathlib import Path

p = Path("ui/static/games_benchmark_main_r3kernel_offline_400.json")
data = json.loads(p.read_text(encoding="utf-8"))

print("=== SUMMARY ===")
for k in ("agent", "max_steps", "total_games", "games_finished", "running",
          "aggregate_score", "aggregate_score_official", "total_levels_completed",
          "games_won", "games_zero_levels", "elapsed_sec"):
    print(f"  {k}: {data.get(k)}")

print("\n=== PRIORITY FIX ===")
print(data.get("priority_fix"))

print("\n=== 逐局结果 ===")
print(f"{'game':8} {'L':>3} {'score':>7} {'acts':>5} {'state':>14} {'elapsed':>7}")
for g in data.get("games", []):
    print(f"{g['game_id']:8} {g.get('levels_completed',0):>3} {g.get('game_score',0):>7.2f} "
          f"{g.get('actions',0):>5} {g.get('state',''):>14} {g.get('elapsed_sec',0):>7.1f}")

errors = [g for g in data.get("games", []) if str(g.get("state","")).startswith("ERROR")]
print(f"\n=== ERROR: {len(errors)} ===")

# Compare with previous run
p2 = Path("ui/static/games_benchmark_main_0486e5c_offline_400.json")
if p2.exists():
    prev = json.loads(p2.read_text(encoding="utf-8"))
    print("\n=== 与上次对比 ===")
    prev_map = {g["game_id"]: g for g in prev.get("games", [])}
    for g in data.get("games", []):
        gid = g["game_id"]
        pg = prev_map.get(gid, {})
        d_score = g.get("game_score",0) - pg.get("game_score",0)
        d_levels = g.get("levels_completed",0) - pg.get("levels_completed",0)
        marker = "  " if d_score == 0 and d_levels == 0 else "!!"
        if marker == "!!":
            print(f"  {marker} {gid:8} score {pg.get('game_score',0):.2f}->{g.get('game_score',0):.2f}  L {pg.get('levels_completed',0)}->{g.get('levels_completed',0)}")
    print(f"  prev aggregate={prev.get('aggregate_score'):.2f}  new aggregate={data.get('aggregate_score'):.2f}")
    print(f"  prev won={prev.get('games_won')}  new won={data.get('games_won')}")
    print(f"  prev levels={prev.get('total_levels_completed')}  new levels={data.get('total_levels_completed')}")
