import json
from pathlib import Path

p = Path("ui/static/games_benchmark_v3_offline_400.json")
data = json.loads(p.read_text(encoding="utf-8"))

print("=== SUMMARY ===")
for k in ("agent", "max_steps", "total_games", "games_finished", "running",
          "aggregate_score", "total_levels_completed", "games_won", "elapsed_sec"):
    print(f"  {k}: {data.get(k)}")

print("\n=== 逐局 ===")
print(f"{'game':8} {'L':>3} {'score':>7} {'acts':>5} {'state':>14} {'elapsed':>7}")
for g in data.get("games", []):
    print(f"{g['game_id']:8} {g.get('levels_completed',0):>3} {g.get('game_score',0):>7.2f} "
          f"{g.get('actions',0):>5} {g.get('state',''):>14} {g.get('elapsed_sec',0):>7.1f}")

# Compare with baseline
p2 = Path("ui/static/games_benchmark_main_0486e5c_offline_400.json")
if p2.exists():
    prev = json.loads(p2.read_text(encoding="utf-8"))
    prev_map = {g["game_id"]: g for g in prev.get("games", [])}
    print("\n=== vs 基线 ===")
    print(f"{'game':8} {'prev_acts':>8} {'new_acts':>8} {'saved':>5} {'pL':>3} {'nL':>3} {'pScore':>7} {'nScore':>7}")
    ts = 0
    for g in data.get("games", []):
        gid = g["game_id"]
        pg = prev_map.get(gid, {})
        pa, na = pg.get("actions",0), g.get("actions",0)
        s = pa - na; ts += s
        m = "  " if s == 0 and pg.get("game_score",0)==g.get("game_score",0) and pg.get("levels_completed",0)==g.get("levels_completed",0) else "!!"
        print(f"{m} {gid:8} {pa:>8} {na:>8} {s:>5} {pg.get('levels_completed',0):>3} {g.get('levels_completed',0):>3} {pg.get('game_score',0):>7.2f} {g.get('game_score',0):>7.2f}")
    print(f"\n  省步={ts}  prev_agg={prev.get('aggregate_score'):.2f} new_agg={data.get('aggregate_score'):.2f}")
    print(f"  prev_won={prev.get('games_won')} new_won={data.get('games_won')}  prev_L={prev.get('total_levels_completed')} new_L={data.get('total_levels_completed')}")
    print(f"  prev_t={prev.get('elapsed_sec'):.0f}s new_t={data.get('elapsed_sec'):.0f}s")
