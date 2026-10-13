import json
from pathlib import Path

p = Path("ui/static/games_benchmark_failclosed_offline_400.json")
data = json.loads(p.read_text(encoding="utf-8"))

print("=== SUMMARY ===")
for k in ("agent", "max_steps", "total_games", "games_finished", "running",
          "aggregate_score", "aggregate_score_official", "total_levels_completed",
          "games_won", "games_zero_levels", "elapsed_sec"):
    print(f"  {k}: {data.get(k)}")

print("\n=== 逐局结果 ===")
print(f"{'game':8} {'L':>3} {'score':>7} {'acts':>5} {'state':>14} {'elapsed':>7}")
for g in data.get("games", []):
    print(f"{g['game_id']:8} {g.get('levels_completed',0):>3} {g.get('game_score',0):>7.2f} "
          f"{g.get('actions',0):>5} {g.get('state',''):>14} {g.get('elapsed_sec',0):>7.1f}")

errors = [g for g in data.get("games", []) if str(g.get("state","")).startswith("ERROR")]
print(f"\n=== ERROR: {len(errors)} ===")

# Compare with previous baseline
p2 = Path("ui/static/games_benchmark_main_0486e5c_offline_400.json")
if p2.exists():
    prev = json.loads(p2.read_text(encoding="utf-8"))
    print("\n=== 与基线对比 ===")
    prev_map = {g["game_id"]: g for g in prev.get("games", [])}
    print(f"{'game':8} {'prev_acts':>9} {'new_acts':>8} {'saved':>6} {'prev_score':>10} {'new_score':>9}")
    total_saved = 0
    for g in data.get("games", []):
        gid = g["game_id"]
        pg = prev_map.get(gid, {})
        pa = pg.get("actions", 0)
        na = g.get("actions", 0)
        saved = pa - na
        total_saved += saved
        ps = pg.get("game_score", 0)
        ns = g.get("game_score", 0)
        marker = "  " if saved == 0 and ps == ns else "!!"
        print(f"{marker} {gid:8} {pa:>9} {na:>8} {saved:>6} {ps:>10.2f} {ns:>9.2f}")
    print(f"\n  总省步数: {total_saved}")
    print(f"  prev aggregate={prev.get('aggregate_score'):.2f}  new aggregate={data.get('aggregate_score'):.2f}")
    print(f"  prev won={prev.get('games_won')}  new won={data.get('games_won')}")
    print(f"  prev levels={prev.get('total_levels_completed')}  new levels={data.get('total_levels_completed')}")
    print(f"  prev elapsed={prev.get('elapsed_sec'):.0f}s  new elapsed={data.get('elapsed_sec'):.0f}s")
