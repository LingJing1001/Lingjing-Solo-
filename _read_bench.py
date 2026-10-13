import json
from pathlib import Path

p = Path("ui/static/games_benchmark_main_0486e5c_offline_400.json")
data = json.loads(p.read_text(encoding="utf-8"))

# Summary
print("=== SUMMARY ===")
for k in ("agent", "max_steps", "total_games", "games_finished", "running",
          "aggregate_score", "aggregate_score_official", "total_levels_completed",
          "games_won", "games_zero_levels", "elapsed_sec"):
    print(f"  {k}: {data.get(k)}")

print("\n=== PRIORITY FIX (零分游戏) ===")
print(data.get("priority_fix"))

print("\n=== 逐局结果 ===")
print(f"{'game':8} {'L':>3} {'score':>7} {'acts':>5} {'state':>14} {'elapsed':>7} error")
for g in data.get("games", []):
    err = (g.get("error") or "")[:40]
    print(f"{g['game_id']:8} {g.get('levels_completed',0):>3} {g.get('game_score',0):>7.2f} "
          f"{g.get('actions',0):>5} {g.get('state',''):>14} {g.get('elapsed_sec',0):>7.1f} {err}")

# Check for ERROR states
errors = [g for g in data.get("games", []) if str(g.get("state","")).startswith("ERROR")]
print(f"\n=== ERROR 局数: {len(errors)} ===")
for e in errors:
    print(f"  {e['game_id']}: {e.get('error','')}")
