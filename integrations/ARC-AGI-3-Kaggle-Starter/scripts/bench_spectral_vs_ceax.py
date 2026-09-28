"""SSA 对比基准：spectral_agi_agent vs ceax_unknown_agent（优先级未知局）。

输出两种汇总 JSON：
  ui/static/bench_ssa_<label>.json + <...>_summary.json
  ui/static/bench_ssa_comparison.json  （对照表 + 结论字段）

用法：
  python scripts/bench_spectral_vs_ceax.py --max-steps 300 \
      --games lp85 vc33 sb26 s5i5 r11l bp35
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT)
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)
__import__("sys").path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

from scripts.benchmark_all_games import run_all  # noqa: E402

ARMS = [
    ("ssa-spectral", ROOT / "agent" / "spectral_agi_agent.py"),
    ("ceax-baseline", ROOT / "agent" / "ceax_unknown_agent.py"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument(
        "--games",
        nargs="*",
        default=["lp85", "vc33", "sb26", "s5i5", "r11l", "bp35"],
    )
    p.add_argument("--out-prefix", default="ui/static/bench_ssa")
    args = p.parse_args()

    results = {}
    for label, agent_path in ARMS:
        out = ROOT / f"{args.out_prefix}_{label}.json"
        t0 = time.time()
        report = run_all(
            max_steps=args.max_steps,
            agent_path=agent_path,
            game_filter=set(args.games),
            out_json=out,
            agent_label=label,
        )
        games = [
            {
                "game_id": g["game_id"],
                "levels": g["levels_completed"],
                "score": g["game_score"],
                "state": g["state"],
                "actions": g["actions"],
            }
            for g in report.get("games", [])
        ]
        results[label] = {
            "elapsed_sec": round(time.time() - t0, 1),
            "aggregate": report.get("aggregate_score"),
            "total_levels": report.get("total_levels_completed"),
            "nonzero": sum(1 for g in games if g["levels"] > 0),
            "games": sorted(games, key=lambda x: x["game_id"]),
        }

    # 对照表
    a, b = results[ARMS[0][0]], results[ARMS[1][0]]
    rows = []
    for ga in a["games"]:
        gb = next((g for g in b["games"] if g["game_id"] == ga["game_id"]), None)
        rows.append(
            {
                "game_id": ga["game_id"],
                f"{ARMS[0][0]}_levels": ga["levels"],
                f"{ARMS[1][0]}_levels": (gb or {}).get("levels", 0),
                "delta": ga["levels"] - (gb or {}).get("levels", 0),
            }
        )
    comparison = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "max_steps": args.max_steps,
        "games": args.games,
        "summary": {
            label: {
                "aggregate": r["aggregate"],
                "total_levels": r["total_levels"],
                "nonzero": r["nonzero"],
                "elapsed_sec": r["elapsed_sec"],
            }
            for label, r in results.items()
        },
        "per_game": rows,
        "spectral_wins": sum(1 for r in rows if r["delta"] > 0),
        "baseline_wins": sum(1 for r in rows if r["delta"] < 0),
    }
    cmp_path = ROOT / f"{args.out_prefix}_comparison.json"
    cmp_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparison["summary"], ensure_ascii=False, indent=2))
    print(f"comparison -> {cmp_path}")


if __name__ == "__main__":
    main()
