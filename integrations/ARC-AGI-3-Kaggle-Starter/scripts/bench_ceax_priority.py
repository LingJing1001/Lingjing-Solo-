"""Run CEAX-primary agent on priority unknown games (no plugins)."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

from scripts.benchmark_all_games import run_all

PRIORITY = [
    "lp85", "vc33", "sb26", "tu93", "s5i5", "r11l", "su15", "bp35",
    "tn36", "cd82", "ft09", "sc25",
]


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--games", nargs="*", default=PRIORITY)
    p.add_argument("--out", default="ui/static/games_benchmark_ceax_priority.json")
    args = p.parse_args()
    out = ROOT / args.out
    t0 = time.time()
    report = run_all(
        max_steps=args.max_steps,
        agent_path=ROOT / "agent" / "ceax_unknown_agent.py",
        game_filter=set(args.games),
        out_json=out,
        agent_label="ceax-primary-unknown",
    )
    nz = [g for g in report["games"] if g.get("levels_completed", 0) > 0]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 2),
        "aggregate": report.get("aggregate_score"),
        "nonzero": len(nz),
        "nonzero_ids": [g["game_id"] for g in nz],
        "total_levels": report.get("total_levels_completed"),
        "games": [
            {
                "game_id": g["game_id"],
                "levels": g["levels_completed"],
                "score": g["game_score"],
                "state": g["state"],
                "actions": g["actions"],
            }
            for g in sorted(report["games"], key=lambda x: (-x["levels_completed"], x["game_id"]))
        ],
    }
    sum_path = out.with_name(out.stem + "_summary.json")
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
