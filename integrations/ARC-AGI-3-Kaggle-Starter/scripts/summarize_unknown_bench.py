"""Post-process unknown/submit benches + sample transfer metrics on a few games."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

PLUGIN_GAMES = {"ls20", "ar25"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(report: dict, label: str) -> dict:
    games = report.get("games") or []
    nonzero = [g for g in games if int(g.get("levels_completed") or 0) > 0]
    wins = [g for g in games if "WIN" in str(g.get("state", ""))]
    unknown = [g for g in games if g.get("game_id") not in PLUGIN_GAMES]
    unknown_nz = [g for g in unknown if int(g.get("levels_completed") or 0) > 0]
    plugin = [g for g in games if g.get("game_id") in PLUGIN_GAMES]
    return {
        "label": label,
        "aggregate": report.get("aggregate_score"),
        "aggregate_official": report.get("aggregate_score_official"),
        "total_games": report.get("total_games"),
        "max_steps": report.get("max_steps"),
        "elapsed_sec": report.get("elapsed_sec"),
        "games_won": len(wins),
        "nonzero_games": len(nonzero),
        "nonzero_ids": [g["game_id"] for g in nonzero],
        "total_levels": report.get("total_levels_completed"),
        "unknown_games": len(unknown),
        "unknown_nonzero": len(unknown_nz),
        "unknown_nonzero_ids": [g["game_id"] for g in unknown_nz],
        "plugin_rows": [
            {
                "game_id": g["game_id"],
                "levels": g.get("levels_completed"),
                "score": g.get("game_score"),
                "state": g.get("state"),
                "actions": g.get("actions"),
            }
            for g in plugin
        ],
        "per_game": [
            {
                "game_id": g["game_id"],
                "levels": g.get("levels_completed"),
                "score": round(float(g.get("game_score") or 0), 2),
                "state": g.get("state"),
                "actions": g.get("actions"),
                "total_levels": g.get("total_levels"),
                "unknown": g.get("game_id") not in PLUGIN_GAMES,
            }
            for g in sorted(games, key=lambda x: x.get("game_id", ""))
        ],
    }


def probe_transfer_metrics(game_ids: list[str], max_steps: int = 80) -> list[dict]:
    """Short runs collecting L6 TransferLayer.metrics() after each game."""
    import arc_agi
    from arc_agi import OperationMode
    from agent.unknown_solo_agent import MyAgent

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    rows = []
    for gid in game_ids:
        env = arc.make(gid)
        agent = MyAgent(
            card_id="xfer-probe",
            game_id=gid,
            agent_name=f"xfer.{gid}",
            ROOT_URL="http://localhost",
            record=False,
            arc_env=env,
            tags=["xfer"],
        )
        MyAgent.MAX_ACTIONS = max_steps
        try:
            agent.main()
            final = agent.frames[-1]
            metrics = {}
            brain = getattr(agent, "brain", None)
            if brain is not None and getattr(brain, "transfer", None) is not None:
                metrics = brain.transfer.metrics()
            rows.append(
                {
                    "game_id": gid,
                    "levels_completed": int(final.levels_completed or 0),
                    "actions": int(agent.action_counter),
                    "state": final.state.name if hasattr(final.state, "name") else str(final.state),
                    "transfer": metrics,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"game_id": gid, "error": f"{type(exc).__name__}: {exc}"})
        print(f"xfer-probe {gid}: {rows[-1]}", flush=True)
    return rows


def main() -> None:
    unknown_path = ROOT / "ui" / "static" / "games_benchmark_unknown_l6_400.json"
    submit_path = ROOT / "ui" / "static" / "games_benchmark_submit_l6_400.json"
    prev_path = ROOT / "ui" / "static" / "games_benchmark_p1_coverage400.json"

    out: dict = {"generated_note": "unknown env comprehensive bench summary"}
    if unknown_path.exists():
        out["unknown_solo_l6"] = summarize(load(unknown_path), "unknown-solo+L6")
    if submit_path.exists():
        out["submit_plugins"] = summarize(load(submit_path), "submit-plugins+solo+L6")
    elif (ROOT / "ui" / "static" / "games_benchmark_p1_coverage400.json").exists():
        out["prev_p1_coverage400"] = summarize(load(prev_path), "prev-p1-coverage400")

    # Sample transfer metrics on a mix of games (short)
    sample = ["vc33", "sp80", "ls20", "ar25", "wa30", "ft09"]
    print("Probing transfer metrics...", flush=True)
    out["transfer_probes"] = probe_transfer_metrics(sample, max_steps=80)

    dest = ROOT / "ui" / "static" / "unknown_env_bench_summary.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: (v if k != "unknown_solo_l6" and k != "submit_plugins" else {
        "aggregate": v.get("aggregate"),
        "nonzero": v.get("nonzero_games"),
        "unknown_nonzero": v.get("unknown_nonzero"),
        "wins": v.get("games_won"),
        "levels": v.get("total_levels"),
    }) for k, v in out.items() if k != "transfer_probes"}, ensure_ascii=False, indent=2))
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
