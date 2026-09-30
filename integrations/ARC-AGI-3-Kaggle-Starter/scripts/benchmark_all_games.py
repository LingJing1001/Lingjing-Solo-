"""全游戏摸底：逐局运行 MyAgent，输出 scorecard 分数与过关数 JSON。"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CANONICAL_ROOT = ROOT.parents[1]
if not CANONICAL_ROOT.is_dir():
    raise SystemExit(f"Project root not found: {CANONICAL_ROOT}")
os.environ.setdefault("LINGJING_SRC", str(CANONICAL_ROOT / "lingjing_solo"))
sys.path.insert(0, str(CANONICAL_ROOT))
VENDOR_DEPS = ROOT / "vendor"
if VENDOR_DEPS.exists():
    sys.path.insert(0, str(VENDOR_DEPS))
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
sys.path.insert(0, str(VENDOR))
TEMPLATE = VENDOR / "agents" / "templates" / "my_agent.py"
TEMPLATE.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(ROOT / "agent" / "my_agent.py", TEMPLATE)

import arc_agi
from arc_agi import OperationMode

OUT_JSON = ROOT / "ui" / "static" / "games_benchmark.json"
TEAM_OUT_JSON = ROOT / "ui" / "static" / "games_benchmark_team.json"


def _load_meta() -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    base = ROOT / "environment_files"
    if not base.exists():
        return meta
    for path in base.rglob("metadata.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        gid = str(data.get("game_id", "")).split("-")[0]
        if not gid:
            continue
        baselines = data.get("baseline_actions") or []
        meta[gid] = {
            "title": data.get("title", gid),
            "tags": data.get("tags") or [],
            "total_levels": len(baselines),
            "human_steps": baselines,
        }
    return meta


def _load_agent(agent_path: Optional[Path] = None):
    path = agent_path or (ROOT / "agent" / "my_agent.py")
    spec = importlib.util.spec_from_file_location("bench_agent", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MyAgent


def _state_name(state: Any) -> str:
    return state.name if hasattr(state, "name") else str(state)


def _pick_run(env_list: Any, game_id: str) -> Optional[dict[str, Any]]:
    if not env_list or not env_list.environments:
        return None
    for env in env_list.environments:
        eid = (env.id or "").split("-")[0]
        if eid != game_id:
            continue
        if not env.runs:
            return None
        best = max(env.runs, key=lambda r: r.score or 0)
        return best.model_dump()
    return None


def run_all(
    max_steps: int = 400,
    agent_path: Optional[Path] = None,
    game_filter: Optional[set[str]] = None,
    on_progress: Optional[Callable[[dict[str, Any]], None]] = None,
    out_json: Optional[Path] = None,
    agent_label: Optional[str] = None,
) -> dict[str, Any]:
    meta = _load_meta()
    try:
        arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    except Exception:
        arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    all_envs = arc.get_environments()
    game_ids = [e.game_id.split("-")[0] for e in all_envs]
    if game_filter:
        game_ids = [g for g in game_ids if g in game_filter]

    agent_file = Path(agent_path) if agent_path else (ROOT / "agent" / "my_agent.py")
    if not agent_file.is_absolute():
        agent_file = (ROOT / agent_file).resolve()
    label = agent_label or str(agent_file.relative_to(ROOT) if agent_file.is_relative_to(ROOT) else agent_file)
    out_path = Path(out_json) if out_json else OUT_JSON

    MyAgent = _load_agent(agent_file)
    if hasattr(MyAgent, "MAX_ACTIONS"):
        MyAgent.MAX_ACTIONS = min(MyAgent.MAX_ACTIONS, max_steps)

    games_out: list[dict[str, Any]] = []
    t0 = time.time()

    for i, game_id in enumerate(game_ids, 1):
        g_t0 = time.time()
        row: dict[str, Any] = {
            "game_id": game_id,
            "index": i,
            "title": meta.get(game_id, {}).get("title", game_id),
            "tags": meta.get(game_id, {}).get("tags", []),
            "total_levels": meta.get(game_id, {}).get("total_levels", 0),
            "human_steps": meta.get(game_id, {}).get("human_steps", []),
            "levels_completed": 0,
            "actions": 0,
            "state": "ERROR",
            "game_score": 0.0,
            "level_scores": [],
            "elapsed_sec": 0.0,
            "error": None,
        }
        try:
            env = arc.make(game_id)
            if env is None:
                row["error"] = "env_create_failed"
                row["state"] = "SKIP"
            else:
                agent = MyAgent(
                    card_id="bench-all",
                    game_id=game_id,
                    agent_name=f"bench.{game_id}",
                    ROOT_URL="http://localhost",
                    record=False,
                    arc_env=env,
                    tags=["benchmark"],
                )
                agent.main()
                final = agent.frames[-1]
                row["levels_completed"] = int(final.levels_completed or 0)
                row["actions"] = int(agent.action_counter)
                row["state"] = _state_name(final.state)

                sc = arc.get_scorecard()
                run = _pick_run(sc, game_id)
                if run:
                    row["game_score"] = float(run.get("score") or 0)
                    row["level_scores"] = run.get("level_scores") or []
                    row["level_actions"] = run.get("level_actions") or []
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["state"] = "ERROR"

        row["elapsed_sec"] = round(time.time() - g_t0, 2)
        games_out.append(row)

        partial = _build_report(
            games_out, meta, max_steps, time.time() - t0, running=True, agent=label
        )
        if on_progress:
            on_progress(partial)
        else:
            out_path.write_text(
                json.dumps(partial, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(
            f"[{i}/{len(game_ids)}] {game_id:8} "
            f"L={row['levels_completed']} score={row['game_score']:.2f} "
            f"act={row['actions']} {row['state']}"
        )

    report = _build_report(
        games_out, meta, max_steps, time.time() - t0, running=False, agent=label
    )
    sc = arc.get_scorecard()
    if sc is not None:
        report["aggregate_score_official"] = float(sc.score or 0)
        report["aggregate_score"] = report["aggregate_score_official"]
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _build_report(
    games: list[dict[str, Any]],
    meta: dict[str, dict[str, Any]],
    max_steps: int,
    elapsed: float,
    running: bool,
    agent: str = "agent/my_agent.py",
) -> dict[str, Any]:
    done = [g for g in games if g.get("state") != "SKIP"]
    agg = sum(g.get("game_score", 0) for g in done) / max(len(done), 1)
    total_levels = sum(g.get("levels_completed", 0) for g in games)
    wins = sum(1 for g in games if "WIN" in str(g.get("state", "")))
    zeros = [g for g in games if g.get("levels_completed", 0) == 0]

    ranked = sorted(games, key=lambda g: (-g.get("game_score", 0), -g.get("levels_completed", 0)))
    priority = sorted(
        zeros,
        key=lambda g: (-g.get("total_levels", 0), g.get("game_id", "")),
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "max_steps": max_steps,
        "total_games": len(games),
        "games_finished": len(games),
        "running": running,
        "elapsed_sec": round(elapsed, 2),
        "aggregate_score": round(agg, 3),
        "aggregate_score_official": None,
        "total_levels_completed": total_levels,
        "games_won": wins,
        "games_zero_levels": len(zeros),
        "priority_fix": [g["game_id"] for g in priority[:10]],
        "games": ranked,
        "score_formula": "单局分=各关 (基准步数/实际步数)²×100 加权平均；总分=所有游戏分数的平均值",
    }


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="全游戏摸底")
    p.add_argument("max_steps", nargs="?", type=int, default=400)
    p.add_argument("--agent", default="agent/my_agent.py")
    p.add_argument("--out", default=None, help="输出 JSON 路径")
    p.add_argument("--label", default=None, help="看板显示的 agent 名称")
    args = p.parse_args()
    out = Path(args.out) if args.out else OUT_JSON
    if not out.is_absolute():
        out = ROOT / out
    report = run_all(
        max_steps=args.max_steps,
        agent_path=args.agent,
        out_json=out,
        agent_label=args.label,
    )
    if report.get("aggregate_score_official") is None:
        report["aggregate_score_official"] = report["aggregate_score"]
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "aggregate": report["aggregate_score"],
            "games": report["total_games"],
            "levels": report["total_levels_completed"],
            "agent": report["agent"],
            "out": str(out),
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
