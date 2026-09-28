"""Smoke: ls20 / ar25 / ft09 must WIN via SmartRouter INLINE."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
STARTER = PKG.parent / "ARC-AGI-3-Kaggle-Starter"
os.chdir(STARTER)
# Avoid picking up Agents/.env OPERATION_MODE=online
os.environ["OPERATION_MODE"] = "offline"
os.environ["ENVIRONMENTS_DIR"] = str(STARTER / "environment_files")
for p in (str(PKG), str(STARTER), str(STARTER / "vendor" / "ARC-AGI-3-Agents")):
    if p not in sys.path:
        sys.path.insert(0, p)

import arc_agi
from arc_agi import OperationMode


def _load_agent():
    path = PKG / "agent" / "my_agent.py"
    spec = importlib.util.spec_from_file_location("smart_router_agent", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.MyAgent


def run_one(MyAgent, game_id: str, max_steps: int = 800) -> dict:
    MyAgent.MAX_ACTIONS = max_steps
    try:
        arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    except Exception:
        arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make(game_id)
    if env is None:
        raise RuntimeError(f"env_create_failed for {game_id}")
    agent = MyAgent(
        card_id="smoke-smart",
        game_id=game_id,
        agent_name=f"smoke.{game_id}",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=env,
        tags=["smoke"],
    )
    agent.main()
    fr = agent.frames[-1]
    return {
        "game_id": game_id,
        "levels": int(fr.levels_completed or 0),
        "state": fr.state.name if hasattr(fr.state, "name") else str(fr.state),
        "actions": int(agent.action_counter),
        "route": getattr(agent, "_route", "?"),
    }


def main() -> None:
    print("cwd", Path.cwd())
    print("env_dir", STARTER / "environment_files", "exists", (STARTER / "environment_files").is_dir())
    MyAgent = _load_agent()
    rows = []
    for gid in ("ls20", "ar25", "ft09"):
        row = run_one(MyAgent, gid)
        rows.append(row)
        print(
            f"{row['game_id']:6} route={row['route']:16} "
            f"L={row['levels']} act={row['actions']} {row['state']}"
        )
    bad = [r for r in rows if "WIN" not in r["state"]]
    if bad:
        raise SystemExit(f"FLOOR FAIL: {bad}")
    print("SMOKE_OK all floors WIN")


if __name__ == "__main__":
    main()
