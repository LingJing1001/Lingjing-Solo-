"""Run `agent/my_agent.py` locally against a real ARC-AGI-3 game.

This is the fast inner-loop: no Docker, no Kaggle round-trip. Uses the
`arc-agi` PyPI package to host the game engine and the ARC-AGI-3-Agents
framework's `Agent.main()` loop to drive it — exactly what the Kaggle
gateway does, just in-process.

Usage:
    .venv/bin/python scripts/play_local.py --game ls20 --max-steps 200
    .venv/bin/python scripts/play_local.py --list
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import shutil
import sys
import time
from pathlib import Path

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
if not VENDOR.exists():
    raise SystemExit(f"Framework not found at {VENDOR}. Run `make setup` first.")
sys.path.insert(0, str(VENDOR))

# The slim framework initializer imports this template. Keep local execution
# on the same agent source that the notebook builder packages.
TEMPLATE = VENDOR / "agents" / "templates" / "my_agent.py"
TEMPLATE.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(ROOT / "agent" / "my_agent.py", TEMPLATE)

import arc_agi
from arc_agi import OperationMode


def _load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        existing = os.environ.get(k, "")
        # Never treat boolean flags like "0"/"1" as weak placeholders.
        weak_key = (
            k.endswith("_KEY")
            or k.endswith("_TOKEN")
            or k == "OPENAI_API_KEY"
        )
        weak = weak_key and (
            len(existing) < 16
            or existing.lower() in {"sk-xxxx", "sk-xxx", "your_api_key", "replace_me"}
        )
        if k and (k not in os.environ or weak):
            os.environ[k] = v


def load_agent_class(agent_path: Path):
    """Import MyAgent from an arbitrary .py path."""
    spec = importlib.util.spec_from_file_location("user_agent_module", agent_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load {agent_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "MyAgent"):
        raise SystemExit(f"{agent_path} must define class MyAgent")
    return module.MyAgent


def load_my_agent_class():
    """Import MyAgent from agent/my_agent.py via importlib."""
    return load_agent_class(ROOT / "agent" / "my_agent.py")


def main() -> None:
    _load_dotenv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default=None,
                   help="Game id to play. If omitted, plays ALL available games "
                        "(mirrors what Kaggle does in competition rerun). "
                        "Comma-separated list also accepted, e.g. ls20,vc33.")
    p.add_argument("--max-steps", type=int, default=400,
                   help="Per-game cap on actions (overrides MyAgent.MAX_ACTIONS).")
    p.add_argument("--list", action="store_true",
                   help="List available games and exit.")
    p.add_argument("--render", default=None, choices=[None, "terminal"],
                   help="Optional terminal rendering each step.")
    p.add_argument(
        "--agent",
        default=None,
        help="Path to agent .py (default: agent/my_agent.py). "
             "Use agent/baselines/aether_prime.py for baseline compare.",
    )
    p.add_argument(
        "--record-dir",
        default=None,
        help="Save recorder JSONL files under this directory; enables recording.",
    )
    args = p.parse_args()
    if args.record_dir:
        os.environ["RECORDINGS_DIR"] = str(Path(args.record_dir).resolve())

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # NORMAL = local execution; game source is downloaded on first call into
    # ./environment_files/ and cached for subsequent runs.
    arc = None
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"Arcade init failed (attempt {attempt+1}/5): {type(e).__name__}: {e}")
            time.sleep(2 + attempt)
    if arc is None:
        raise SystemExit(f"Could not init Arcade after retries: {last_err}")
    all_envs = arc.get_environments()

    if args.list:
        print(f"{len(all_envs)} environments:")
        for e in all_envs:
            print(f"  {e.game_id}: {getattr(e, 'title', '?')}")
        return

    # Resolve which games to play. `arc.make()` accepts the short game id
    # (e.g. "ls20") even though the EnvironmentInfo.game_id includes the
    # version suffix ("ls20-9607627b"), so we normalize to short ids.
    if args.game:
        wanted = {g.strip().split("-")[0] for g in args.game.split(",")}
        game_ids = [e.game_id.split("-")[0] for e in all_envs
                    if e.game_id.split("-")[0] in wanted]
        missing = wanted - set(game_ids)
        if missing:
            raise SystemExit(f"Unknown game id(s): {sorted(missing)}. Run --list.")
    else:
        game_ids = [e.game_id.split("-")[0] for e in all_envs]
        print(f"No --game specified; playing all {len(game_ids)} games "
              f"(this is what Kaggle does in competition rerun).\n")

    agent_path = Path(args.agent) if args.agent else (ROOT / "agent" / "my_agent.py")
    if not agent_path.is_absolute():
        agent_path = (ROOT / agent_path).resolve()
    MyAgentCls = load_agent_class(agent_path)
    print(f"Agent file: {agent_path}")
    if hasattr(MyAgentCls, "MAX_ACTIONS"):
        MyAgentCls.MAX_ACTIONS = min(MyAgentCls.MAX_ACTIONS, args.max_steps)

    per_game = []
    for i, game_id in enumerate(game_ids, 1):
        print(f"=== [{i}/{len(game_ids)}] {game_id} ===")
        env = arc.make(game_id, render_mode=args.render)
        if env is None:
            print(f"  could not create env for {game_id!r}, skipping")
            continue

        agent = MyAgentCls(
            card_id="local-dev",
            game_id=game_id,
            agent_name=f"MyAgent.local.{game_id}",
            ROOT_URL="http://localhost",
            record=bool(args.record_dir),
            arc_env=env,
            tags=["local-dev"],
        )
        agent.main()

        final = agent.frames[-1]
        per_game.append((game_id, final.state, final.levels_completed,
                         agent.action_counter))
        print(f"  → state={final.state}, levels_completed={final.levels_completed}, "
              f"actions={agent.action_counter}")

    sc = arc.get_scorecard()
    print("\n========= SUMMARY =========")
    for gid, state, levels, actions in per_game:
        print(f"  {gid:8} levels={levels:3}  actions={actions:5}  state={state}")
    score_val = sc.score if hasattr(sc, "score") else sc
    print(f"\nAggregate scorecard score: {score_val}")


if __name__ == "__main__":
    main()
