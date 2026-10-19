"""Run `agent/my_agent.py` locally against a real ARC-AGI-3 game.

This is the fast inner-loop: no Docker, no Kaggle round-trip. Uses the
`arc-agi` PyPI package to host the game engine and the ARC-AGI-3-Agents
framework's `Agent.main()` loop to drive it — exactly what the Kaggle
gateway does, just in-process.

Every game writes `runs/<game_id>.json` the moment it finishes (atomically,
even when the agent raises), so a crash or Ctrl-C never loses results.

Usage:
    # one game, serial
    .venv/bin/python scripts/play_local.py --game ls20 --max-steps 200

    # four games, four at a time
    .venv/bin/python scripts/play_local.py --game ls20,ar25,vc33,ft09 \\
        --max-steps 320 --parallel 4

    # whole corpus, 6 concurrent (8 logical cores => leave headroom)
    .venv/bin/python scripts/play_local.py --max-steps 400 --parallel 6

    .venv/bin/python scripts/play_local.py --list
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import queue
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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


def _load_baselines() -> dict[str, list[int]]:
    """`baseline_actions` per game from environment_files/**/metadata.json."""
    out: dict[str, list[int]] = {}
    base = ROOT / "environment_files"
    if not base.exists():
        return out
    for path in base.rglob("metadata.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        gid = str(data.get("game_id", "")).split("-")[0]
        baselines = data.get("baseline_actions") or []
        if gid and baselines:
            out[gid] = [int(x) for x in baselines]
    return out


def _state_name(state: Any) -> str:
    return state.name if hasattr(state, "name") else str(state)


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write via a temp file so a kill mid-write never leaves half a JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _pick_run(scorecard: Any, game_id: str) -> Optional[dict]:
    """Pull this game's best run out of the scorecard (same as benchmark_all)."""
    envs = getattr(scorecard, "environments", None)
    if not envs:
        return None
    for env in envs:
        if (env.id or "").split("-")[0] != game_id:
            continue
        if not env.runs:
            return None
        best = max(env.runs, key=lambda r: r.score or 0)
        return best.model_dump()
    return None


def _level_scores_estimate(human: list[int], actions_per_level: list[int]) -> list[float]:
    """100 * (human/used)^2 per level, capped at 115 like the official scale.

    Only used when the scorecard gives us nothing; kept in a separate field so
    an estimate is never mistaken for an official score.
    """
    out: list[float] = []
    for h, used in zip(human, actions_per_level):
        if not used or used <= 0:
            break
        out.append(round(min(1.15, (float(h) / float(used)) ** 2) * 100, 4))
    return out


def main() -> None:
    _load_dotenv()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default=None,
                   help="Game id to play. If omitted, plays ALL available games "
                        "(mirrors what Kaggle does in competition rerun). "
                        "Comma-separated list also accepted, e.g. ls20,vc33.")
    p.add_argument("--max-steps", type=int, default=400,
                   help="Per-game cap on actions (overrides MyAgent.MAX_ACTIONS).")
    p.add_argument("--parallel", type=int, default=1,
                   help="How many games to run at once (threads). Default 1 = "
                        "serial, i.e. the old behaviour. 8 logical cores -> 4-6 "
                        "is a sensible ceiling because the search is pure Python.")
    p.add_argument("--out-dir", default="runs",
                   help="Directory for per-game JSON + summary.json (default: runs/).")
    p.add_argument("--list", action="store_true",
                   help="List available games and exit.")
    p.add_argument("--render", default=None, choices=[None, "terminal"],
                   help="Optional terminal rendering each step (needs --parallel 1).")
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
    p.add_argument("--quiet", action="store_true",
                   help="Only warnings/errors from the agent (step log is noisy "
                        "when several games run at once).")
    args = p.parse_args()
    if args.record_dir:
        os.environ["RECORDINGS_DIR"] = str(Path(args.record_dir).resolve())

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

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
    if not game_ids:
        raise SystemExit("No games to play.")

    parallel = max(1, min(int(args.parallel), len(game_ids)))
    if args.render and parallel > 1:
        raise SystemExit("--render terminal interleaves badly; use --parallel 1.")

    agent_path = Path(args.agent) if args.agent else (ROOT / "agent" / "my_agent.py")
    if not agent_path.is_absolute():
        agent_path = (ROOT / agent_path).resolve()
    MyAgentCls = load_agent_class(agent_path)
    if hasattr(MyAgentCls, "MAX_ACTIONS"):
        MyAgentCls.MAX_ACTIONS = min(MyAgentCls.MAX_ACTIONS, args.max_steps)
    agent_sha = hashlib.md5(agent_path.read_bytes()).hexdigest()[:12]

    # The repo agent gates every calibration plan behind LINGJING_SCRIPTS,
    # which defaults to "off". Local runs then silently route everything to
    # CEAX_UNKNOWN and the notebook's plan behaviour is never reproduced.
    mod = sys.modules.get("user_agent_module")
    scripts_enabled = bool(getattr(mod, "SCRIPTS_ENABLED", True))
    if not scripts_enabled:
        print("[warn] LINGJING_SCRIPTS is off -> calibration plans and INLINE "
              "branches are disabled,\n"
              "       every game routes to CEAX_UNKNOWN. This does NOT match the "
              "notebook.\n"
              "       Re-run with LINGJING_SCRIPTS=on to exercise the plan path.\n")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    baselines = _load_baselines()
    cpu_n = os.cpu_count() or 1
    print(f"Agent file: {agent_path}  (md5 {agent_sha}, "
          f"MAX_ACTIONS={getattr(MyAgentCls, 'MAX_ACTIONS', '?')})")
    print(f"Games: {len(game_ids)}  |  parallel: {parallel} / {cpu_n} cores  "
          f"|  results: {out_dir}\n")

    make_lock = threading.Lock()
    print_lock = threading.Lock()
    results_lock = threading.Lock()
    results: dict[str, dict] = {}
    progress = {"done": 0}
    total = len(game_ids)

    def run_one(game_id: str, index: int) -> None:
        started = time.time()
        row: dict[str, Any] = {
            "game_id": game_id,
            "index": index,
            "agent": str(agent_path.relative_to(ROOT)
                          if agent_path.is_relative_to(ROOT) else agent_path),
            "agent_md5": agent_sha,
            "max_steps": int(getattr(MyAgentCls, "MAX_ACTIONS", args.max_steps)),
            "state": "ERROR",
            "levels_completed": 0,
            "actions": 0,
            "elapsed_sec": 0.0,
            "fps": 0.0,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "levels_total": len(baselines.get(game_id, [])),
            "human_steps": baselines.get(game_id, []),
            "game_score": None,
            "score_source": None,
            "level_scores": [],
            "level_actions": [],
            "game_score_estimate": None,
            "level_scores_estimate": [],
            "error": None,
            "traceback": None,
        }
        agent = None
        try:
            # Serialised on purpose: `Arcade.make()` installs the default
            # scorecard and loads the engine, neither is thread safe. Same
            # ordering as the official Swarm (envs first, then play).
            with make_lock:
                env = arc.make(game_id, render_mode=args.render)
            if env is None:
                row["state"] = "SKIP"
                row["error"] = "env_create_failed"
            else:
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
                row["state"] = _state_name(final.state)
                row["levels_completed"] = int(final.levels_completed or 0)
                row["actions"] = int(getattr(agent, "action_counter", 0))
        except BaseException as exc:  # noqa: BLE001 - keep whatever the game did
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()[-4000:]
            if agent is not None:
                row["actions"] = int(getattr(agent, "action_counter", 0))
                try:
                    row["levels_completed"] = int(agent.frames[-1].levels_completed or 0)
                    row["state"] = _state_name(agent.frames[-1].state)
                except Exception:  # noqa: BLE001
                    pass
        finally:
            row["elapsed_sec"] = round(time.time() - started, 2)
            row["fps"] = round(row["actions"] / max(row["elapsed_sec"], 0.1), 2)
            row["finished_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_write_json(out_dir / f"{game_id}.json", row)
            with results_lock:
                results[game_id] = row
                progress["done"] += 1
                done = progress["done"]
            with print_lock:
                tag = f"  !! {row['error']}" if row["error"] else ""
                print(f"[{done}/{total}] {game_id:8} L={row['levels_completed']} "
                      f"act={row['actions']:4} {row['elapsed_sec']:7.1f}s "
                      f"{row['state']}{tag}", flush=True)

    work: queue.Queue[tuple[int, str]] = queue.Queue()
    for i, gid in enumerate(game_ids, 1):
        work.put((i, gid))

    def worker() -> None:
        while True:
            try:
                index, game_id = work.get_nowait()
            except queue.Empty:
                return
            run_one(game_id, index)

    threads = [
        threading.Thread(target=worker, daemon=True, name=f"game-{k}")
        for k in range(parallel)
    ]
    t0 = time.time()
    interrupted = False
    try:
        for t in threads:
            t.start()
        for t in threads:
            while t.is_alive():
                t.join(timeout=0.5)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[interrupted] finished games are already on disk; "
              "still-running ones will be killed.")

    wall = time.time() - t0

    # One scorecard pass at the end, in the main thread: no race with agents,
    # and the official per-game score gets stamped into each JSON.
    official: Optional[float] = None
    try:
        sc = arc.get_scorecard()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] get_scorecard failed: {type(exc).__name__}: {exc}")
        sc = None
    if sc is not None:
        score = getattr(sc, "score", None)
        official = float(score) if score is not None else None
        for gid, row in list(results.items()):
            run = _pick_run(sc, gid)
            if not run:
                continue
            row["game_score"] = float(run.get("score") or 0)
            row["score_source"] = "official"
            row["level_scores"] = run.get("level_scores") or []
            row["level_actions"] = run.get("level_actions") or []
            _atomic_write_json(out_dir / f"{gid}.json", row)

    # Separate estimate field: readable JSON even without a scorecard, but never
    # confused with the official number.
    for row in results.values():
        if row["game_score"] is None and row["level_actions"] and row["human_steps"]:
            row["level_scores_estimate"] = _level_scores_estimate(
                row["human_steps"], row["level_actions"])
            row["game_score_estimate"] = round(
                sum(row["level_scores_estimate"])
                / max(len(row["human_steps"]), 1), 4)
            _atomic_write_json(out_dir / f"{row['game_id']}.json", row)

    ranked = sorted(
        results.values(),
        key=lambda r: (-(r["game_score"] or 0), -r["levels_completed"], r["game_id"]),
    )
    game_sec = sum(r["elapsed_sec"] for r in results.values())
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "agent": str(agent_path),
        "agent_md5": agent_sha,
        "max_steps": getattr(MyAgentCls, "MAX_ACTIONS", args.max_steps),
        "scripts_enabled": scripts_enabled,
        "parallel": parallel,
        "cores": cpu_n,
        "interrupted": interrupted,
        "games_total": total,
        "games_done": len(results),
        "games_win": [r["game_id"] for r in ranked if "WIN" in r["state"]],
        "games_zero_levels": [r["game_id"] for r in ranked
                              if r["levels_completed"] == 0],
        "levels_completed": sum(r["levels_completed"] for r in ranked),
        "wall_sec": round(wall, 2),
        "sum_game_sec": round(game_sec, 2),
        "speedup": round(game_sec / wall, 2) if wall > 0 else 0.0,
        "mean_game_score": round(
            sum((r["game_score"] or 0) for r in ranked) / max(len(ranked), 1), 4),
        "official_scorecard_score": official,
        "games_with_official_score": sum(1 for r in ranked
                                         if r["score_source"] == "official"),
        "games_with_estimate_only": sum(1 for r in ranked
                                        if r["score_source"] is None),
        "games": ranked,
    }
    _atomic_write_json(out_dir / "summary.json", summary)

    print("\n========= SUMMARY =========")
    for r in ranked:
        score = "-" if r["game_score"] is None else f"{r['game_score']:.2f}"
        print(f"  {r['game_id']:8} levels={r['levels_completed']:3}  "
              f"act={r['actions']:4}  {r['elapsed_sec']:7.1f}s  "
              f"score={score:>5}  {r['state']}")
    print(f"\n  {len(results)}/{total} games  |  {summary['levels_completed']} levels  "
          f"|  mean game score {summary['mean_game_score']:.3f}  "
          f"|  official {official}")
    print(f"  wall {wall:.1f}s  (sum of per-game {game_sec:.1f}s, "
          f"effective speedup {summary['speedup']}x on {parallel} threads)")
    print(f"  wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
