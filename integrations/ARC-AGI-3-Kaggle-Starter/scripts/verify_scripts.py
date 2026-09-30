"""Verify ScriptBank JSON against live Arcade: each level must raise levels_completed.

Usage:
  python scripts/verify_scripts.py
  python scripts/verify_scripts.py ar25
  python scripts/verify_scripts.py ls20 --strict
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(ROOT))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, GameState
from lingjing_solo.planning.script_bank import load_level_scripts, list_scripted_games


def _apply(env, spec):
    if isinstance(spec, dict):
        name = str(spec.get("action") or "ACTION1").upper()
        ga = getattr(GameAction, name)
        if name == "ACTION6":
            ga.set_data({"x": int(spec.get("x", 32)), "y": int(spec.get("y", 32))})
        return env.step(ga)
    return env.step(getattr(GameAction, str(spec).upper()))


def verify_game(game_id: str, *, strict: bool = False) -> dict:
    scripts = load_level_scripts(game_id, reload=True)
    result = {
        "game_id": game_id,
        "ok": True,
        "levels_checked": 0,
        "levels_passed": 0,
        "final_levels": 0,
        "errors": [],
    }
    if not scripts:
        result["ok"] = False
        result["errors"].append("no scripts found")
        return result

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make(game_id)
    if env is None:
        result["ok"] = False
        result["errors"].append("make() failed")
        return result

    obs = env.step(GameAction.RESET)
    keys = sorted(scripts.keys(), key=lambda k: int(k))
    for key in keys:
        expect_from = int(key)
        lv0 = int(getattr(obs, "levels_completed", 0) or 0)
        if lv0 != expect_from:
            msg = f"L{key}: expected levels_completed={expect_from} before script, got {lv0}"
            result["errors"].append(msg)
            result["ok"] = False
            if strict:
                break
            # try to continue anyway
        for i, step in enumerate(scripts[key]):
            obs = _apply(env, step)
            if obs is None:
                result["errors"].append(f"L{key} step {i}: env returned None")
                result["ok"] = False
                break
            st = getattr(obs, "state", None)
            if st == GameState.GAME_OVER:
                result["errors"].append(f"L{key} step {i}: GAME_OVER")
                result["ok"] = False
                break
        result["levels_checked"] += 1
        lv1 = int(getattr(obs, "levels_completed", 0) or 0)
        result["final_levels"] = lv1
        if lv1 > expect_from or st == GameState.WIN:
            result["levels_passed"] += 1
        else:
            result["errors"].append(
                f"L{key}: script finished but levels_completed stayed {lv1} (need >{expect_from})"
            )
            result["ok"] = False
            if strict:
                break
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("game_id", nargs="?", default="")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    games = [args.game_id] if args.game_id else list_scripted_games()
    rows = [verify_game(g, strict=args.strict) for g in games]
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if not all(r["ok"] for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
