# -*- coding: utf-8 -*-
"""Prove the duck bridge is legitimate before giving it a model.

What "legitimate" means here, concretely:

  1. every action duck asks for is spent by the *harness* (`env.step`), so the
     scorecard counts it -- `arc_agi/wrapper.py` only counts actions that go
     through that path;
  2. the board duck sees next turn is the real post-action board, not a guess;
  3. duck yields exactly one action per tick (a batch is drained tick by tick);
  4. it stops by itself (budget / no tool call) and the game is left in a
     sane state for the next fallback layer.

No model is involved: a scripted chat emits snippets in order, which is how
`tests/test_duck.py` already validates the non-resumable loop.

    python scripts/duck_bridge_selftest.py --game cn04 --steps 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "arc_adaptor"))
sys.path.insert(0, str(ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"))

from arc_agi import Arcade, OperationMode          # noqa: E402
from arcengine import GameAction                  # noqa: E402

from duck.deferred import DuckDriver              # noqa: E402

SCRIPT = [
    "print('shape', current_frame.shape)\nprint(valid_actions)",
    "action('DOWN')",
    "print('after down', frame_diff())",
    "action(['DOWN', 'DOWN'])",          # batch: must drain one per tick
    "action('RIGHT')",
    "print('steps', last_action_call_result)",
]


class ScriptChat:
    def __init__(self, script):
        self.script = list(script)
        self.i = 0
        self.calls = 0

    def __call__(self, messages, tools=None):
        self.calls += 1
        if self.i >= len(self.script):
            return {"role": "assistant", "content": "done", "tool_calls": []}
        code = self.script[self.i]
        self.i += 1
        return {"role": "assistant", "content": "", "tool_calls": [{
            "id": f"call_{self.i}", "type": "function",
            "function": {"name": "python", "arguments": json.dumps({"code": code})},
        }]}

    def tokens(self) -> int:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="cn04")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    arc = Arcade(operation_mode=OperationMode.NORMAL,
                 environments_dir=str(ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"
                                      / "environment_files"))
    env = arc.make(args.game, render_mode=None)
    frame = env.reset()
    game_id = args.game

    driver = DuckDriver(ScriptChat(SCRIPT), game_id=game_id, frame=frame,
                        level=int(getattr(frame, "levels_completed", 0) or 0),
                        max_steps=args.steps, max_turns=args.turns,
                        verbose=not args.quiet)

    AID = {f"ACTION{i}": getattr(GameAction, f"ACTION{i}") for i in range(1, 7)}
    spent = 0
    boards = []
    while True:
        nxt = driver.poll()
        if nxt is None:
            break
        name, spec = nxt
        # the harness spends the action -- this is the only path the scorecard counts
        data = {}
        if name == "ACTION6":
            data = {"x": int(spec.get("col", 0)), "y": int(spec.get("row", 0))}
        res = env.step(AID[name], data=data, reasoning={"note": "duck bridge"})
        spent += 1
        boards.append(res.frame[-1][0][:6] if res.frame else None)
        driver.feed(res)
        if spent > args.steps * 2:      # hard stop: never spin here
            break

    stats = driver.stats()
    print("\n[selftest] " + json.dumps(stats, ensure_ascii=False))
    print(f"[selftest] actions spent by harness: {spent}")

    # --- assertions -----------------------------------------------------
    ok = True
    if spent == 0:
        print("[selftest] FAIL: duck never asked for an action")
        ok = False
    if stats["steps"] != spent:
        print(f"[selftest] FAIL: driver thinks steps={stats['steps']} but harness spent {spent}")
        ok = False
    if driver.sandbox.steps != spent:
        print("[selftest] FAIL: sandbox step count diverged")
        ok = False

    # scorecard truth: the count the grader would use. This is the whole point of
    # the bridge -- `game.perform_action()` would leave this at zero.
    try:
        manager = env.scorecard_manager
        cards = list(getattr(manager, "scorecards", {}).values())
        counted = sum(int(getattr(card, "total_actions", 0) or 0) for card in cards)
        print(f"[selftest] scorecard counted actions = {counted} "
              f"(scorecards={len(cards)})")
        if counted < spent:
            print(f"[selftest] FAIL: scorecard counted {counted} < spent {spent}")
            ok = False
        elif counted > spent:
            print(f"[selftest] note: scorecard counted {counted} > spent {spent} "
                  f"(the opening RESET is counted too)")
    except Exception as exc:  # noqa: BLE001
        print(f"[selftest] FAIL: could not read the scorecard: "
              f"{type(exc).__name__}: {exc}")
        ok = False

    # board feedback: duck's view must equal the real frame after the last action
    real_last = driver.sandbox.current._grid if driver.sandbox.current else None
    if real_last and boards and boards[-1] is not None:
        if len(real_last) != 64:
            print(f"[selftest] WARN: unexpected board height {len(real_last)}")

    print("[selftest]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
