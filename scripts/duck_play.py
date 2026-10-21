"""Play one ARC-AGI-3 game with the champion-style loop.

Two backends:

    --backend script    no network. A fixed snippet list drives the loop, which is
                        how the sandbox and the turn boundary get tested.
    --backend openai    a real OpenAI-compatible endpoint (llama.cpp, vllm, sglang,
                        or anything else that speaks /v1/chat/completions).

Examples
--------
    # offline: does the loop run at all?
    python scripts/duck_play.py --game ar25 --steps 12 --backend script

    # local llama.cpp / vllm
    python scripts/duck_play.py --game ar25 --steps 60 \
        --backend openai --url http://127.0.0.1:8001/v1 --model local
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "arc_adaptor"))
sys.path.insert(0, str(ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"))

from duck.agent import DuckAgent          # noqa: E402
from duck.chat import Chat, Endpoint      # noqa: E402
from duck.env import ArcEnv               # noqa: E402

# A snippet list that exercises every path worth testing offline: a pure read, a
# read that calls a helper, an action, a repeated action from the same board
# (must be refused), a batch, and a read after the board moved.
SCRIPT = [
    "print('shape', current_frame.shape)\nprint(current_frame.ascii[:240])",
    "print('valid', valid_actions)\nprint('nodes', len(current_frame.segmentation['nodes']))",
    "action('ACTION2')",
    "action('ACTION2')",            # same board again -> RepeatedActionInStateError
    "print('after', current_frame.shape)\nprint(frame_diff())",
    "action(['ACTION3', 'ACTION3'])",
    "def count_colours(frame):\n"
    "    from collections import Counter\n"
    "    return Counter(c for row in frame.ascii.splitlines() for c in row).most_common(4)\n"
    "print(count_colours(current_frame))",
    "print('retained worked:', count_colours(current_frame))",
    "action('ACTION1')",
    "print('steps', last_action_call_result)",
]


class ScriptChat:
    """Returns the next snippet, shaped exactly like a tool-calling reply."""

    def __init__(self, script: list[str]):
        self.script = script
        self.i = 0
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def __call__(self, messages, tools=None):
        self.calls += 1
        if self.i >= len(self.script):
            return {"role": "assistant", "content": "done", "tool_calls": []}
        code = self.script[self.i]
        self.i += 1
        self.completion_tokens += len(code) // 4
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": f"call_{self.i}",
                "type": "function",
                "function": {"name": "python", "arguments": json.dumps({"code": code})},
            }],
            "_latency": 0.0,
        }

    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ar25")
    ap.add_argument("--steps", type=int, default=40, help="action budget")
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--backend", choices=["script", "openai"], default="script")
    ap.add_argument("--url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--think", action="store_true", help="let a Qwen3-style model think")
    ap.add_argument("--summary-tokens", type=int, default=60_000)
    ap.add_argument("--pin-policy", choices=["drain", "level"], default="drain",
                    help="when the solved-level block is written into the system "
                         "prompt. 'drain' (default) writes it only when the history is "
                         "compacted -- cache-safe, same as upstream. 'level' writes it "
                         "at each level-up: right for short runs that never compact.")
    ap.add_argument("--no-memory", action="store_true",
                    help="disable solved-level memory (the A/B arm)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    env = ArcEnv(args.game)
    chat = (ScriptChat(SCRIPT) if args.backend == "script"
            else Chat(Endpoint(args.url, args.model), think=args.think))

    agent = DuckAgent(env, chat, max_steps=args.steps, max_turns=args.turns,
                      summary_tokens=args.summary_tokens,
                      level_memory=not args.no_memory,
                      pin_policy=args.pin_policy,
                      verbose=not args.quiet)
    print(f"[duck] game={args.game} backend={args.backend} "
          f"endpoint={getattr(chat, 'ep', 'script')} "
          f"memory={'off' if args.no_memory else args.pin_policy}", flush=True)
    stats = agent.play()
    print("[duck] " + json.dumps(stats, ensure_ascii=False), flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
