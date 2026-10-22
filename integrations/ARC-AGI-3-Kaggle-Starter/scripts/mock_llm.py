"""A fake OpenAI endpoint, good enough to run the whole decision layer offline.

No GPU, no weights, no download.  It speaks exactly the two routes the agent uses:

    GET  /v1/models              -> one served id, so `llm_up.py attach` works
    POST /v1/chat/completions    -> {"actions": [...], "why": "..."}

Three ways to decide what to answer:

    --mode cycle    (default) rotate A1..A4, so every action gets tried
    --mode script   replay --script ACTION3,ACTION3,ACTION2, then repeat
    --mode random   uniform random over the offered actions (seeded)

It also honours `chat_template_kwargs`, so you can exercise both the thinking and
the non-thinking code path in `llm_planner` without a real model.

Usage:
    python scripts/mock_llm.py                        # 127.0.0.1:1234
    python scripts/mock_llm.py --port 8001 --mode script --script ACTION2,ACTION2
    python scripts/llm_up.py attach --url http://127.0.0.1:1234/v1 --model mock-arc

Then, with `.env` written by `llm_up`, run the agent as usual.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "mock-arc"
ALL_ACTIONS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]


class State:
    mode = "cycle"
    script: list[str] = []
    rng = random.Random(0)
    n = 0
    offered: list[str] = list(ALL_ACTIONS)
    script_i = 0


def _offered(user: str) -> list[str]:
    """Which actions the game allows, read back from the prompt when it lists them."""
    found = sorted(set(re.findall(r"ACTION[1-7]", user)))
    return found or list(ALL_ACTIONS)


def decide(user: str) -> tuple[list[str], str]:
    offered = _offered(user)
    State.offered = offered
    State.n += 1
    if State.mode == "script" and State.script:
        picks = [State.script[(State.script_i + k) % len(State.script)] for k in range(3)]
        State.script_i += 3
        picks = [p for p in picks if p in offered] or [offered[0]]
        why = f"mock(script) -> {' '.join(picks)}"
    elif State.mode == "random":
        picks = [State.rng.choice(offered) for _ in range(3)]
        why = f"mock(random) -> {' '.join(picks)}"
    else:
        base = State.n * 3
        picks = [offered[(base + k) % len(offered)] for k in range(3)]
        why = f"mock(cycle #{State.n}) -> {' '.join(picks)}"
    return picks, why


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.rstrip("/")
        if path.endswith("/v1/models") or path.endswith("/models"):
            self._send(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "created": int(time.time()),
                 "owned_by": "mock"}]})
        elif path.endswith("/health"):
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("content-length", 0) or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            body = json.loads(raw)
        except Exception:  # noqa: BLE001
            self._send(400, {"error": {"message": "bad json"}})
            return

        model = body.get("model")
        if model and model != MODEL_ID:
            self._send(404, {"error": {"message":
                                       f"model {model!r} not found; serves {MODEL_ID!r}"}})
            return

        user = ""
        for m in body.get("messages") or []:
            if m.get("role") == "user":
                user = m.get("content") or ""
        think = bool((body.get("chat_template_kwargs") or {}).get("enable_thinking"))

        acts, why = decide(user)
        message: dict = {"role": "assistant",
                         "content": json.dumps({"actions": acts, "why": why},
                                               ensure_ascii=False)}
        if think:
            message["reasoning_content"] = (
                f"Offered: {', '.join(State.offered)}. Mode {State.mode}. "
                f"Emitting {len(acts)} actions.")
        print(f"[mock-llm] #{State.n} think={int(think)} -> {acts}", flush=True)
        self._send(200, {
            "id": f"mock-{State.n}", "object": "chat.completion",
            "created": int(time.time()), "model": MODEL_ID,
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": max(1, len(user) // 4), "completion_tokens": 24,
                      "total_tokens": max(1, len(user) // 4) + 24},
        })

    def log_message(self, *args) -> None:
        pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=1234)
    p.add_argument("--mode", default="cycle", choices=["cycle", "script", "random"])
    p.add_argument("--script", default="ACTION2,ACTION2,ACTION3,ACTION4",
                   help="actions replayed in --mode script")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    State.mode = args.mode
    State.script = [a.strip().upper() for a in args.script.split(",") if a.strip()]
    State.rng = random.Random(args.seed)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[mock-llm] serving {MODEL_ID} on http://{args.host}:{args.port}/v1 "
          f"(mode={args.mode})", flush=True)
    print(f"[mock-llm] wire it: python scripts/llm_up.py attach "
          f"--url http://{args.host}:{args.port}/v1 --model {MODEL_ID}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[mock-llm] bye", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
