"""A minimal OpenAI-compatible endpoint that emits tool calls.

Purpose: prove the whole chain works -- HTTP, tool_calls, argument parsing, the
turn boundary -- without needing a model. Capacities are the model's job; this
only checks the plumbing.

    python scripts/mock_duck_server.py --port 18080
    python scripts/duck_play.py --game ar25 --backend openai \
        --url http://127.0.0.1:18080/v1 --model mock --steps 8

It answers in script order: two reads, then one action per turn, and repeats the
last action forever after. Deliberately dull -- anything clever here would be
testing this file instead of the harness.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCRIPT = [
    "print('shape', current_frame.shape)\n"
    "print('valid', valid_actions)\n"
    "print(current_frame.ascii[:120])",
    "print('objects', [n['pixels'] for n in current_frame.segmentation['nodes']][:8])",
    "def marker_pos(frame):\n"
    "    rows = frame.ascii.splitlines()\n"
    "    for r, line in enumerate(rows):\n"
    "        c = line.find('5')\n"
    "        if c >= 0:\n"
    "            return (r, c)\n"
    "    return None\n"
    "print('marker', marker_pos(current_frame))",
    "print('marker again', marker_pos(current_frame))\naction('DOWN')",
    "action('DOWN')",
    "print(frame_diff())\naction('DOWN')",
]

_last_action = "action('RIGHT')"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    turn = 0

    def log_message(self, *args):  # keep the console readable
        pass

    def do_GET(self):
        if self.path.endswith("/models"):
            self._json(200, {"data": [{"id": "mock"}]})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        msgs = body.get("messages") or []
        tools = body.get("tools") or []

        if not tools:
            # summary request: no tools offered
            reply = {"role": "assistant",
                     "content": "Mock summary: nothing established yet, still probing."}
            self._reply(msgs, reply, prompt_guess=len(msgs))
            return

        idx = Handler.turn
        Handler.turn += 1
        code = SCRIPT[idx] if idx < len(SCRIPT) else _last_action
        reply = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": f"call_{idx}",
                "type": "function",
                "function": {"name": "python", "arguments": json.dumps({"code": code})},
            }],
        }
        self._reply(msgs, reply, prompt_guess=len(msgs))

    def _reply(self, msgs, reply, *, prompt_guess: int):
        payload = {
            "id": "mock-1",
            "object": "chat.completion",
            "model": "mock",
            "choices": [{"index": 0, "message": reply, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": prompt_guess * 40,
                      "completion_tokens": 40,
                      "total_tokens": prompt_guess * 40 + 40},
        }
        self._json(200, payload)

    def _json(self, code: int, obj) -> None:
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18080)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[mock] listening on http://127.0.0.1:{args.port}/v1", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
