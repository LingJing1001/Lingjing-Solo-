"""Minimal OpenAI-compatible stub for exercising the batch action protocol.

Returns a fixed 3-step plan so we can verify that:
  call #1 -> ACTION1 is returned and ACTION2/ACTION5 are queued,
  the following two choose_action() ticks consume the queue without a request.
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

PLAN = {"actions": ["ACTION1", "ACTION2", "ACTION5"], "why": "mock batch"}
CALLS = {"n": 0}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("content-length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        CALLS["n"] += 1
        print(f"[mock-llm] request #{CALLS['n']} bytes={len(raw)}", flush=True)
        payload = json.dumps(
            {"choices": [{"message": {"content": json.dumps(PLAN)}}]}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print("mock-llm on 127.0.0.1:1234", flush=True)
    HTTPServer(("127.0.0.1", 1234), Handler).serve_forever()
