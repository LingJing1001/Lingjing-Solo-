"""最小 mock OpenAI 兼容端点（纯 stdlib，不装包）。

用途：duck 接入测试。返回固定动作循环（ACTION3→ACTION2→ACTION6→ACTION3），
让 DuckDriver 真跑起来，验证反思代码路径完整可达。
真模型不会这么笨，但 mock 能让 duck 起来就能验证：
- DuckBridge 在端点可达时 enable=True
- DuckDriver 真接管 tick 循环
- game_over 钩子触发 FailedRoutesMemory.record
- _refresh_system 钉 KNOWN DEAD-ENDS 块
- atexit 落盘 + 下次 load_from_dir 读回

跑法（一个终端起服务，另一个终端跑 ar25）：
    python tests/mock_llm_server.py 8001
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

# 固定循环动作序列——让 duck 真发出 action 调用
ACTIONS = ["ACTION3", "ACTION2", "ACTION6", "ACTION3", "ACTION5", "ACTION4"]
_call_count = {"n": 0}


def _resp(handler, status: int, body: dict):
    data = json.dumps(body).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, *args, **kwargs):
        pass  # 静音

    def do_POST(self):
        n = _call_count["n"]
        _call_count["n"] += 1
        # 解析请求体（仅看路径，不真处理）
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        # 必须返回一个 tool_call：name=python, code=action(ACTION_X)
        action = ACTIONS[n % len(ACTIONS)]
        code = f"action({action!r})"
        msg = {
            "role": "assistant",
            "content": f"// mock turn {n}, action={action}",
            "tool_calls": [{
                "id": f"call_{n}",
                "type": "function",
                "function": {"name": "python", "arguments": json.dumps({"code": code})},
            }],
        }
        _resp(self, 200, {
            "id": f"chatcmpl-{n}",
            "object": "chat.completion",
            "model": "mock-qwen3",
            "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        })

    def do_GET(self):
        # /v1/models 端点：让 _probe 能拿到可达信号
        _resp(self, 200, {"object": "list", "data": [{"id": "mock-qwen3"}]})


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    srv = HTTPServer(("127.0.0.1", port), MockHandler)
    print(f"[mock-llm] listening on http://127.0.0.1:{port}/v1", flush=True)
    print(f"[mock-llm] returns {len(ACTIONS)}-action cycle: {ACTIONS}", flush=True)
    print(f"[mock-llm] for duck integration test only; not a real LLM", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()