"""One OpenAI-compatible chat call with tool support.

Kept separate from the agent so the loop can be exercised without a network:
`agent.DuckAgent` takes any callable with this signature, which is how the
offline tests drive it.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            "Read and run Python against the current game state. Reading and computing "
            "cost nothing; only `action(...)` spends the level budget. The sandbox "
            "exposes `current_frame`, `previous_frame`, `history`, `transitions`, "
            "`frame_diff`, `last_action_call_result`, `valid_actions`, and `action(...)`. "
            "Print whatever you want to see -- the printed output is returned to you."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source to execute."},
            },
            "required": ["code"],
        },
    },
}

_PRIVATE_PREFIXES = ("127.", "10.", "192.168.", "172.16.", "172.17.", "172.18.",
                     "172.19.", "172.2", "172.30.", "172.31.", "localhost")


def _is_private(url: str) -> bool:
    host = url.split("//", 1)[-1].split("/", 1)[0].split(":")[0]
    return host == "localhost" or host.startswith(_PRIVATE_PREFIXES)


def _opener(url: str):
    """Loopback must not go through a system proxy.

    A proxy in front of 127.0.0.1 answers 502 and the failure looks like 'the
    model is down'. This bit us once already.
    """
    if _is_private(url):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


class Endpoint:
    def __init__(self, url: str | None = None, model: str | None = None,
                 key: str | None = None, timeout: float | None = None):
        self.url = (url or os.environ.get("LINGJING_LLM_URL")
                    or os.environ.get("OPENAI_BASE_URL")
                    or "http://127.0.0.1:8001/v1").rstrip("/")
        self.model = (model or os.environ.get("LINGJING_LLM_MODEL")
                      or os.environ.get("OPENAI_MODEL") or "local")
        self.key = (key or os.environ.get("LINGJING_LLM_KEY")
                    or os.environ.get("OPENAI_API_KEY") or "EMPTY")
        self.timeout = float(timeout or os.environ.get("LINGJING_LLM_TIMEOUT") or 120)

    def __repr__(self) -> str:
        return f"<Endpoint {self.url} model={self.model}>"


class Chat:
    """Stateless: the caller owns the message list."""

    def __init__(self, endpoint: Endpoint | None = None, *, temperature: float = 0.6,
                 max_tokens: int = 3072, think: bool = False):
        self.ep = endpoint or Endpoint()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.think = think
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.last_error: str | None = None

    def __call__(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        body: dict = {
            "model": self.ep.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        # Qwen3's chat template enables thinking unless told otherwise, and a
        # thinking reply spends the whole budget before emitting a tool call.
        body["chat_template_kwargs"] = {"enable_thinking": bool(self.think)}

        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.ep.url}/chat/completions", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.ep.key}"})
        self.calls += 1
        t0 = time.perf_counter()
        try:
            with _opener(self.ep.url).open(req, timeout=self.ep.timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            # a server that rejects chat_template_kwargs is common; retry without
            if exc.code == 400 and "chat_template_kwargs" in detail:
                body.pop("chat_template_kwargs", None)
                req.data = json.dumps(body).encode()
                with _opener(self.ep.url).open(req, timeout=self.ep.timeout) as resp:
                    data = json.loads(resp.read().decode())
            else:
                self.last_error = f"HTTP {exc.code}: {detail}"
                raise
        usage = data.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        msg["_latency"] = round(time.perf_counter() - t0, 2)
        return msg

    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
