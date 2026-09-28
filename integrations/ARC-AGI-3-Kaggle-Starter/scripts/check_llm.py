"""Connectivity check for Aether LLM advisor (OpenAI-compatible APIs).

Usage:
  .\\.venv\\Scripts\\python.exe scripts\\check_llm.py
  .\\.venv\\Scripts\\python.exe scripts\\check_llm.py --base-url https://api.deepseek.com/v1 --model deepseek-chat

Reads (in order): process env, then .env in project root.
Required:
  OPENAI_API_KEY   — API key / token
Optional:
  OPENAI_BASE_URL  — default https://api.openai.com/v1
  AETHER_LLM_MODEL — default gpt-4o-mini
  HTTPS_PROXY / HTTP_PROXY — if you need a local proxy to reach the API
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        # Allow .env to override placeholder env vars
        existing = os.environ.get(k, "")
        placeholder = (
            len(existing) < 16
            or existing.lower() in {"sk-xxxx", "sk-xxx", "your_api_key"}
            or existing.lower().startswith("replace_with")
        )
        if k and (k not in os.environ or placeholder):
            os.environ[k] = v


def resolve_from_env() -> tuple[str, str, str]:
    """Mirror MyAgent credential resolution for the checker."""
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base = normalize_base(os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    model = os.environ.get("AETHER_LLM_MODEL", "").strip()

    def bad(k: str) -> bool:
        kk = k.lower()
        return len(k) < 16 or kk in {"sk-xxxx", "sk-xxx"} or kk.startswith("replace_with")

    if openai_key and not bad(openai_key):
        return openai_key, base, model or (
            "deepseek-v4-pro" if "deepseek" in base else "gpt-4o-mini"
        )
    ds = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if ds and not bad(ds):
        return (
            ds,
            normalize_base(os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")),
            model or "deepseek-v4-pro",
        )
    ork = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if ork and not bad(ork):
        return (
            ork,
            normalize_base(
                os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            ),
            model or "openai/gpt-4o-mini",
        )
    return openai_key, base, model or "gpt-4o-mini"


def normalize_base(url: str) -> str:
    u = url.strip().rstrip("/")
    if u.endswith("/chat/completions"):
        u = u[: -len("/chat/completions")]
    if not u.endswith("/v1") and "openai.com" in u:
        u = u + "/v1"
    return u


def main() -> int:
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--timeout", type=float, default=30.0)
    args = p.parse_args()

    key, base_default, model_default = resolve_from_env()
    base = normalize_base(args.base_url or base_default)
    model = (args.model or model_default).strip()

    print(f"base_url = {base}")
    print(f"model    = {model}")
    print(f"api_key  = {'SET len='+str(len(key)) if key else 'MISSING'}")
    for prox in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
        if os.environ.get(prox):
            print(f"{prox} = {os.environ[prox]}")

    if not key:
        print("\nFAIL: set OPENAI_API_KEY in .env or environment.")
        print("See .env.example for templates (OpenAI / DeepSeek / OpenRouter / 中转).")
        return 2

    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "Reply with exactly: {\"ok\":true}"},
            {"role": "user", "content": "ping"},
        ],
    }
    url = f"{base}/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout, context=ctx) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        content = raw["choices"][0]["message"]["content"]
        print(f"\nOK  HTTP {resp.status}")
        print(f"reply: {content[:200]!r}")
        usage = raw.get("usage")
        if usage:
            print(f"usage: {usage}")
        print("\nNext: set AETHER_LLM=1 and run play_local, e.g.")
        print("  $env:AETHER_LLM='1'")
        print("  .\\.venv\\Scripts\\python.exe scripts\\play_local.py --game sp80 --max-steps 200")
        return 0
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        print(f"\nFAIL HTTP {e.code}: {e.reason}")
        print(detail)
        if e.code in (401, 403):
            print("Hint: API Key 无效或无权访问该模型。")
        elif e.code == 404:
            print("Hint: Base URL 或模型名不对。DeepSeek 用 https://api.deepseek.com/v1 + deepseek-chat")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"\nFAIL {type(e).__name__}: {e}")
        print(
            "Hint: 连不上官方 OpenAI 时，请配置 OPENAI_BASE_URL 为可用中转，"
            "或设置 HTTPS_PROXY=http://127.0.0.1:端口"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
