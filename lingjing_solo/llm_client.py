"""OpenAI-compatible LLM client for R5 advisor (local only; Kaggle forces offline)."""
from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


def load_dotenv(path: Optional[Path] = None) -> None:
    root = Path(__file__).resolve().parents[1]
    env_path = path or (root / ".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        existing = os.environ.get(k, "")
        placeholder = (
            len(existing) < 16
            or existing.lower() in {"sk-xxxx", "sk-xxx", "your_api_key"}
            or existing.lower().startswith("replace_with")
        )
        if k and (k not in os.environ or placeholder):
            os.environ[k] = v


def _normalize_base(url: str) -> str:
    u = url.strip().rstrip("/")
    if u.endswith("/chat/completions"):
        u = u[: -len("/chat/completions")]
    return u


def _bad_key(k: str) -> bool:
    kk = k.lower()
    return len(k) < 16 or kk in {"sk-xxxx", "sk-xxx"} or kk.startswith("replace_with")


def resolve_credentials() -> tuple[str, str, str]:
    load_dotenv()
    model = (
        os.environ.get("AETHER_LLM_MODEL", "").strip()
        or os.environ.get("R5_LLM_MODEL", "").strip()
    )

    # Prefer MiniMax when key present
    mm = os.environ.get("MINIMAX_API_KEY", "").strip()
    if mm and not _bad_key(mm):
        base = _normalize_base(
            os.environ.get("OPENAI_BASE_URL", "")
            or os.environ.get("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")
        )
        if "minimax" not in base:
            base = "https://api.minimaxi.com/v1"
        return mm, base, model or "MiniMax-M3"

    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base = _normalize_base(os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    if openai_key and not _bad_key(openai_key):
        if "minimax" in base:
            return openai_key, base, model or "MiniMax-M3"
        return openai_key, base, model or (
            "deepseek-chat" if "deepseek" in base else "gpt-4o-mini"
        )

    ds = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if ds and not _bad_key(ds):
        return (
            ds,
            _normalize_base(os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")),
            model or "deepseek-chat",
        )
    return "", base, model or "gpt-4o-mini"


def _strip_think(text: str) -> str:
    """Remove MiniMax/DeepSeek style <think>...</think> blocks if mixed into content."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"<thinking>[\s\S]*?</thinking>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip() or text.strip()


def chat_completion(
    prompt: str,
    *,
    system: str = "你是严谨的因果推理与工程复盘顾问。用简体中文、Markdown 五问结构回答。",
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    timeout: float = 180.0,
    model: Optional[str] = None,
) -> str:
    key, base, model_default = resolve_credentials()
    if not key:
        raise RuntimeError("缺少 MINIMAX_API_KEY / OPENAI_API_KEY，无法跑 LLM。")
    model = (model or model_default).strip()

    max_tokens = max_tokens or int(os.environ.get("AETHER_LLM_MAX_TOKENS", "4096"))
    body: dict = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    if "deepseek" in base and os.environ.get("AETHER_LLM_THINKING", "0") == "1":
        body["reasoning_effort"] = os.environ.get("AETHER_LLM_REASONING_EFFORT", "medium")
    # MiniMax-M3: prefer final answer without long think dump in content
    if "minimax" in base.lower() or "MiniMax" in model:
        body["thinking"] = {"type": "disabled"}

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
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"LLM HTTP {e.code}: {detail}") from e

    msg = raw["choices"][0]["message"]
    content = msg.get("content") or ""
    if not content and msg.get("reasoning_content"):
        content = str(msg["reasoning_content"])
    if not content:
        raise RuntimeError(f"LLM 返回空内容: keys={list(msg.keys())}")
    return _strip_think(content)


def make_llm_fn():
    """返回 inject 给 R5UpdateReflectAgent 的 llm_fn(prompt)->str。"""

    def _fn(prompt: str) -> str:
        return chat_completion(prompt)

    return _fn


__all__ = ["load_dotenv", "resolve_credentials", "chat_completion", "make_llm_fn"]
