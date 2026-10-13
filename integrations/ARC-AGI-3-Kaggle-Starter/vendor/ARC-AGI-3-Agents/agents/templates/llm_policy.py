"""Local OpenAI-compatible LLM policy for ARC-AGI-3 (Reki-style JSON action).

Fail-closed: any miss / timeout / illegal action → return None so SmartRouter
keeps CEAX / idle / RESET. Intended for Kaggle offline vLLM (Qwen etc.).

Env:
  LINGJING_LLM          on|off (caller usually gates this)
  LINGJING_LLM_URL      default http://127.0.0.1:1234/v1
  LINGJING_LLM_MODEL    default vrfai/Qwen3.6-27B-FP8
  LINGJING_LLM_KEY      Bearer token (optional; vLLM may ignore)
  LINGJING_LLM_TIMEOUT  seconds (default 45)
  LINGJING_LLM_MAX_TOKENS  default 256
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Optional


def llm_enabled() -> bool:
    return str(os.environ.get("LINGJING_LLM", "off")).strip().lower() in {
        "1",
        "on",
        "true",
        "yes",
    }


def _cfg() -> dict[str, Any]:
    return {
        "url": str(
            os.environ.get("LINGJING_LLM_URL", "http://127.0.0.1:1234/v1")
        ).rstrip("/"),
        "model": str(
            os.environ.get("LINGJING_LLM_MODEL", "vrfai/Qwen3.6-27B-FP8")
        ),
        "key": str(os.environ.get("LINGJING_LLM_KEY", "EMPTY")),
        "timeout": float(os.environ.get("LINGJING_LLM_TIMEOUT", "45") or 45),
        "max_tokens": int(os.environ.get("LINGJING_LLM_MAX_TOKENS", "256") or 256),
    }


def grid_to_ascii(grid: Any, max_side: int = 64) -> str:
    """Compact 64x64 (or smaller) digit grid for prompts."""
    if grid is None:
        return "(no grid)"
    try:
        import numpy as np

        arr = np.asarray(grid)
        if arr.ndim != 2:
            return "(bad grid)"
        h, w = arr.shape
        h = min(int(h), max_side)
        w = min(int(w), max_side)
        lines = []
        for r in range(h):
            lines.append("".join(str(int(arr[r, c]) % 16) for c in range(w)))
        return "\n".join(lines)
    except Exception:
        return "(grid error)"


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    # strip common fences / think tags
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I)
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _chat(messages: list[dict[str, str]]) -> Optional[str]:
    cfg = _cfg()
    url = f"{cfg['url']}/chat/completions"
    body = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": cfg["max_tokens"],
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg['key']}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg["timeout"]) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"[llm_policy] request failed: {type(exc).__name__}: {exc}", flush=True)
        return None
    try:
        payload = json.loads(raw)
        return str(payload["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        print(f"[llm_policy] bad response: {type(exc).__name__}: {exc}", flush=True)
        return None


def choose_json_action(
    *,
    game_id: str,
    levels: int,
    valid_actions: list[str],
    grid: Any,
    note: str = "",
) -> Optional[dict[str, Any]]:
    """Ask the local LLM for one JSON action. Returns dict or None."""
    if not valid_actions:
        return None
    ascii_board = grid_to_ascii(grid)
    system = (
        "You play ARC-AGI-3 interactive games. Reply with ONE JSON object only. "
        'Schema: {"action":"ACTION1..ACTION6|RESET","x":0-63,"y":0-63,"why":"..."}. '
        "ACTION6 requires integer x,y. Prefer legal actions. Be action-efficient."
    )
    user = (
        f"game_id={game_id} levels_completed={levels}\n"
        f"legal_actions={valid_actions}\n"
        f"note={note or 'none'}\n"
        f"grid_ascii (row-major digits):\n{ascii_board}\n"
        "Return JSON only."
    )
    content = _chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    if not content:
        return None
    obj = _extract_json(content)
    if not obj:
        print("[llm_policy] no JSON in model output", flush=True)
        return None
    return obj


def to_game_action(
    obj: dict[str, Any],
    valid_actions: list[str],
    game_action_cls: Any,
) -> Any:
    """Convert JSON dict → GameAction; None if illegal."""
    if not obj or game_action_cls is None:
        return None
    name = str(obj.get("action") or obj.get("name") or "").strip().upper()
    if name.isdigit():
        name = f"ACTION{int(name)}"
    # map duck-style names
    alias = {
        "UP": "ACTION1",
        "DOWN": "ACTION2",
        "LEFT": "ACTION3",
        "RIGHT": "ACTION4",
        "SPACE": "ACTION5",
        "MOUSE": "ACTION6",
        "CLICK": "ACTION6",
    }
    name = alias.get(name, name)
    legal = {a.upper() for a in valid_actions}
    if name == "RESET":
        try:
            act = game_action_cls.RESET
            act.reasoning = {"text": f"llm_policy:{obj.get('why', '')}"}
            return act
        except Exception:
            return None
    if name not in legal:
        return None
    try:
        act = getattr(game_action_cls, name)
    except AttributeError:
        return None
    if name == "ACTION6":
        try:
            x = int(obj.get("x", obj.get("col", 32)))
            y = int(obj.get("y", obj.get("row", 32)))
        except (TypeError, ValueError):
            return None
        x = max(0, min(63, x))
        y = max(0, min(63, y))
        act.set_data({"x": x, "y": y})
    try:
        act.reasoning = {"text": f"llm_policy:{obj.get('why', '')}"}
    except Exception:
        pass
    return act


def propose_action(
    *,
    game_id: str,
    levels: int,
    valid_actions: list[str],
    grid: Any,
    game_action_cls: Any,
    note: str = "",
) -> Any:
    """End-to-end: LLM JSON → GameAction or None."""
    if not llm_enabled():
        return None
    obj = choose_json_action(
        game_id=game_id,
        levels=levels,
        valid_actions=valid_actions,
        grid=grid,
        note=note,
    )
    if not obj:
        return None
    return to_game_action(obj, valid_actions, game_action_cls)
