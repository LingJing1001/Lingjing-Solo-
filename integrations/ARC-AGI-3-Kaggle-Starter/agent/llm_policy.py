"""Local OpenAI-compatible LLM policy for ARC-AGI-3 (Reki-style JSON action).

Fail-closed: any miss / timeout / illegal action → return None so SmartRouter
keeps CEAX / idle / RESET. Intended for Kaggle offline vLLM (Qwen etc.).

The prompt always includes an ASCII grid. When LINGJING_LLM_VISION is on
(default), the same grid is also rendered to a PNG and sent as an image part.
If the channel rejects images, the request retries once as text-only.

Env:
  LINGJING_LLM          on|off (caller usually gates this)
  LINGJING_LLM_URL      default http://127.0.0.1:1234/v1
  LINGJING_LLM_MODEL    default vrfai/Qwen3.6-27B-FP8
  LINGJING_LLM_KEY      Bearer token (optional; vLLM may ignore)
  LINGJING_LLM_TIMEOUT  seconds (default 45)
  LINGJING_LLM_MAX_TOKENS  default 256
  LINGJING_LLM_VISION   on|off (default on) — attach base64 PNG
"""
from __future__ import annotations

import base64
import json
import os
import re
import struct
import urllib.error
import urllib.request
import zlib
from typing import Any, Optional

# ARC-like 0–15 colors. Index = cell value mod 16.
_PALETTE = (
    (0, 0, 0),
    (0, 116, 217),
    (255, 65, 54),
    (46, 204, 64),
    (255, 220, 0),
    (170, 170, 170),
    (240, 18, 190),
    (255, 133, 27),
    (127, 219, 255),
    (177, 13, 201),
    (133, 20, 75),
    (255, 255, 255),
    (0, 0, 128),
    (0, 128, 0),
    (128, 0, 0),
    (128, 128, 128),
)


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


def vision_enabled() -> bool:
    return str(os.environ.get("LINGJING_LLM_VISION", "on")).strip().lower() not in {
        "0",
        "off",
        "false",
        "no",
        "none",
    }


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def grid_png_b64(grid: Any, scale: int = 4, max_side: int = 64) -> Optional[str]:
    """Nearest-neighbor RGB PNG of the grid, base64, or None if unusable."""
    if grid is None:
        return None
    try:
        import numpy as np

        arr = np.asarray(grid)
        if arr.ndim != 2 or arr.size == 0:
            return None
        h = min(int(arr.shape[0]), max_side)
        w = min(int(arr.shape[1]), max_side)
        scale = max(1, int(scale))
        out_h, out_w = h * scale, w * scale
        raw = bytearray()
        for r in range(h):
            expanded = bytearray()
            for c in range(w):
                color = _PALETTE[int(arr[r, c]) % 16]
                expanded.extend(bytes(color) * scale)
            scan = b"\x00" + bytes(expanded)
            raw.extend(scan * scale)
        if len(raw) != out_h * (1 + out_w * 3):
            return None
        ihdr = struct.pack(">IIBBBBB", out_w, out_h, 8, 2, 0, 0, 0)
        png = (
            b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 6))
            + _png_chunk(b"IEND", b"")
        )
        return base64.b64encode(png).decode("ascii")
    except Exception:
        return None


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


def _extract_json_any(text: str) -> Any:
    """Like _extract_json but also accepts a top-level array (plan format)."""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I)
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pat in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        m = re.search(pat, text)
        if not m:
            continue
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return None


def batch_limit() -> int:
    """How many actions one call may return. 1 = batch protocol off."""
    try:
        n = int(os.environ.get("LINGJING_LLM_BATCH", "4") or 4)
    except (TypeError, ValueError):
        n = 4
    return max(1, min(12, n))


_ACTION_ALIAS = {
    "UP": "ACTION1",
    "DOWN": "ACTION2",
    "LEFT": "ACTION3",
    "RIGHT": "ACTION4",
    "SPACE": "ACTION5",
    "NOOP": "ACTION5",
    "MOUSE": "ACTION6",
    "CLICK": "ACTION6",
}


def _click_xy(spec: str) -> Optional[tuple[int, int]]:
    """`ACTION6@10,20` / `ACTION6(10,20)` / `click 10 20` → (10, 20)."""
    m = re.search(r"@\s*(-?\d+)\s*[,;x\s]\s*(-?\d+)", spec, re.I)
    if not m:
        m = re.search(r"[\(\[]\s*(-?\d+)\s*[,;\s]\s*(-?\d+)\s*[\)\]]", spec)
    if not m:
        return None
    try:
        return int(m.group(1)), int(m.group(2))
    except (TypeError, ValueError):
        return None


def normalize_action_spec(item: Any) -> Optional[dict[str, Any]]:
    """Any model-emitted action form → {'action':name,'x':int,'y':int,'why':str}.

    Accepted: "ACTION6@3,4" | "ACTION1" | {"action":..,"x":..} | {"a":..} |
    ["ACTION6", 3, 4] | {"click": [3, 4]}. Coordinates are clamped to 0..63.
    """
    why = ""
    raw_name: Any = None
    x: Any = None
    y: Any = None
    if isinstance(item, str):
        spec = item.strip()
        why = spec
        xy = _click_xy(spec)
        head = re.split(r"[@\(\[]", spec, maxsplit=1)[0].strip()
        raw_name = head or spec
        if xy:
            x, y = xy
    elif isinstance(item, dict):
        why = str(item.get("why") or item.get("reason") or item.get("note") or "")
        raw_name = (
            item.get("action")
            or item.get("a")
            or item.get("name")
            or item.get("type")
            or item.get("act")
        )
        if raw_name is None and len(item) == 1:
            raw_name = next(iter(item))
        pair = item.get("xy") or item.get("pos") or item.get("click")
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            x, y = pair[0], pair[1]
        else:
            x = item.get("x", item.get("col"))
            y = item.get("y", item.get("row"))
    elif isinstance(item, (list, tuple)) and item:
        raw_name = item[0]
        if len(item) >= 3:
            x, y = item[1], item[2]
        elif len(item) == 2 and isinstance(item[1], (list, tuple)) and len(item[1]) >= 2:
            x, y = item[1][0], item[1][1]
    if raw_name is None:
        return None
    name = str(raw_name).strip().upper()
    if name.isdigit():
        name = f"ACTION{int(name)}"
    name = _ACTION_ALIAS.get(name, name)
    out: dict[str, Any] = {"action": name, "why": why[:120]}
    if x is not None and y is not None:
        try:
            out["x"] = max(0, min(63, int(x)))
            out["y"] = max(0, min(63, int(y)))
        except (TypeError, ValueError):
            pass
    return out


def parse_plan(
    obj: Any, valid_actions: list[str], limit: int
) -> list[dict[str, Any]]:
    """Model JSON → validated action list. Truncates at the first bad step."""
    if isinstance(obj, list):
        raw: Any = obj
    elif isinstance(obj, dict):
        raw = obj.get("actions")
        if raw is None:
            raw = obj.get("plan")
        if raw is None:
            raw = obj.get("sequence")
        if raw is None:
            raw = [obj]  # legacy single-action schema
    else:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    legal = {a.upper() for a in valid_actions}
    out: list[dict[str, Any]] = []
    for item in raw[: max(1, int(limit))]:
        spec = normalize_action_spec(item)
        if not spec:
            break
        name = spec["action"]
        # A plan containing RESET would wipe the level it is trying to solve.
        if name == "RESET":
            break
        if legal and name not in legal:
            break
        if name == "ACTION6" and "x" not in spec:
            break
        out.append(spec)
    return out


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(parts)
    return ""


def _strip_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            out.append({**msg, "content": _message_text(content)})
        else:
            out.append(msg)
    return out


def _post_chat(messages: list[dict[str, Any]]) -> tuple[Optional[str], bool]:
    """Return (text, retry_without_image). Retry only on image-reject HTTP codes."""
    cfg = _cfg()
    url = f"{cfg['url']}/chat/completions"
    body = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": cfg["max_tokens"],
        # Qwen3.x thinking can consume the whole max_tokens budget and leave
        # content empty → "no JSON". Disable when the channel supports it.
        "chat_template_kwargs": {"enable_thinking": False},
        "enable_thinking": False,
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
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:180]
        except Exception:
            detail = ""
        print(
            f"[llm_policy] http {exc.code}: {detail}",
            flush=True,
        )
        return None, exc.code in {400, 404, 415, 422}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"[llm_policy] request failed: {type(exc).__name__}: {exc}", flush=True)
        return None, False
    try:
        payload = json.loads(raw)
        msg = payload["choices"][0]["message"]
        content = msg.get("content")
        if content is None or (isinstance(content, str) and not content.strip()):
            # Some Qwen channels put draft text in `reasoning` and leave content empty.
            content = msg.get("reasoning") or msg.get("refusal") or ""
        text = str(content or "")
        return (text or None), False
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        print(f"[llm_policy] bad response: {type(exc).__name__}: {exc}", flush=True)
        return None, False


def _chat(messages: list[dict[str, Any]]) -> Optional[str]:
    """POST chat. Image parts that the channel rejects are retried as text."""
    content, retry = _post_chat(messages)
    if content:
        return content
    has_image = any(
        isinstance(msg.get("content"), list)
        and any(isinstance(p, dict) and p.get("type") == "image_url" for p in msg["content"])
        for msg in messages
    )
    if not (retry and has_image):
        return None
    print("[llm_policy] vision rejected → retry text-only", flush=True)
    text, _ = _post_chat(_strip_images(messages))
    return text


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
    png_b64 = grid_png_b64(grid) if vision_enabled() else None
    system = (
        "You play ARC-AGI-3 interactive games. Reply with ONE JSON object only. "
        'Schema: {"action":"ACTION1..ACTION6|RESET","x":0-63,"y":0-63,"why":"..."}. '
        "ACTION6 requires integer x,y. Prefer legal actions. Be action-efficient. "
        "If an image is attached, it is the current 64x64 grid (x right, y down); "
        "click coordinates must match that image."
    )
    user = (
        f"game_id={game_id} levels_completed={levels}\n"
        f"legal_actions={valid_actions}\n"
        f"engine_note={note or 'none'}\n"
        f"grid_ascii (row-major digits):\n{ascii_board}\n"
        "Return JSON only."
    )
    user_content: Any = user
    if png_b64:
        user_content = [
            {"type": "text", "text": user},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{png_b64}"},
            },
        ]
    content = _chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
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


def choose_json_plan(
    *,
    game_id: str,
    levels: int,
    valid_actions: list[str],
    grid: Any,
    note: str = "",
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Ask the LLM for a SHORT action burst instead of a single action.

    One call buys up to `limit` steps, which is what makes `(human/used)^2`
    scoring reachable: fewer round-trips for the same behaviour, and the model
    can plan a verified micro-sequence ("click A then B") in one shot.
    """
    if not valid_actions:
        return []
    n = max(1, int(limit if limit is not None else batch_limit()))
    ascii_board = grid_to_ascii(grid)
    png_b64 = grid_png_b64(grid) if vision_enabled() else None
    if n == 1:
        system = (
            "You play ARC-AGI-3 interactive games. Reply with ONE JSON object only. "
            'Schema: {"actions":["ACTION1"],"why":"..."}. '
            "ACTION6 needs coordinates as ACTION6@x,y with x,y in 0-63. "
            "Never emit RESET. Be action-efficient."
        )
    else:
        system = (
            "You play ARC-AGI-3 interactive games. Reply with ONE JSON object only, no prose.\n"
            'Schema: {"actions":["ACTION1","ACTION3","ACTION6@12,30"],"why":"<one short line>"}\n'
            f"Give 1-{n} actions, executed IN ORDER starting from the current frame. "
            "ACTION6 (mouse click) must be written as ACTION6@x,y with integer x,y in 0-63. "
            "Only the names listed in legal_actions are accepted; never emit RESET. "
            "STOP the list as soon as the outcome becomes unpredictable - a wrong step "
            "costs as much score as a missing one, so prefer a short burst you are sure of. "
            "If the board looks unchanged and you have no verifiable idea, return 1 action only. "
            "If an image is attached, it is the current grid (x right, y down); "
            "click coordinates must match that image."
        )
    user = (
        f"game_id={game_id} levels_completed={levels}\n"
        f"legal_actions={valid_actions}\n"
        f"engine_note={note or 'none'}\n"
        f"grid_ascii (row-major digits):\n{ascii_board}\n"
        "Return JSON only."
    )
    user_content: Any = user
    if png_b64:
        user_content = [
            {"type": "text", "text": user},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{png_b64}"},
            },
        ]
    content = _chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
    )
    if not content:
        return []
    obj = _extract_json_any(content)
    if obj is None:
        print("[llm_policy] no JSON in plan output", flush=True)
        return []
    plan = parse_plan(obj, valid_actions, n)
    if not plan:
        print(f"[llm_policy] plan rejected raw={str(obj)[:160]}", flush=True)
    return plan


def propose_plan(
    *,
    game_id: str,
    levels: int,
    valid_actions: list[str],
    note: str = "",
    grid: Any,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """End-to-end batch: LLM JSON → list of normalized action dicts."""
    if not llm_enabled():
        return []
    try:
        return choose_json_plan(
            game_id=game_id,
            levels=levels,
            valid_actions=valid_actions,
            grid=grid,
            note=note,
            limit=limit,
        )
    except Exception as exc:  # noqa: BLE001 — never kill the game thread
        print(f"[llm_policy] plan error: {type(exc).__name__}: {exc}", flush=True)
        return []



def propose_action(
    *,
    game_id: str,
    levels: int,
    valid_actions: list[str],
    grid: Any,
    game_action_cls: Any,
    note: str = "",
) -> Any:
    """End-to-end single action: LLM JSON → GameAction or None.

    Kept for the non-batch call sites. Internally it now asks the batch schema
    with limit=1 and takes the first step, so both paths share one parser.
    """
    if not llm_enabled():
        return None
    plan = propose_plan(
        game_id=game_id,
        levels=levels,
        valid_actions=valid_actions,
        grid=grid,
        note=note,
        limit=1,
    )
    if not plan:
        return None
    return to_game_action(plan[0], valid_actions, game_action_cls)
