"""LLM decision layer: the champion's second half, made pluggable.

Why this file exists
--------------------
The champion (Daniel Franzen, Milestone 2) does not win by searching. It wins by
handing a 128K-context model a **structured view of the board** — `current_frame.code`
(objects, stable picture names, placements), `current_frame.changes_since(...)`, and
`current_frame.rules` (the same `when <action> ... then <change>` grammar we ported) —
and letting the model reason in words before it acts. Every part of that except the
model itself already exists in `arc_adaptor/lexicon/`: `Perception.code_text()`,
`rules_text()` and `predict()` produce the exact same three surfaces.

What was missing is the socket the model plugs into. This module is that socket:
it compiles one decision context from the lexicon (board + rules + what just
happened + what each action is predicted to do), asks a model, and parses the reply
back into engine actions. It is deliberately backend-agnostic:

  backend="openai"  any OpenAI-compatible /chat/completions endpoint (vLLM, SGLang,
                    DashScope, DeepSeek, Ollama's /v1 ...). This is the champion path:
                    point `LINGJING_LLM_URL` at a served Qwen and it runs as-is.
  backend="file"    no network: write `turn_0007.json`, wait for `turn_0007.out`.
                    This is how a human, or an agent with no local weights, can sit in
                    the model's chair — and how this layer is tested without a GPU.
  backend="mock"    replay a fixed action list. Pipeline smoke test only.

Failure is always soft: timeout, garbage, an action the game does not offer, or an
empty reply returns `None`, and the agent falls back to the rule planner. The model
is an upgrade, never a dependency.

Env:
  SELFLEARN_LLM         1|0            enable this layer (default 0)
  SELFLEARN_LLM_BACKEND openai|file|mock   default openai
  SELFLEARN_LLM_CHUNK   int            actions to ask for per call (default 6)
  SELFLEARN_LLM_EVERY   int            min steps between two calls (default 4)
  SELFLEARN_LLM_MAXCHARS int           board-text budget (default 6000)
  LINGJING_LLM_URL / _MODEL / _KEY / _TIMEOUT   endpoint (see agent/llm_policy.py)
  SELFLEARN_LLM_IO      dir            file backend's mailbox (default .llm_io)
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

# Engine action names, and the champion harness's model-facing aliases.
_ALIAS = {
    "UP": "ACTION1", "W": "ACTION1", "NORTH": "ACTION1",
    "DOWN": "ACTION2", "S": "ACTION2", "SOUTH": "ACTION2",
    "LEFT": "ACTION3", "A": "ACTION3", "WEST": "ACTION3",
    "RIGHT": "ACTION4", "D": "ACTION4", "EAST": "ACTION4",
    "SPACE": "ACTION5", "NOOP": "ACTION5", "WAIT": "ACTION5", "PASS": "ACTION5",
    "UNDO": "ACTION7", "BACK": "ACTION7",
}
_ACT_RE = re.compile(r"\bACTION\s*([1-7])\b", re.I)
_ALIAS_RE = re.compile(r"\b(UP|DOWN|LEFT|RIGHT|SPACE|NOOP|WAIT|PASS|UNDO|NORTH|SOUTH|EAST|WEST)\b", re.I)
_JSON_ARR = re.compile(r"\[[^\[\]]*\]")

SYSTEM = (
    "You are playing an ARC-AGI-3 game through an API. You cannot see pixels: you read a "
    "structured view of the board (objects with stable names and grid positions) and a set of "
    "rules learned from your own previous steps in this level.\n"
    "Each step you pick exactly one action. The game is deterministic. Your job is to find the "
    "rule that makes the level advance, then exploit it.\n"
    "How to read the view:\n"
    "- BOARD lists objects as name @ (row,col); scale is the true cell size in screen pixels.\n"
    "- NATIVE is the whole board, one character per cell: use it for distances, walls and "
    "corridors, which the object list alone does not show. (If it is absent, MAP is the same "
    "board sampled every other cell.)\n"
    "- ACTION6 is a click and needs a cell: write it as ACTION6@row,col.\n"
    "- INFERRED is what helper layers concluded: a guessed goal mask, which keys move "
    "what, the distance still to cover, and how much of the board has been walked. It is "
    "derived, not observed -- it is often wrong, and on a board where it disagrees with "
    "NATIVE or CHANGES, believe those instead. Use it as a head start, not as orders.\n"
    "- RULES are `when <action> [if/unless <condition>] then <change>`, with right/wrong counts. "
    "A rule with many rights is real; 0/0 means it is still a guess; wrong counts mean exception.\n"
    "- PREDICTED says what each available action would do under those rules. `nothing` is a "
    "strong signal: repeating it wastes a step.\n"
    "- CHANGES are what actually happened on the last step, including anything no rule explains "
    "yet. An unexplained change is the most informative thing on the board.\n"
    "What to optimise:\n"
    "1. If some action visibly moves/creates/destroys objects toward an obvious goal (a target, "
    "an exit, a matching pattern, a counter), do that.\n"
    "2. Otherwise explore: prefer an action whose predicted next board is one you have not stood "
    "on. Never oscillate between two states.\n"
    "3. If the rules disagree with reality, say so in `why` and probe the unexplained part.\n"
    "Reply with one JSON object and nothing else:\n"
    '{"actions": ["ACTION2", "ACTION2"], "why": "<=15 words"}\n'
    "`actions` is the queue you want executed in order (1 to {chunk}); it will be re-planned "
    "sooner if the board surprises you."
)


def _clip(text: str, limit: int) -> str:
    """Keep the head of a long board dump; the legend and first objects carry the meaning."""
    if limit <= 0 or len(text) <= limit:
        return text
    head = text[: limit]
    cut = head.rfind("\n")
    if cut > limit // 2:
        head = head[:cut]
    return head + f"\n  ... ({len(text) - len(head)} chars elided)"


def _name(action: str) -> str:
    a = (action or "").strip().replace("_", "")
    m = re.match(r"(?:ACTION\s*)?([1-7])\s*(?:@\s*(-?\d+)\s*,\s*(-?\d+))?$", a, re.I)
    if m:
        s = f"ACTION{m.group(1)}"
        if m.group(2) is not None:
            s += f"@{m.group(2)},{m.group(3)}"
        return s
    return _ALIAS.get(a.upper(), "")


def _mini_map(grid, step: int = 2) -> list[str]:
    """The board at 1/`step` resolution, one hex digit per cell.

    The object list says *what* is on the board; this says *where*, which is the half
    the champion's `.native` print gives the model. 64x64 of text would eat the
    context, so sample every other cell. Each block reports its *rarest* colour when it
    contains more than one, so a 1-cell wall or target is not averaged away.
    """
    rows = [list(map(int, r)) for r in grid if r]
    if not rows:
        return []
    h, w = len(rows), len(rows[0])
    step = max(1, int(step))
    hexd = "0123456789abcdef"
    ruler = "    " + "".join(str((c // 10) % 10) if c % 10 < step else " "
                             for c in range(0, w, step))
    out = [ruler]
    for r in range(0, h, step):
        tag = f"{r:3d} " if r % 8 == 0 else "    "
        cells = []
        for c in range(0, w, step):
            cnt: dict = {}
            for i in range(r, min(r + step, h)):
                for j in range(c, min(c + step, w)):
                    v = rows[i][j]
                    cnt[v] = cnt.get(v, 0) + 1
            common = max(cnt, key=cnt.get)
            rare = [v for v in cnt if v != common]
            val = max(rare, key=lambda v: cnt[v]) if rare else common
            cells.append(hexd[val % 16])
        out.append(tag + "".join(cells))
    return out


# One pass over free text, in reading order: actions (with click cells), clicks, aliases, repeats.
_TOK = re.compile(
    r"ACTION\s*([1-7])\s*(?:@\s*(-?\d+)\s*,\s*(-?\d+))?"
    r"|(CLICK|MOUSE)\s*[(@]\s*(?:row\s*=\s*)?(-?\d+)\s*,\s*(?:col\s*=\s*)?(-?\d+)"
    r"|\b(UP|DOWN|LEFT|RIGHT|SPACE|NOOP|WAIT|PASS|UNDO|NORTH|SOUTH|EAST|WEST)\b"
    r"|[x*×]\s*(\d{1,3})",
    re.I,
)


def _scan(text: str) -> list[str]:
    out: list[str] = []
    for m in _TOK.finditer(text):
        if m.group(1):
            a = f"ACTION{m.group(1)}"
            if m.group(2) is not None:
                a += f"@{m.group(2)},{m.group(3)}"
            out.append(a)
        elif m.group(5):
            out.append(f"ACTION6@{m.group(5)},{m.group(6)}")   # CLICK(row,col) is a click
        elif m.group(7):
            out.append(_ALIAS[m.group(7).upper()])
        elif m.group(8) and out:
            out.extend([out[-1]] * (min(int(m.group(8)), 20) - 1))
    return out


def parse_actions(reply: str, allowed: Optional[tuple] = None) -> list[str]:
    """Model text -> engine actions. Tolerant: JSON, prose, aliases, repeats ('A2 x3')."""
    if not reply:
        return []
    text = str(reply)
    out: list[str] = []

    # Preferred: a JSON object with "actions".
    for blob in re.findall(r"\{[^{}]*\}", text):
        try:
            obj = json.loads(blob)
        except Exception:
            continue
        acts = obj.get("actions") if isinstance(obj, dict) else None
        if isinstance(acts, str):
            acts = [acts]
        if isinstance(acts, list) and acts:
            out = [n for n in (_name(str(a)) for a in acts) if n]
            if out:
                break

    # Fallback: a bare JSON array, then free text in reading order.
    if not out:
        for blob in _JSON_ARR.findall(text):
            try:
                arr = json.loads(blob)
            except Exception:
                continue
            if isinstance(arr, list) and arr and all(isinstance(x, str) for x in arr):
                out = [n for n in (_name(x) for x in arr) if n]
                if out:
                    break
    if not out:
        out = _scan(text)

    if allowed:
        allow = {str(a).upper() for a in allowed}
        out = [a for a in out if a.split("@")[0] in allow]
    return out[:24]


def _load_dotenv() -> None:
    """Load the project's `.env` once, without overriding anything already set.

    The agent is launched by the harness, so it never sees the shell that the user
    edited; without this, a key written into `.env` would be silently ignored and the
    layer would look "broken". Same placeholder rules as `scripts/check_llm.py`.
    """
    if os.environ.get("_SELFLEARN_DOTENV"):
        return
    os.environ["_SELFLEARN_DOTENV"] = "1"
    here = Path(__file__).resolve()
    cands = [
        Path(os.environ.get("SELFLEARN_ENV", "")) if os.environ.get("SELFLEARN_ENV") else None,
        Path.cwd() / ".env",
        here.parents[2] / "integrations" / "ARC-AGI-3-Kaggle-Starter" / ".env",
        here.parents[2] / ".env",
    ]
    for p in cands:
        if p is None or not p.is_file():
            continue
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if not k:
                    continue
                cur = os.environ.get(k, "")
                # A placeholder check by length only makes sense for credentials:
                # every switch is short, so `SELFLEARN_LLM=1` on the command line was
                # being read as a placeholder and silently replaced by the .env value.
                # The shell always wins for everything that is not a secret.
                secret = any(t in k.upper() for t in ("KEY", "TOKEN", "SECRET"))
                placeholder = secret and (
                    len(cur) < 8 or cur.lower().startswith("replace_with")
                    or cur.lower() in {"sk-xxxx", "sk-xxx", "your_api_key"})
                if k not in os.environ or placeholder:
                    os.environ[k] = v
        except Exception:  # noqa: BLE001
            continue


def selflearn_llm_enabled() -> bool:
    """总开关：SELFLEARN_LLM=1 → 大模型决策层；=0 → 原策略（纯算法）。

    这是 ``SELFLEARN_LLM`` 的权威实现，``agent/my_agent.py`` 与
    ``agent/self_learner.py`` 使用同一套语义：

    * ``SELFLEARN_LLM=1`` → 挂上 :class:`LLMPlanner`，由大模型出步；
    * ``SELFLEARN_LLM=0`` → 不挂，整条链路退回纯算法（规则前瞻 / 搜索 / 兜底）；
    * 未设置 → 兼容旧开关 ``LINGJING_LLM``（默认 off），老 notebook 行为不变。

    显式设置 ``SELFLEARN_LLM`` 时它说了算，``LINGJING_LLM`` 只当兜底，
    这样 A/B 两轮跑测只改一个变量。
    """
    sl = str(os.environ.get("SELFLEARN_LLM", "") or "").strip().lower()
    if sl in ("1", "on", "true", "yes"):
        return True
    if sl in ("0", "off", "false", "no", "none"):
        return False
    return str(os.environ.get("LINGJING_LLM", "off")).strip().lower() in {
        "1", "on", "true", "yes",
    }


class LLMPlanner:
    """Ask a model for the next few actions, from the same views the champion exposes."""

    def __init__(self, chunk: Optional[int] = None, every: Optional[int] = None,
                 backend: Optional[str] = None, max_chars: Optional[int] = None,
                 io_dir: Optional[str] = None, timeout: Optional[float] = None):
        self.chunk = int(chunk if chunk is not None else os.environ.get("SELFLEARN_LLM_CHUNK", "6"))
        self.every = int(every if every is not None else os.environ.get("SELFLEARN_LLM_EVERY", "4"))
        self.backend = (backend or os.environ.get("SELFLEARN_LLM_BACKEND", "openai")).strip().lower()
        self.max_chars = int(max_chars if max_chars is not None
                             else os.environ.get("SELFLEARN_LLM_MAXCHARS", "6000"))
        self.io_dir = Path(io_dir or os.environ.get("SELFLEARN_LLM_IO", ".llm_io"))
        self.timeout = float(timeout if timeout is not None
                             else os.environ.get("LINGJING_LLM_TIMEOUT", "45") or 45)
        self.calls = 0
        self.hits = 0          # calls that returned a usable queue
        self.rejects = 0
        self._net_down = False  # already complained about a dead endpoint
        self.last_reply = ""
        self.last_why = ""
        self._last_step = -10 ** 9
        self._cache: dict[tuple, list[str]] = {}
        self._seq = 0
        self._mock = [a for a in (os.environ.get("SELFLEARN_LLM_MOCK", "").split(",")) if a.strip()]
        self._guard_logged = False

    # ------------------------------------------------------------------ guard

    def _guard_off(self, where: str) -> bool:
        """最后一道闸：``SELFLEARN_LLM=0`` 时拒绝一切模型交互。

        ``decide()`` 已经有守卫，但那是"调用方自觉"。这里钉在**出网的那一层**：
        任何绕过 ``decide()`` 直接调 ``_post()`` / ``_call_file()`` 的路径，
        在 ``SELFLEARN_LLM=0`` 下也一定不会碰网络、不会写邮箱。

        背景：离线跑测里出现过 ``[learn] llm unreachable (TimeoutError)``——
        那说明有调用方在 ``SELFLEARN_LLM=0`` 下仍然触发了 HTTP，每次白等
        45 秒超时，还会把"原策略臂"的墙钟预算吃光。这个守卫把这条路彻底堵死。
        """
        if selflearn_llm_enabled():
            return False
        if not self._guard_logged:
            self._guard_logged = True
            print(f"[learn] llm blocked by SELFLEARN_LLM=0 at {where}", flush=True)
        return True

    # ------------------------------------------------------------------ context

    def build_context(self, perc, grid, step: int, level: Any = None,
                      allowed: Optional[tuple] = None, recent: Optional[list] = None,
                      outcome: Optional[str] = None, planner=None,
                      inferred: Optional[list] = None) -> tuple[str, str]:
        """(system, user). Everything the model sees is derived from the frame alone."""
        allowed = tuple(allowed or ())
        acts = " ".join(allowed) or "(unknown)"
        lines = [f"LEVEL {level if level is not None else getattr(perc, 'level', '?')}  "
                 f"STEP {step}  AVAILABLE: {acts}"]

        if recent:
            lines.append("RECENT: " + " | ".join(recent[-8:]))
        if outcome:
            lines.append(f"LAST RESULT: {outcome}")

        try:
            board = perc.code_text(grid)
        except Exception as e:  # noqa: BLE001
            board = f"(board unavailable: {type(e).__name__}: {e})"
        lines.append("BOARD (objects: name @ (row,col); scale = cell size in screen pixels):")
        lines.append(_clip(board, self.max_chars))

        try:
            mm = _mini_map(grid)
        except Exception:  # noqa: BLE001
            mm = []
        native = ""
        if str(os.environ.get("LINGJING_LLM_NATIVE", "1")).strip().lower() not in (
                "0", "off", "false", "no"):
            try:
                native = perc.native_ascii(grid)
            except Exception:  # noqa: BLE001
                native = ""
        if native:
            # The champion's `.native` print: one character per world cell. The object
            # list says *what* is on the board; this says *where* at full resolution,
            # which is what multi-step reasoning actually needs.
            lines.append("NATIVE (one char per cell, legend below):")
            lines.append(_clip(native, self.max_chars))
        elif mm:
            lines.append("MAP (every 2nd cell, colour 0-9a-f):")
            lines.extend(mm)

        # Hypotheses sit between the raw board and the learned rules: the model has
        # just read what is there, and this says what the helper layers concluded
        # about it. Read-only on purpose -- see lexicon/view.py.
        if inferred:
            lines.extend(inferred)

        try:
            rules = perc.rules_text()
        except Exception:  # noqa: BLE001
            rules = "RULES: (none)"
        lines.append(_clip(rules, max(800, self.max_chars // 3)))

        # What each action would do, per the learned rules — saves the model the inference.
        pred: list[str] = []
        try:
            for a in allowed:
                effs = perc.predict(grid, a)
                if not effs:
                    pred.append(f"  {a} -> nothing (per rules)")
                else:
                    bits = []
                    for _r, i, eff in effs[:3]:
                        kind = eff[0]
                        if kind == "move":
                            bits.append(f"obj{i} moves ({eff[1][0]:+d},{eff[1][1]:+d})")
                        elif kind == "vanish":
                            bits.append(f"obj{i} vanishes")
                        elif kind == "appear":
                            bits.append(f"obj{i} appears")
                        else:
                            bits.append(f"obj{i} {kind}")
                    pred.append(f"  {a} -> " + "; ".join(bits))
        except Exception:  # noqa: BLE001
            pass
        if pred:
            lines.append("PREDICTED (from rules; empty rules = all guesses):")
            lines.extend(pred)

        if planner is not None:
            try:
                seen = len(getattr(planner, "visits", {}) or {})
                lines.append(f"EXPLORED: {seen} distinct boards this level")
            except Exception:  # noqa: BLE001
                pass

        lines.append(f"Reply with the next up to {self.chunk} actions.")
        return SYSTEM.replace("{chunk}", str(self.chunk)), "\n".join(lines)

    # ------------------------------------------------------------------ backends

    def _endpoint(self) -> dict:
        """OpenAI-compatible endpoint.

        Accepts either naming so an existing `.env` keeps working:
          LINGJING_LLM_URL / LINGJING_LLM_MODEL / LINGJING_LLM_KEY   (this layer)
          OPENAI_BASE_URL / AETHER_LLM_MODEL / OPENAI_API_KEY        (llm_policy.py)

        No API key is required for the champion's setup: it serves Qwen3.8-Flash-Next
        through a local SGLang at 127.0.0.1:8001/v1, and vLLM/SGLang do not check the
        bearer token.  The key here is a placeholder so the same code also works
        against a hosted endpoint.
        """
        _load_dotenv()
        base = (os.environ.get("LINGJING_LLM_URL")
                or os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("LOCAL_ANALYZER_BASE_URL")
                or "http://127.0.0.1:1234/v1").rstrip("/")
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")]
        model = (os.environ.get("LINGJING_LLM_MODEL")
                 or os.environ.get("AETHER_LLM_MODEL")
                 or os.environ.get("LOCAL_ANALYZER_MODEL_ID")
                 or "vrfai/Qwen3.6-27B-FP8")
        key = (os.environ.get("LINGJING_LLM_KEY")
               or os.environ.get("OPENAI_API_KEY")
               or "EMPTY")
        return {"url": base + "/chat/completions", "model": model, "key": key}

    def _call_openai(self, system: str, user: str) -> str:
        cfg = self._endpoint()
        # Qwen3 in thinking mode: the champion runs `LOCAL_ANALYZER_ENABLE_THINKING=true`
        # and spends 12K output tokens per turn.  Thinking is what turns "list the
        # objects" into "simulate two moves ahead", so it is worth the latency -- but it
        # must be asked for explicitly, and the token cap has to grow or the reply gets
        # truncated before the JSON ever appears.
        think = str(os.environ.get("SELFLEARN_LLM_THINK", "0")).strip().lower() in (
            "1", "true", "yes", "on")
        default_tok = "1536" if think else "220"
        body = {
            "model": cfg["model"],
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": float(os.environ.get(
                "SELFLEARN_LLM_TEMP", "0.6" if think else "0.1")),
            "max_tokens": int(os.environ.get("SELFLEARN_LLM_MAXTOK", default_tok)),
            "stop": ["\n\n\n"],
        }
        if think:
            body["chat_template_kwargs"] = {"enable_thinking": True}
            body["top_p"] = 0.95
            body.pop("stop", None)  # stop strings fire inside the reasoning trace
        else:
            # Qwen3's chat template thinks BY DEFAULT.  Sending no kwargs is not
            # "no thinking": the model opens with <think> and burns the whole
            # max_tokens budget there, so `content` comes back empty and every
            # call looks like a parse failure.  Turn it off explicitly.
            body["chat_template_kwargs"] = {"enable_thinking": False}
        text = self._post(cfg, body)
        if not text and "chat_template_kwargs" in body and not think:
            # Some endpoints reject the field outright (HTTP 400); retry plain.
            body.pop("chat_template_kwargs")
            text = self._post(cfg, body)
        return text

    def _post(self, cfg: dict, body: dict) -> str:
        if self._guard_off("_post"):
            return ""
        req = urllib.request.Request(
            cfg["url"], data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {cfg['key']}"},
            method="POST")
        # A system-wide HTTP_PROXY turns requests to 127.0.0.1 into 502 Bad Gateway.
        # Local endpoints must bypass it, or every offline run looks "unreachable".
        if any(h in cfg["url"] for h in ("127.0.0.1", "localhost", "0.0.0.0", "[::1]")):
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
        else:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
        try:
            msg = data["choices"][0]["message"] or {}
        except Exception:
            return ""
        content = (msg.get("content") or "").strip()
        if not content and (msg.get("reasoning_content") or "").strip():
            # Thinking ran out of tokens before the answer started: raise MAXTOK.
            print("[learn] llm reply truncated in reasoning -> raise SELFLEARN_LLM_MAXTOK",
                  flush=True)
        return content

    def _call_file(self, system: str, user: str) -> str:
        """Mailbox protocol: we write the request, someone (human / outer model) writes .out."""
        if self._guard_off("_call_file"):
            return ""
        self.io_dir.mkdir(parents=True, exist_ok=True)
        self._seq += 1
        stem = self.io_dir / f"turn_{self._seq:04d}"
        req = {"system": system, "user": user, "chunk": self.chunk,
               "written": time.strftime("%Y-%m-%d %H:%M:%S")}
        tmp = stem.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(req, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, stem.with_suffix(".json"))
        out = stem.with_suffix(".out")
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if out.exists():
                try:
                    text = out.read_text(encoding="utf-8").strip()
                except Exception:
                    text = ""
                if text:
                    return text
            time.sleep(0.2)
        return ""

    def _call_mock(self, system: str, user: str) -> str:
        return json.dumps({"actions": self._mock, "why": "mock"})

    # ------------------------------------------------------------------ decision

    def ready(self, step: int) -> bool:
        """Rate limit: do not spend a call on every single step."""
        if not selflearn_llm_enabled():
            return False
        return step - self._last_step >= self.every

    def decide(self, perc, grid, step: int, level: Any = None,
               allowed: Optional[tuple] = None, recent: Optional[list] = None,
               outcome: Optional[str] = None, planner=None,
               fp: Any = None, inferred: Optional[list] = None) -> Optional[list[str]]:
        """Next action queue, or None (caller falls back to the rule planner)."""
        # Fail-closed: SELFLEARN_LLM=0 means "no model in the chair", even if a
        # caller built this object anyway. The rule planner keeps the game moving.
        if not selflearn_llm_enabled():
            return None
        if not self.ready(step):
            return None
        key = (level, fp)
        if key in self._cache:
            self.hits += 1
            self._last_step = step
            return list(self._cache[key])

        system, user = self.build_context(perc, grid, step, level, allowed, recent,
                                          outcome, planner, inferred=inferred)
        self.calls += 1
        self._last_step = step
        try:
            if self.backend == "file":
                reply = self._call_file(system, user)
            elif self.backend == "mock":
                reply = self._call_mock(system, user)
            else:
                reply = self._call_openai(system, user)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            self.rejects += 1
            # Print once per outage: a dead endpoint would otherwise print on every
            # decision and bury the useful lines.  Recovery prints again.
            if not self._net_down:
                self._net_down = True
                print(f"[learn] llm unreachable ({type(e).__name__}: {e}) "
                      f"-> rule planner (silencing further calls)", flush=True)
            return None
        except Exception as e:  # noqa: BLE001
            self.rejects += 1
            print(f"[learn] llm err {type(e).__name__}: {e} -> rule planner", flush=True)
            return None
        if self._net_down:
            self._net_down = False
            print("[learn] llm endpoint back", flush=True)

        self.last_reply = reply[:400]
        acts = parse_actions(reply, allowed)
        if not acts:
            self.rejects += 1
            print(f"[learn] llm no usable action in {reply[:160]!r}", flush=True)
            return None
        try:
            m = re.search(r'"why"\s*:\s*"([^"]{0,120})"', reply)
            self.last_why = m.group(1) if m else ""
        except Exception:
            self.last_why = ""
        acts = acts[: self.chunk]
        self.hits += 1
        self._cache[key] = list(acts)
        return acts

    def stats(self) -> str:
        return f"llm calls={self.calls} hits={self.hits} rejects={self.rejects}"


# Load .env at import time: `self_learner` reads SELFLEARN_LLM before it builds the
# planner, so the switch has to be in os.environ before that line runs.
_load_dotenv()
