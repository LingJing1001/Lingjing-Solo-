"""The loop.

One turn is: hand the model the board, let it write code, run that code, and if
the code spent an action, end the turn and hand back the new board. Everything
else is bookkeeping around that.

Three things are load-bearing and easy to get wrong:

1. **A turn ends when an action executes**, not when the model stops talking.
   Otherwise the model keeps reading a board it has already invalidated.
2. **History is compacted by the model itself.** When the context budget is
   reached we ask it to summarize, then drop everything and keep the summary
   (the champion's `SUMMARY_REQUEST_PROMPT`). A summarizer we write ourselves
   would drop exactly what it does not understand.
3. **Compaction must never sever a tool call from its result.** A message list
   with a `tool` role whose `tool_call_id` has been dropped is a 400 from every
   provider, and it will look like the model is broken.
"""
from __future__ import annotations

import json
import time

from . import prompts
from .chat import PYTHON_TOOL
from .sandbox import Sandbox


class DuckAgent:
    def __init__(self, env, chat, *, max_steps: int = 200, max_turns: int = 60,
                 summary_tokens: int = 60_000, level_transfer: bool = True,
                 verbose: bool = True, max_inner: int = 6):
        self.env = env
        self.chat = chat
        self.max_steps = max_steps
        self.max_turns = max_turns
        self.summary_tokens = summary_tokens
        self.level_transfer = level_transfer
        self.verbose = verbose
        self.max_inner = max_inner

        self.sandbox = Sandbox(env)
        self.system = prompts.build_system(level_transfer=level_transfer)
        self.messages: list[dict] = [{"role": "system", "content": self.system}]
        self.turns = 0
        self.summaries = 0
        self.level = 0
        self.best_level = 0
        self.started = time.time()

    # ------------------------------------------------------------------ helpers

    def _say(self, text: str) -> None:
        if self.verbose:
            print(text, flush=True)

    def _opener(self, *, need_action: bool = True) -> str:
        f = self.sandbox.current
        return prompts.build_opener(
            ascii_board=f.ascii if f else "",
            shape=f.shape if f else (0, 0),
            valid=self.sandbox.valid_actions,
            level=self.level, step=self.sandbox.steps,
            need_action=need_action)

    def _append_tool_result(self, call_id: str, result: dict) -> None:
        self.messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(result, ensure_ascii=False, default=str),
        })

    @staticmethod
    def _arguments(call: dict) -> dict:
        raw = (call.get("function") or {}).get("arguments")
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw or "{}")
        except (TypeError, ValueError):
            return {}

    # ------------------------------------------------------------------ summary

    def _compact(self) -> None:
        """Ask the model for a summary, then drop everything before it."""
        self.messages.append({"role": "user", "content": prompts.SUMMARY_REQUEST_PROMPT})
        try:
            reply = self.chat(self.messages, None)
        except Exception as exc:  # noqa: BLE001 - compaction is best effort
            self._say(f"[duck] summary failed ({type(exc).__name__}: {exc}); "
                      f"dropping the oldest turns instead")
            self._drop_oldest()
            return
        text = (reply.get("content") or "").strip()
        if not text:
            self._drop_oldest()
            return
        self.summaries += 1
        self._say(f"[duck] compacted after {self.chat.tokens()} tokens "
                  f"(summary {len(text)} chars)")
        self.messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content":
                "Your own summary of the play so far. Everything before it has been "
                "dropped; this is now the only record.\n\n" + text},
            {"role": "assistant", "content": "Understood. Continuing from that state."},
        ]

    def _drop_oldest(self) -> None:
        """Fallback compaction that preserves call/result pairing.

        Keeps the system message, the newest assistant turn, and everything after
        it -- so no `tool` message is left without the call that produced it.
        """
        keep_from = len(self.messages) - 1
        while keep_from > 1 and self.messages[keep_from].get("role") != "assistant":
            keep_from -= 1
        tail = self.messages[keep_from:]
        self.messages = [self.messages[0]] + tail

    # ------------------------------------------------------------------ loop

    def play(self) -> dict:
        first = self.env.reset()
        self.sandbox.adopt(first.frame, level=first.level)
        self.level = first.level
        self.best_level = first.level

        while self.sandbox.steps < self.max_steps and self.turns < self.max_turns:
            self.turns += 1
            self.messages.append({"role": "user", "content": self._opener()})
            acted = self._inner()
            if not acted:
                self._say("[duck] no action executed this turn; stopping "
                          f"at step {self.sandbox.steps}, level {self.level}")
                break
            if self.chat.tokens() >= self.summary_tokens:
                self._compact()
            if self._transitioned():
                return self._done("level_changed")

        return self._done("budget")

    def _inner(self) -> bool:
        """Let the model read/compute until it spends an action. True if it did."""
        nudges = 0
        for _ in range(self.max_inner):
            try:
                reply = self.chat(self.messages, [PYTHON_TOOL])
            except Exception as exc:  # noqa: BLE001
                self._say(f"[duck] chat failed: {type(exc).__name__}: {exc}")
                return False

            calls = reply.get("tool_calls") or []
            assistant = {"role": "assistant", "content": reply.get("content") or ""}
            if calls:
                assistant["tool_calls"] = calls
            self.messages.append(assistant)

            if not calls:
                text = (reply.get("content") or "").strip()
                nudges += 1
                self._say(f"[duck] turn {self.turns}: no tool call "
                          f"({len(text)} chars of prose, {reply.get('_latency') or 0}s)")
                if nudges >= 2 or not text:
                    return False
                self.messages.append({
                    "role": "user",
                    "content": ("You wrote prose but did not act. Run the code that settles "
                                "the question and call `action(...)` -- a turn that only "
                                "reads the board changes nothing."),
                })
                continue

            acted = False
            for i, call in enumerate(calls):
                fn = (call.get("function") or {}).get("name") or "python"
                if fn != "python":
                    self._append_tool_result(call.get("id") or f"call_{i}",
                                             {"ok": False,
                                              "error": f"unknown tool {fn!r}; only 'python'"})
                    continue
                code = self._arguments(call).get("code") or ""
                result = self.sandbox.run(code)
                executed = int(result.get("executed_now") or 0)
                self._trace(result, executed)
                self._append_tool_result(call.get("id") or f"call_{i}", result)
                if executed:
                    acted = True
            if acted:
                return True
        return False

    def _trace(self, result: dict, executed: int) -> None:
        if not self.verbose:
            return
        stdout = (result.get("stdout") or "").strip().replace("\n", " ⏎ ")
        if len(stdout) > 160:
            stdout = stdout[:160] + "…"
        tag = "ACTION" if executed else "read"
        extra = ""
        if not result.get("ok"):
            extra = f" !! {result.get('stop_reason')}: {result.get('error')}"
        self._say(f"[duck] t{self.turns} {tag} lv={self.level} "
                  f"steps={result.get('steps_used')} {stdout}{extra}")

    def _transitioned(self) -> bool:
        """New level, death, or a win: the board we would plan against is gone."""
        if self.sandbox.level != self.level:
            if self.sandbox.level > self.level:
                self.best_level = self.sandbox.level
                self._say(f"[duck] LEVEL UP -> {self.sandbox.level} "
                          f"(steps={self.sandbox.steps})")
                self.level = self.sandbox.level
                self.sandbox.reset_level_scope()
                self.messages.append({"role": "user",
                                      "content": prompts.LEVEL_START_USER_PROMPT})
            else:
                self.level = self.sandbox.level
            return False
        if self.sandbox._terminal:
            reason = self.sandbox._terminal
            self._say(f"[duck] {reason} at level {self.level}, step {self.sandbox.steps}")
            if reason in {"game_over", "level_completed"}:
                reset = self.env.reset()
                self.sandbox.adopt(reset.frame, level=reset.level)
                # A death returns the board to the level start, so what was
                # learned about the mechanics still holds -- the champion's
                # guards say so explicitly ("deaths and resets deliberately do
                # NOT clear the store"). Only the terminal latch is cleared,
                # otherwise every later action would be refused.
                self.sandbox.clear_terminal()
                self.messages.append({
                    "role": "user",
                    "content": (f"The previous attempt ended ({reason}). The board below is "
                                f"the level start again. What you established about the "
                                f"mechanics still holds; what you were doing when it ended "
                                f"does not. Do not repeat the fatal route without a change "
                                f"of reasoning.\n\n" + self._opener()),
                })
                return False
            return True
        return False

    def _done(self, why: str) -> dict:
        return {
            "game": self.env.game_id,
            "level": self.level,
            "best_level": self.best_level,
            "steps": self.sandbox.steps,
            "turns": self.turns,
            "summaries": self.summaries,
            "calls": getattr(self.chat, "calls", 0),
            "tokens": getattr(self.chat, "tokens", lambda: 0)(),
            "refusals": dict(self.sandbox.refusals),
            "retained": sorted(self.sandbox.retained),
            "seconds": round(time.time() - self.started, 1),
            "stopped": why,
        }
