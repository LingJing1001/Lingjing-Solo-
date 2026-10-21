# -*- coding: utf-8 -*-
"""Duck, resumable one action at a time.

Why this module exists
----------------------
The champion runs its own action gateway: the model's `action(...)` call executes
immediately inside the sandbox. We cannot copy that on this harness, and the
reason is a scoring rule, not taste:

    arc_agi/wrapper.py:186-195  ->  scorecard.update_scorecard() runs only for
    actions that pass through `env.step()`.  `game.perform_action()` does NOT
    count.

So an action executed straight on the game object is an *uncounted* action: the
level would look cheaper than it was, and the score (`(baseline/actions)**2`)
would be inflated. That is not a shortcut, it is a wrong number.

The harness calls the agent once per tick and steps the engine with whatever the
agent returns. Therefore duck has to hand one action out per tick:

    driver.poll()  -> ("ACTION2", {...})      agent returns it, harness steps
    driver.feed(step_result)                  driver stores it; the next poll()
                                              delivers it and runs on to the next action

`feed` stores rather than sends: the generator is already parked on the next
`yield` by then, so sending from `feed` would make the next `poll`'s `send(None)`
resume it with `None`. That failure mode is silent -- the action's result simply
disappears -- so the handshake is `poll` -> `feed` -> `poll`, always.

That is the champion's `ToolAgent` yield/resume, which docs/duck-port.md lists
as unimplemented. This file implements just that, keeping `sandbox.py` intact so
its own tests keep meaning what they meant.

Nothing here decides anything: the model still writes the code. This layer only
turns "execute now" into "yield, then continue".
"""
from __future__ import annotations

import json
import time

from . import prompts
from .chat import PYTHON_TOOL
from .env import wrap_raw
from .memory import SolvedLevelMemory
from .sandbox import (Sandbox, MODEL_TO_ENGINE, action_signature,
                      normalise_actions)


class ActionPending(BaseException):
    """Raised from inside a snippet when the model spends an action.

    Inherits BaseException on purpose, exactly like the refusal types in
    `sandbox.py`: model code habitually wraps `action(...)` in
    `except Exception`, and a swallowed turn boundary would let the snippet keep
    running against a board that is about to change.
    """

    def __init__(self, name: str, spec: dict, rest: list[dict]):
        super().__init__(name)
        self.name = name
        self.spec = spec
        self.rest = rest


class DeferredEnv:
    """Engine-shaped, but `step()` only announces the action.

    Reads (`grid_of`, `available`) are real: the sandbox needs them to build
    views. Writes are deferred to the harness, which is the only thing allowed
    to spend a scored action.
    """

    def __init__(self, game_id: str):
        self.game_id = game_id
        self.pending: ActionPending | None = None

    @staticmethod
    def grid_of(frame) -> list[list[int]]:
        import numpy as np

        layers = getattr(frame, "frame", None)
        if not layers:
            return []
        return np.asarray(layers[-1]).astype(int).tolist()

    def available(self, frame) -> list[str]:
        from .env import ENGINE_TO_MODEL

        raw = getattr(frame, "available_actions", None) or []
        names = []
        for a in raw:
            if isinstance(a, bool):
                continue
            key = f"ACTION{int(a)}" if isinstance(a, int) else (
                getattr(a, "name", None) or str(a))
            if key.isdigit():
                key = f"ACTION{key}"
            names.append(ENGINE_TO_MODEL.get(key, key))
        return names

    def step(self, name: str, spec: dict | None = None):
        raise ActionPending(name, dict(spec or {}), [])

    def reset(self):
        raise ActionPending("RESET", {}, [])

    def close(self) -> None:  # pragma: no cover - nothing owns a socket here
        pass


class DeferredSandbox(Sandbox):
    """Sandbox whose `action()` hands the action out instead of spending it."""

    def _do_action(self, specs: list[dict]) -> dict:
        if not specs:
            return {"executed": 0, "note": "no actions given"}
        if self._terminal:
            self.refusals["terminal"] += 1
            from .sandbox import TerminalStateActionError, _TERMINAL_DETAIL
            raise TerminalStateActionError(
                _TERMINAL_DETAIL.get(self._terminal, self._terminal))

        spec = specs[0]
        name = MODEL_TO_ENGINE.get(spec["action"], spec["action"])
        sig = action_signature(spec)
        allowed = {MODEL_TO_ENGINE.get(a, a) for a in self.valid_actions} or None
        if name != "RESET" and allowed and name not in allowed:
            from .sandbox import ActionError
            raise ActionError(
                f"{spec['action']} is not available; valid_actions={self.valid_actions}")

        start = self._here()
        if (start, sig) in self._executed_in_snippet:
            from .sandbox import RepeatedActionInStateError
            self.refusals["repeated"] += 1
            raise RepeatedActionInStateError(
                f"{spec['action']} was already executed from this exact board inside "
                f"this snippet. If you meant to repeat it, write a new snippet.")

        known = self._noop.get(start, set())
        if sig in known:
            key = (start, sig)
            if self._armed_noop == key:
                self._armed_noop = None
            else:
                from .sandbox import KnownNoOpActionError
                self._armed_noop = key
                self.refusals["noop"] += 1
                raise KnownNoOpActionError(
                    f"{spec['action']} is known to change nothing from this exact board. "
                    f"Nothing ran. If you have a specific reason, issue it again.")

        # The fatal branch of the same guard: a no-op wastes one action, this
        # wastes the attempt. Kept here as well as in `Sandbox` because this
        # class re-implements `_do_action` to hand actions out instead of
        # spending them -- but the ledger is written in `deliver`, which is
        # where the real outcome finally arrives.
        fatal = self._deadly.get(start, set())
        if sig in fatal:
            key = (start, sig)
            if self._armed_deadly == key:
                self._armed_deadly = None
            else:
                from .sandbox import KnownDeadlyActionError
                self._armed_deadly = key
                self.refusals["deadly"] += 1
                raise KnownDeadlyActionError(
                    f"{spec['action']} ended the run from this exact board last time. "
                    f"Nothing ran. If you have a specific reason to expect a different "
                    f"outcome, issue it again.")

        pending = ActionPending(name, spec, specs[1:])
        self.env.pending = pending
        raise pending

    # ------------------------------------------------------------------ resume

    def deliver(self, step_result, name: str, spec: dict, start: str,
                prior_specs: list[dict] | None = None) -> None:
        """Bookkeeping for an action the harness just spent."""
        before_grid = self.current._grid if self.current is not None else None
        self.adopt(step_result.frame, level=step_result.level)
        self.previous_grid = before_grid

        changed = self._gameplay_changed(before_grid, self.current._grid)
        entry = {"action": spec.get("action", name)}
        self.transitions.append({
            "action": entry["action"],
            "before_frame": before_grid,
            "after_frame": self.current._grid,
            "result": {**step_result.flags, "gameplay_changed": changed},
        })
        self.history.append({"action": entry["action"], "result": step_result.flags})
        self.steps += 1

        for terminal in ("game_over", "run_complete", "level_completed", "done"):
            if step_result.flags.get(terminal):
                self._terminal = "run_complete" if terminal == "done" else terminal
                break

        sig = action_signature(spec)
        if changed is False and self._terminal != "game_over":
            # A fatal action often leaves the board untouched; filing it as a no-op
            # would make the refusal say "this does nothing" instead of "this kills
            # you", which is a strictly worse hint to give the model.
            self._noop.setdefault(start, set()).add(sig)
        elif changed is True:
            self._noop.get(start, set()).discard(sig)
        self._executed_in_snippet.add((start, sig))

        if self._terminal == "game_over":
            self._deadly.setdefault(start, set()).add(sig)
        elif sig in self._deadly.get(start, set()):
            self._deadly[start].discard(sig)

        self.last_result = {
            "executed": 1 + len(prior_specs or []),
            "actions": [entry],
            "result": {**step_result.flags, "gameplay_changed": changed},
            "state": step_result.state,
            "level": self.level,
            "steps": self.steps,
        }

    def run(self, code: str) -> dict:
        """Run a snippet; if it spends an action, report it instead of spending."""
        self._executed_in_snippet.clear()
        steps_before = self.steps
        result = super().run(code)
        pending = self.env.pending if isinstance(self.env, DeferredEnv) else None
        if pending is not None:
            self.env.pending = None
            result["pending"] = pending
            result["executed_now"] = 0     # nothing was actually spent yet
            result["ok"] = True            # a turn boundary is not an error
            # ActionPending is a BaseException, so `Sandbox.run` caught it as a
            # snippet failure and attached a traceback. That traceback is the
            # *control-flow* unwinding, not an error the model can use -- leaving
            # it in the tool result reads as "your code crashed".
            for key in ("error", "stop_reason", "traceback"):
                result.pop(key, None)
        return result


class DuckDriver:
    """Turn loop as a resumable generator. No model, no decisions: plumbing only."""

    def __init__(self, chat, *, game_id: str, frame, level: int = 0,
                 max_steps: int = 200, max_turns: int = 40, max_inner: int = 6,
                 summary_tokens: int = 60_000, level_transfer: bool = True,
                 level_memory: bool = True, pin_policy: str = "drain",
                 env=None, verbose: bool = True):
        self.chat = chat
        self.game_id = game_id
        self.max_steps = max_steps
        self.max_turns = max_turns
        self.max_inner = max_inner
        self.summary_tokens = summary_tokens
        self.verbose = verbose
        if pin_policy not in {"drain", "level"}:
            raise ValueError("pin_policy must be 'drain' (cache-safe) or 'level'")
        self.pin_policy = pin_policy

        self.env = env if env is not None else DeferredEnv(game_id)
        self.sandbox = DeferredSandbox(self.env)
        self.sandbox.adopt(frame, level=level)
        self.memory = SolvedLevelMemory(enabled=level_memory)
        self._system_base = prompts.build_system(level_transfer=level_transfer)
        self.system = self._system_base
        self.messages: list[dict] = [{"role": "system", "content": self.system}]
        self.level = level
        self.best_level = level
        self._prev_levels_completed = level
        self._level_actions: list[str] = []
        self._last_reasoning = ""
        self._compacted = False
        self.turns = 0
        self.summaries = 0
        self.started = time.time()
        self.done = False
        self.last_pending: ActionPending | None = None
        self.last_start_hash = ""
        self._pending_feed = None
        # Set by a harness that owns more than one game (scheduler.py). Called when
        # an attempt ends, it must return the frame of the *new* attempt; without
        # it the sandbox keeps describing the board the run died on.
        self.reset_hook = None
        self._gen = self._loop()

    # ------------------------------------------------------------------ plumbing

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
        result = {k: v for k, v in result.items() if k != "pending"}
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

    # ------------------------------------------------------------------ public API

    def poll(self) -> tuple[str, dict] | None:
        """Next action the harness should spend, or None when duck is finished.

        The handshake is `poll` -> `feed` -> `poll`, and the value handed back by
        `feed` is *delivered by the next poll*. Sending from `feed` instead looks
        equivalent and is not: the generator would already be parked at the next
        `yield` when the harness calls `poll()` again, so `poll`'s own `send(None)`
        would resume it with `None` and the action's result would be lost.
        """
        if self.done:
            return None
        value, self._pending_feed = self._pending_feed, None
        try:
            nxt = self._gen.send(value)
        except StopIteration:
            self.done = True
            return None
        name, spec = nxt
        return name, spec

    def feed(self, step_result) -> None:
        """Hand the real result of the action poll() returned.

        Accepts what the harness actually has: the engine's `FrameDataRaw`. A
        `StepResult` is accepted too (the self-test and any harness that wraps it
        already), and is passed through untouched.
        """
        if self.done:
            return
        if self._pending_feed is not None:
            raise RuntimeError("feed() called twice without an intervening poll()")
        if not hasattr(step_result, "flags"):
            step_result = wrap_raw(step_result,
                                   previous_level=self._prev_levels_completed)
        self._prev_levels_completed = int(step_result.level)
        self._pending_feed = step_result

    # ------------------------------------------------------------------ loop

    def _note_reply(self, reply: dict) -> None:
        """Keep the tail of this turn's text, and harvest a `Rule for level N:` line."""
        parts = []
        for key in ("content", "reasoning_content", "reasoning"):
            value = reply.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value)
        if not parts:
            return
        text = "\n".join(parts)
        self._last_reasoning = text[-self.memory.reasoning_chars:]
        rule = self.memory.capture(text)
        if rule:
            self._say(f"[duck-drv] rule recorded for level {self.memory.levels[-1].level}: "
                      f"{rule[:80]}")
            self._refresh_system()

    def _refresh_system(self) -> None:
        """Pin the block, if this is a moment where re-reading costs nothing."""
        if self.pin_policy == "drain" and not self._compacted:
            return
        if not self.memory.changed():
            return
        self.system = self.memory.rewrite(self._system_base)
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = self.system
        self._say(f"[duck-drv] solved-level block pinned "
                  f"({len(self.memory)} level(s), {len(self.memory.block())} chars)")

    def _inner(self):
        """Model reads/computes until it spends an action (yields it)."""
        nudges = 0
        for _ in range(self.max_inner):
            try:
                reply = self.chat(self.messages, [PYTHON_TOOL])
            except Exception as exc:  # noqa: BLE001
                self._say(f"[duck-drv] chat failed: {type(exc).__name__}: {exc}")
                return False

            self._note_reply(reply)
            calls = reply.get("tool_calls") or []
            assistant = {"role": "assistant", "content": reply.get("content") or ""}
            if calls:
                assistant["tool_calls"] = calls
            self.messages.append(assistant)

            if not calls:
                text = (reply.get("content") or "").strip()
                nudges += 1
                self._say(f"[duck-drv] t{self.turns}: no tool call ({len(text)} chars)")
                if nudges >= 2 or not text:
                    return False
                self.messages.append({
                    "role": "user",
                    "content": ("You wrote prose but did not act. Run the code that settles "
                                "the question and call `action(...)` -- a turn that only "
                                "reads the board changes nothing."),
                })
                continue

            for i, call in enumerate(calls):
                fn = (call.get("function") or {}).get("name") or "python"
                if fn != "python":
                    self._append_tool_result(call.get("id") or f"call_{i}",
                                             {"ok": False,
                                              "error": f"unknown tool {fn!r}; only 'python'"})
                    continue
                code = self._arguments(call).get("code") or ""
                result = self.sandbox.run(code)
                pending = result.get("pending")
                if pending is None:
                    self._append_tool_result(call.get("id") or f"call_{i}", result)
                    continue

                # one action per tick; a batch is drained tick by tick
                queue = [(pending.name, pending.spec)] + [
                    (MODEL_TO_ENGINE.get(s["action"], s["action"]), s)
                    for s in pending.rest]
                for idx, (name, spec) in enumerate(queue):
                    start = self.sandbox._here()
                    step_result = yield (name, spec)
                    self.sandbox.deliver(step_result, name, spec, start)
                    # A RESET starts a new attempt: only the winning attempt's
                    # actions are worth remembering (same rule as upstream).
                    if name == "RESET":
                        self._level_actions = []
                    else:
                        self._level_actions.append(str(spec.get("action") or name))
                    if self.sandbox._terminal:
                        break
                self._append_tool_result(call.get("id") or f"call_{i}", {
                    **{k: v for k, v in result.items() if k != "pending"},
                    "executed_now": 1,
                    "steps_used": self.sandbox.steps,
                    "last_action_call_result": self.sandbox.last_result,
                    "valid_actions": self.sandbox.valid_actions,
                })
                return True
        return False

    def _compact(self) -> None:
        self.messages.append({"role": "user", "content": prompts.SUMMARY_REQUEST_PROMPT})
        try:
            reply = self.chat(self.messages, None)
            text = (reply.get("content") or "").strip()
        except Exception:  # noqa: BLE001
            text = ""
        # The conversation is being rebuilt: the server must re-read it anyway,
        # so this is the one moment where changing the system prompt is free.
        self._compacted = True
        self._refresh_system()
        if not text:
            keep_from = len(self.messages) - 1
            while keep_from > 1 and self.messages[keep_from].get("role") != "assistant":
                keep_from -= 1
            self.messages = [self.messages[0]] + self.messages[keep_from:]
            return
        self.summaries += 1
        self.messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content":
                "Your own summary of the play so far. Everything before it has been "
                "dropped; this is now the only record.\n\n" + text},
            {"role": "assistant", "content": "Understood. Continuing from that state."},
        ]

    def _loop(self):
        while self.sandbox.steps < self.max_steps and self.turns < self.max_turns:
            self.turns += 1
            opener = self._opener()
            ask = self.memory.begin_turn()
            if ask:
                opener += "\n\n" + ask
            self.messages.append({"role": "user", "content": opener})
            acted = yield from self._inner()
            if not acted:
                self._say("[duck-drv] no action executed; stopping at "
                          f"step {self.sandbox.steps}, level {self.level}")
                return
            if getattr(self.chat, "tokens", None) and self.chat.tokens() >= self.summary_tokens:
                self._compact()

            # Level-up is checked first: the engine reports both `level_completed`
            # and a higher level, and only this branch is true about what happened.
            if self.sandbox.level > self.level:
                cleared = self.level
                if self.memory.enabled:
                    self.memory.record(cleared, list(self._level_actions),
                                       self._last_reasoning)
                    self._refresh_system()
                self._level_actions = []
                self.best_level = self.sandbox.level
                self.level = self.sandbox.level
                self.sandbox.reset_level_scope()
                self._say(f"[duck-drv] LEVEL UP -> {self.level} "
                          f"(steps={self.sandbox.steps})")
                self.messages.append({"role": "user",
                                      "content": prompts.LEVEL_START_USER_PROMPT})
                continue

            if self.sandbox._terminal:
                reason = self.sandbox._terminal
                self._say(f"[duck-drv] {reason} at level {self.level}, step {self.sandbox.steps}")
                if reason in {"game_over", "level_completed"}:
                    # the harness will have reset the level; hand the board back
                    self.sandbox.clear_terminal()
                    if self.reset_hook is not None:
                        # The engine has already restarted the attempt, so the board
                        # this generator is about to describe in its next opener is
                        # the one that no longer exists. Ask the harness for the new
                        # one instead of guessing: a stale board here makes the model
                        # reason about a dead position for the rest of the run.
                        try:
                            fresh = self.reset_hook()
                        except Exception as exc:  # noqa: BLE001
                            self._say(f"[duck-drv] reset_hook failed: {type(exc).__name__}: {exc}")
                            fresh = None
                        if fresh is not None:
                            fresh_level = int(getattr(fresh, "level", self.level))
                            self.sandbox.adopt(fresh, level=fresh_level)
                            self._prev_levels_completed = fresh_level
                    self.messages.append({
                        "role": "user",
                        "content": (f"The previous attempt ended ({reason}). What you "
                                    "established about the mechanics still holds; what you "
                                    "were doing when it ended does not."),
                    })
                    continue
                return

    def stats(self) -> dict:
        return {
            "game": self.game_id,
            "level": self.level,
            "best_level": self.best_level,
            "steps": self.sandbox.steps,
            "turns": self.turns,
            "summaries": self.summaries,
            "calls": getattr(self.chat, "calls", 0),
            "tokens": getattr(self.chat, "tokens", lambda: 0)(),
            "refusals": dict(self.sandbox.refusals),
            "solved_levels": self.memory.stats(),
            "seconds": round(time.time() - self.started, 1),
        }
