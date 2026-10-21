"""The python tool: run model-written code against a live frame.

This is the champion harness's actual decision mechanism. There is no policy in
this file -- the model writes the policy, per turn, as code. What this file
provides is:

* a stable set of names to read the board with (`current_frame`, `history`,
  `transitions`, `frame_diff`);
* exactly one way to spend an action (`action(...)`), so cost accounting is
  honest;
* refusals that cannot be swallowed. The champion's note explains why they
  derive from BaseException: models wrap `action(...)` in `try/except Exception`
  routinely, and one run treated 1500 consecutive refusals as proof that every
  cell on the board was inert.

Deliberate simplifications against the champion, all of which cost capability
and none of which change the shape of the loop:

* `exec` in this process, not `python -I -S` in a child. Our snippets are written
  by our own model, not by an untrusted user, so the child process buys isolation
  we do not need -- and it would cost a serialisation boundary around the env.
* No animation frames, no image attachments, no death ledger. Those need
  per-action frame counts the local engine does not expose.
"""
from __future__ import annotations

import ast
import io
import time
import traceback
from contextlib import redirect_stdout

# Engine name -> the name the model sees. The mapping follows the ARC-AGI-3
# convention, and the prompt says outright that what these controls DO depends on
# the game: on ar25, ACTION3 was measured moving the piece RIGHT, so a model that
# trusts the label over the experiment will walk the wrong way. The label is a
# handle, not a promise.
MODEL_TO_ENGINE = {
    "UP": "ACTION1", "DOWN": "ACTION2", "LEFT": "ACTION3", "RIGHT": "ACTION4",
    "SPACE": "ACTION5", "MOUSE": "ACTION6", "CLICK": "ACTION6", "UNDO": "ACTION7",
    "RESET": "RESET",
    "ACTION1": "ACTION1", "ACTION2": "ACTION2", "ACTION3": "ACTION3",
    "ACTION4": "ACTION4", "ACTION5": "ACTION5", "ACTION6": "ACTION6",
    "ACTION7": "ACTION7",
}

_TERMINAL_DETAIL = {
    "game_over": ("the previous action reported game_over. The level has been reset to its "
                  "start, so the board you planned against is gone and this action was not "
                  "executed."),
    "level_completed": ("the previous action reported level_completed. A new level has been "
                        "loaded, so the board you planned against is gone and this action "
                        "was not executed."),
    "run_complete": ("the previous action reported run_complete. The run is finished and this "
                     "action was not executed."),
}


class KnownNoOpActionError(BaseException):
    """Raised INSTEAD of executing an action already proven inert here.

    BaseException on purpose: a `try/except Exception` around action() must not be
    able to turn a refusal into an apparent success.
    """


class TerminalStateActionError(BaseException):
    """Raised when a snippet acts after the attempt or level already ended."""


class RepeatedActionInStateError(BaseException):
    """Raised when one snippet runs the same action from the same board twice.

    Buggy loop code spends a whole budget getting nowhere -- a cursor that wraps
    turns RIGHT into a short cycle. Writing a new snippet IS the response, so
    unlike the no-op guard there is no override slot.
    """


class ActionError(Exception):
    """Bad argument, or an action not currently available. Catchable."""


def action_signature(spec: dict) -> str:
    """Identity of one action. Click coordinates are part of it: a click is only
    'the same action' at the same cell."""
    name = str(spec.get("action") or "").upper()
    if name in {"MOUSE", "CLICK", "ACTION6"}:
        return f"ACTION6({spec.get('row')},{spec.get('col')})"
    return MODEL_TO_ENGINE.get(name, name)


def normalise_actions(value) -> list[dict]:
    """str | dict | list of those -> [{action, row?, col?}]."""
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    out = []
    for item in items:
        if isinstance(item, dict):
            name = str(item.get("action") or item.get("name") or "").strip().upper()
            spec = {"action": name}
            if name in {"MOUSE", "CLICK", "ACTION6"}:
                row, col = item.get("row"), item.get("col")
                if row is None or col is None:
                    raise ActionError("MOUSE needs integer `row` and `col`")
                spec["row"], spec["col"] = int(row), int(col)
            out.append(spec)
            continue
        if isinstance(item, (int, float)):
            out.append({"action": f"ACTION{int(item)}"})
            continue
        text = " ".join(str(item).split()).upper()
        if not text:
            continue
        # 'ACTION6@12,30' is the compact form used elsewhere in this repo
        if "@" in text:
            head, _, tail = text.partition("@")
            try:
                row, col = (int(v) for v in tail.replace(" ", "").split(","))
            except ValueError as exc:
                raise ActionError(f"cannot read click coordinates from {item!r}") from exc
            out.append({"action": MODEL_TO_ENGINE.get(head, head), "row": row, "col": col})
            continue
        out.append({"action": text})
    return out


def interior_hash(grid: list[list[int]], border: int = 1) -> str:
    """Hash of the board with its border cropped.

    The border is where a timer or step bar lives, and it changes every action.
    Hashing the full board makes every state unique, which silently disables
    every guard keyed on state identity -- the champion hit this and fixed it the
    same way (`_state_identity_border`).
    """
    if not grid:
        return ""
    h, w = len(grid), len(grid[0])
    b = max(0, int(border))
    if h <= 2 * b or w <= 2 * b:
        b = 0
    rows = []
    for r in range(b, h - b):
        rows.append(",".join(f"{int(v):x}" for v in grid[r][b:w - b]))
    return "|".join(rows)


class Sandbox:
    """One live level. `run(code)` executes a snippet against it."""

    def __init__(self, env, *, border: int = 1, max_seconds: float = 12.0,
                 max_stdout: int = 2400):
        self.env = env
        self.border = border
        self.max_seconds = max_seconds
        self.max_stdout = max_stdout

        self.retained: dict[str, str] = {}
        self.history: list = []
        self.transitions: list = []
        self.current = None          # views.FrameView
        self.previous = None
        self.last_result: dict = {}
        self.valid_actions: list[str] = []
        self.level: int = 0
        self.steps: int = 0

        # guards
        self._noop: dict[str, set[str]] = {}      # interior_hash -> {action_sig}
        self._armed_noop: tuple[str, str] | None = None
        self._terminal: str | None = None
        self._executed_in_snippet: set[tuple[str, str]] = set()
        self.refusals = {"noop": 0, "terminal": 0, "repeated": 0}

    # ------------------------------------------------------------------ state

    def adopt(self, frame, *, level: int | None = None) -> None:
        """Take the engine's current frame as truth."""
        from .views import FrameView

        grid = self.env.grid_of(frame)
        self.previous = self.current
        self.current = FrameView(grid)
        if level is not None:
            if level != self.level:
                self._noop.clear()
                self._armed_noop = None
            self.level = level
        avail = self.env.available(frame)
        if avail:
            self.valid_actions = avail

    def _here(self) -> str:
        return interior_hash(self.current._grid, self.border) if self.current else ""

    # ------------------------------------------------------------------ action

    def _do_action(self, specs: list[dict]) -> dict:
        """Execute a batch. Raises the refusal types above."""
        if not specs:
            return {"executed": 0, "note": "no actions given"}
        if self._terminal:
            self.refusals["terminal"] += 1
            raise TerminalStateActionError(_TERMINAL_DETAIL.get(self._terminal, self._terminal))

        here = self._here()
        executed = []
        changed = None
        result = None
        for spec in specs:
            name = MODEL_TO_ENGINE.get(spec["action"], spec["action"])
            sig = action_signature(spec)
            allowed = {MODEL_TO_ENGINE.get(a, a) for a in self.valid_actions} or None
            if name != "RESET" and allowed and name not in allowed:
                raise ActionError(
                    f"{spec['action']} is not available; valid_actions={self.valid_actions}")

            # Keyed on the board the action STARTS from. Recording the board it
            # ends on makes the second press of a batch look like a repeat of the
            # first, which refuses every legitimate `action(['DOWN', 'DOWN'])`.
            start = here
            if (start, sig) in self._executed_in_snippet:
                self.refusals["repeated"] += 1
                raise RepeatedActionInStateError(
                    f"{spec['action']} was already executed from this exact board inside "
                    f"this snippet. If you meant to repeat it, write a new snippet.")

            known = self._noop.get(start, set())
            if sig in known:
                key = (start, sig)
                if self._armed_noop == key:
                    self._armed_noop = None      # the model insisted: let it run
                else:
                    self._armed_noop = key
                    self.refusals["noop"] += 1
                    raise KnownNoOpActionError(
                        f"{spec['action']} is known to change nothing from this exact board. "
                        f"Nothing ran. If you have a specific reason, issue it again.")

            before_grid = self.current._grid if self.current else None
            result = self.env.step(name, spec)
            executed.append({"action": spec["action"], **{k: v for k, v in spec.items()
                                                          if k != "action"}})
            self.adopt(result.frame, level=result.level)
            self.previous_grid = before_grid

            changed = self._gameplay_changed(before_grid, self.current._grid)
            self.transitions.append({
                "action": spec["action"],
                "before_frame": before_grid,
                "after_frame": self.current._grid,
                "result": {**result.flags, "gameplay_changed": changed},
            })
            self.history.append({"action": spec["action"], "result": result.flags})
            self.steps += 1

            if changed is False:
                self._noop.setdefault(start, set()).add(sig)
            elif changed is True:
                # evidence contradicts the recorded no-op: drop it, or the same
                # refusal will fire forever on a board that has since changed
                self._noop.get(start, set()).discard(sig)
            self._executed_in_snippet.add((start, sig))
            here = self._here()

            flag = result.flags
            for terminal in ("game_over", "run_complete", "level_completed", "done"):
                if flag.get(terminal):
                    self._terminal = "run_complete" if terminal == "done" else terminal
                    break
            if self._terminal:
                break

        self.last_result = {
            "executed": len(executed),
            "actions": executed,
            "result": {**result.flags, "gameplay_changed": changed},
            "state": result.state,
            "level": self.level,
            "steps": self.steps,
        }
        return self.last_result

    def _gameplay_changed(self, before, after):
        """Ignoring the border, where a timer bar lives."""
        if before is None or after is None or len(before) != len(after):
            return None
        b = self.border
        h, w = len(before), len(before[0]) if before else 0
        if h <= 2 * b or w <= 2 * b:
            b = 0
        for r in range(b, h - b):
            if before[r][b:w - b] != after[r][b:w - b]:
                return True
        return False

    # ------------------------------------------------------------------ run

    def run(self, code: str) -> dict:
        """Execute one snippet. Returns a plain dict for the tool message."""
        self._executed_in_snippet.clear()
        steps_before = self.steps
        from .views import frame_diff as _diff

        out = io.StringIO()
        result: dict = {"ok": True}
        g: dict = {}
        g.update(self._base_globals(_diff))
        try:
            with redirect_stdout(out):
                self._exec_guarded(code, g)
        except KnownNoOpActionError as exc:
            result.update(ok=False, stop_reason="known_noop", error=str(exc))
        except TerminalStateActionError as exc:
            result.update(ok=False, stop_reason="terminal", error=str(exc))
        except RepeatedActionInStateError as exc:
            result.update(ok=False, stop_reason="repeated_action", error=str(exc))
        except ActionError as exc:
            result.update(ok=False, stop_reason="bad_action", error=str(exc))
        except BaseException as exc:  # noqa: BLE001 - the model's code, not ours
            result.update(ok=False, stop_reason="exception",
                          error=f"{type(exc).__name__}: {exc}",
                          traceback=self._short_tb())
        finally:
            text = out.getvalue()
        if len(text) > self.max_stdout:
            text = text[-self.max_stdout:]
            result["stdout_truncated"] = True
        result["stdout"] = text
        # what THIS snippet spent, not what the level has spent so far: the trace
        # and the turn boundary both depend on the difference
        result["executed_now"] = self.steps - steps_before
        kept = self._capture_functions(code, g)
        if kept:
            result["retained_functions"] = kept
        if self.last_result:
            result["last_action_call_result"] = self.last_result
        result["level"] = self.level
        result["steps_used"] = self.steps
        result["valid_actions"] = self.valid_actions
        return result

    def _base_globals(self, diff_fn) -> dict:
        def action(actions):
            return self._do_action(normalise_actions(actions))

        def frame_diff(before=None, after=None):
            from .views import FrameView

            b = before if before is not None else self.previous
            a = after if after is not None else self.current
            if isinstance(b, list) and b and isinstance(b[0], list):
                b = FrameView(b)
            if isinstance(a, list) and a and isinstance(a[0], list):
                a = FrameView(a)
            if b is None or a is None:
                return {"error": "need two frames (is this the first turn?)"}
            return diff_fn(b, a)

        return {
            "current_frame": self.current,
            "previous_frame": self.previous,
            "history": self.history,
            "transitions": self.transitions,
            "frame_diff": frame_diff,
            "action": action,
            "last_action_call_result": self.last_result,
            "valid_actions": self.valid_actions,
            "KnownNoOpActionError": KnownNoOpActionError,
            "TerminalStateActionError": TerminalStateActionError,
            "RepeatedActionInStateError": RepeatedActionInStateError,
        }

    def _exec_guarded(self, code: str, g: dict) -> None:
        """exec with a wall-clock budget.

        No threads: a stuck thread cannot be killed in CPython, and the champion's
        own budget is a child process. A line-level trace hook is the portable
        equivalent, and it is only paid while a snippet runs.
        """
        deadline = time.perf_counter() + self.max_seconds

        def tracer(frame, event, arg):
            if event == "line" and time.perf_counter() > deadline:
                raise TimeoutError(
                    f"snippet exceeded {self.max_seconds:.0f}s and was stopped")
            return tracer

        # retained functions first: this is the whole point of keeping them
        prelude = "\n\n".join(self.retained.values())
        prev_trace = None
        try:
            import sys
            prev_trace = sys.gettrace()
            sys.settrace(tracer)
            if prelude:
                exec(compile(prelude, "<retained>", "exec"), g)
            exec(compile(code, "<snippet>", "exec"), g)
        finally:
            import sys
            sys.settrace(prev_trace)

    def _capture_functions(self, code: str, g: dict) -> list[str]:
        """Keep top-level `def`s that actually got defined, for the rest of the level.

        The champion keeps them "for the rest of this level"; so do we, because the
        level is the unit whose mechanics the functions encode.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        kept = []
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name not in g or not callable(g[node.name]):
                continue
            try:
                src = ast.get_source_segment(code, node)
            except Exception:  # noqa: BLE001
                src = None
            if not src:
                continue
            self.retained[node.name] = src
            kept.append(node.name)
        return kept

    @staticmethod
    def _short_tb(limit: int = 3) -> str:
        lines = traceback.format_exc().strip().splitlines()
        return "\n".join(lines[-limit:])

    def reset_level_scope(self) -> None:
        """New level: drop the guards and the retained functions, keep nothing.

        Retained functions encode the mechanics of the level that defined them.
        Carrying them into a new level silently re-applies a model of a board that
        no longer exists -- the champion scopes them per level for that reason.
        """
        self.retained.clear()
        self._noop.clear()
        self._armed_noop = None
        self._terminal = None
        self.history.clear()
        self.transitions.clear()
        # `steps` is NOT reset: the budget belongs to the game, not the level.
        # Clearing it here would let every level start the action count over.

    def clear_terminal(self) -> None:
        """After an automatic reset the level is playable again.

        Separate from `reset_level_scope` because a death keeps the level: the
        no-op records still describe this board, so they must survive.
        """
        self._terminal = None
