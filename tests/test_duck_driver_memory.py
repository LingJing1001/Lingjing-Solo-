"""End-to-end wiring of solved-level memory through the resumable driver.

`test_duck_memory.py` pins the block's contract in isolation. This file pins the
part that can only break at the seams, and each assertion is a failure mode we
would otherwise only notice after a run:

* a cleared level is recorded with the **winning attempt's** actions -- a RESET
  in between must not leak the dead attempt's actions into the record;
* the model is asked for its rule on the turns right after the clear;
* the rule line lands in the block verbatim, and the record stops asking;
* `pin_policy="drain"` leaves the system prompt alone until a compaction (that is
  what keeps the prompt cache alive), `pin_policy="level"` writes it at level-up.

No engine and no network: `FakeEnv` inherits `DeferredEnv`'s read methods, and
step results are dictated by the test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "arc_adaptor"))

from duck.deferred import DeferredEnv, DuckDriver       # noqa: E402
from duck.env import StepResult                         # noqa: E402

GRID = [[0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0]]


class FakeFrame:
    """Just enough of `FrameDataRaw`: layers for `grid_of`, actions for `available`."""

    def __init__(self, grid=GRID):
        self.frame = [grid]
        self.available_actions = [1, 2, 3, 4]


class FakeEnv(DeferredEnv):
    def __init__(self):
        super().__init__("fake")
        self.pending = None


def _snippet(code: str) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": "call_1", "type": "function",
        "function": {"name": "python", "arguments": json.dumps({"code": code})}}]}


class ScriptChat:
    def __init__(self, script):
        self.script = list(script)
        self.i = 0
        self.calls = 0
        self.seen: list[list[dict]] = []

    def __call__(self, messages, tools=None):
        self.calls += 1
        self.seen.append([dict(m) for m in messages])
        if self.i >= len(self.script):
            return {"role": "assistant", "content": "done", "tool_calls": []}
        item = self.script[self.i]
        self.i += 1
        return item

    def tokens(self) -> int:
        return 0


_TICKS = [0]


def _result(level: int, *, cleared_from: int | None = None, over: bool = False) -> StepResult:
    """A step result whose board actually differs from the previous one.

    A frozen board would trip the sandbox's no-op guard on the second identical
    action, and the test would then be measuring the guard instead of the memory.
    """
    _TICKS[0] += 1
    n = _TICKS[0]
    grid = [row[:] for row in GRID]
    # Interior only: the no-op guard hashes the board with a 1-cell border cropped
    # away (where a timer bar lives), so a change on the rim is invisible to it.
    grid[1 + (n % 2)][1 + ((n // 2) % 2)] = 2
    flags: dict = {}
    if cleared_from is not None:
        flags["level_completed"] = True
        flags["from_level"] = cleared_from
    if over:
        flags["game_over"] = True
    state = "GAME_OVER" if over else "NOT_FINISHED"
    return StepResult(FakeFrame(grid), grid, flags, state, level)


def _driver(script, *, pin_policy="level", **kw):
    chat = ScriptChat(script)
    return DuckDriver(chat, game_id="fake", frame=FakeFrame(), env=FakeEnv(),
                      level=0, pin_policy=pin_policy, verbose=False, **kw), chat


def test_clear_records_actions_and_asks_for_a_rule():
    driver, chat = _driver([
        _snippet("action('DOWN')"),
        {"role": "assistant", "content":
            "Rule for level 0: pressing DOWN moves the block one cell down while the "
            "border bar ticks once per action."},
        _snippet("action('RIGHT')"),
    ])

    assert driver.poll()[0] == "ACTION2"          # the action duck asks the harness for
    driver.feed(_result(1, cleared_from=0))       # harness reports the clear
    driver.poll()                                 # delivers it, records, asks, resumes

    rec = driver.memory.get(0)
    assert rec is not None and rec.n_actions == 1
    assert rec.actions == ["DOWN"]                 # model names, not ACTION2
    assert rec.rule.startswith("pressing DOWN")

    joined = "\n".join(m["content"] for m in chat.seen[-1] if m["role"] == "user")
    assert "Rule for level 0:" in joined           # it was actually asked
    assert driver.level == 1

    # the block is pinned at the END, so the shared prompt prefix is untouched
    assert driver.messages[0]["role"] == "system"
    assert driver.messages[0]["content"].rstrip().endswith("=== END SOLVED LEVELS ===")
    assert "Your rule: pressing DOWN" in driver.messages[0]["content"]


def test_reset_discards_the_dead_attempt_but_keeps_the_winning_one():
    driver, _ = _driver([
        _snippet("action('LEFT')"),
        _snippet("action('RESET')"),
        _snippet("action('UP')"),
        _snippet("action('UP')"),
    ])
    assert driver.poll()[0] == "ACTION3"
    driver.feed(_result(0))               # nothing learned
    assert driver.poll()[0] == "RESET"
    driver.feed(_result(0))               # attempt abandoned
    assert driver.poll()[0] == "ACTION1"
    driver.feed(_result(0))
    assert driver.poll()[0] == "ACTION1"
    driver.feed(_result(1, cleared_from=0))   # this attempt is the one that cleared
    driver.poll()

    rec = driver.memory.get(0)
    assert rec.actions == ["UP", "UP"]     # LEFT belongs to the dead attempt
    assert rec.n_actions == 2


def test_pin_policy_drain_waits_for_compaction():
    driver, _ = _driver([
        _snippet("action('DOWN')"),
        {"role": "assistant", "content":
            "Rule for level 0: DOWN always moves the block one cell and spends one tick."},
        _snippet("action('RIGHT')"),
    ], pin_policy="drain")

    driver.poll()
    driver.feed(_result(1, cleared_from=0))
    driver.poll()

    assert driver.memory.get(0).rule            # the rule was captured
    assert "=== SOLVED LEVELS" not in driver.messages[0]["content"]   # ...but not pinned

    driver._compact()                           # a drain: the server re-reads anyway
    assert "=== SOLVED LEVELS" in driver.messages[0]["content"]
    assert "Your rule: DOWN always moves" in driver.messages[0]["content"]


def test_rule_is_not_asked_for_forever():
    driver, chat = _driver([
        _snippet("action('DOWN')"),
        {"role": "assistant", "content": "ignoring the request"},
        _snippet("action('RIGHT')"),
        _snippet("action('LEFT')"),
    ], max_inner=2)

    driver.poll()
    driver.feed(_result(1, cleared_from=0))
    driver.poll()
    driver.feed(_result(1))
    driver.poll()

    # the last snapshot holds every message ever appended, so an ask is counted once
    final = chat.seen[-1]
    asked = sum(1 for m in final
                if m["role"] == "user" and "Rule for level 0:" in (m.get("content") or ""))
    assert 1 <= asked <= 2                     # bounded, and it did try
    assert driver.memory.get(0).rule == ""      # silence is recorded as "not stated"
    assert "Your rule: not stated" in driver.memory.block()


def test_game_over_is_not_a_level_clear():
    driver, _ = _driver([
        _snippet("action('DOWN')"),
        {"role": "assistant", "content": "let me look"},
        _snippet("action('UP')"),
    ], max_inner=2)
    driver.poll()
    driver.feed(_result(0, over=True))
    driver.poll()
    assert len(driver.memory) == 0              # a death records nothing
    assert driver.sandbox._terminal is None     # and the terminal latch was released


# --------------------------------------------------------------- DuckAgent path
# `deferred.py` is the harness bridge; `agent.py` is the standalone loop that
# `scripts/duck_play.py` drives. Both must remember cleared levels.

class SteppingEnv(FakeEnv):
    """An env the non-deferred `Sandbox` can step directly."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def reset(self):
        return _result(0)

    def step(self, name, spec=None):
        self.calls += 1
        if self.calls == 2:                     # the second action clears level 0
            return _result(1, cleared_from=0)
        return _result(0)


def test_duck_agent_records_and_pins_after_a_clear():
    from duck.agent import DuckAgent

    chat = ScriptChat([
        _snippet("action('DOWN')"),
        _snippet("action('DOWN')"),
        {"role": "assistant", "content":
            "Rule for level 0: DOWN walks the marker to the row below, one cell per press."},
        _snippet("action('LEFT')"),
        _snippet("action('RIGHT')"),
    ])
    env = SteppingEnv()
    agent = DuckAgent(env, chat, max_steps=4, max_turns=4,
                      pin_policy="level", verbose=False)
    stats = agent.play()

    assert stats["best_level"] == 1
    rec = agent.memory.get(0)
    assert rec is not None and rec.actions == ["DOWN", "DOWN"]   # the winning attempt
    assert rec.rule.startswith("DOWN walks the marker")
    assert "=== SOLVED LEVELS" in agent.messages[0]["content"]
    assert "Your rule: DOWN walks" in agent.messages[0]["content"]
    assert stats["solved_levels"]["levels"] == 1
