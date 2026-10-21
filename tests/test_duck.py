"""Pins the sandbox's semantics, not its implementation.

Every test here is a bug that already cost a run somewhere:

* a batch of two identical presses was refused as a "repeat", because the guard
  keyed on the board the action ENDED on;
* refusing a no-op permanently made the refutation unreachable;
* `frame_diff` mixed lists into a set and died on the first call;
* an action after game over executed against a board that no longer existed.

No engine, no network: a fake env returns frames we dictate.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "arc_adaptor"))

from duck.sandbox import (Sandbox, KnownNoOpActionError,   # noqa: E402
                          RepeatedActionInStateError, TerminalStateActionError,
                          interior_hash, normalise_actions)
from duck.views import FrameView, frame_diff, segment      # noqa: E402


# --------------------------------------------------------------------- fixtures

class FakeFrame:
    def __init__(self, grid, state="NOT_FINISHED", level=0):
        self.frame = [grid]
        self.available_actions = [1, 2, 3, 4]
        self.state = type("S", (), {"name": state})()
        self.levels_completed = level


class StepResult:
    __slots__ = ("frame", "flags", "state", "level")

    def __init__(self, frame, flags, state, level):
        self.frame = frame
        self.flags = flags
        self.state = state
        self.level = level


class FakeEnv:
    """Moves a single marker, unless told to swallow the action."""

    game_id = "fake"

    def __init__(self, grid, *, reactive=True):
        self.grid_state = [row[:] for row in grid]
        self.reactive = reactive
        self.calls = 0
        self.pending_state = "NOT_FINISHED"
        self.level = 0

    grid_of = staticmethod(lambda frame: [row[:] for row in frame.frame[-1]])

    def available(self, frame):
        names = ["UP", "DOWN", "LEFT", "RIGHT"]
        return names

    def step(self, engine_name, spec=None):
        self.calls += 1
        if self.reactive:
            moved = [row[:] for row in self.grid_state]
            found = False
            for r in range(len(moved)):
                for c in range(len(moved[0])):
                    if self.grid_state[r][c] == 5:
                        found = True
                        moved[r][c] = 0
                        nr = min(len(moved) - 1, r + 1) if engine_name == "ACTION2" else r
                        moved[nr][c] = 5
            if found:
                self.grid_state = moved
        flags = {}
        if self.pending_state == "GAME_OVER":
            flags["game_over"] = True
        if self.pending_state == "WIN":
            flags["run_complete"] = True
        return StepResult(FakeFrame(self.grid_state, self.pending_state, self.level),
                          flags, self.pending_state, self.level)


def _blank(h=12, w=12, bg=0):
    return [[bg] * w for _ in range(h)]


def _mark(grid, r, c, colour=5):
    grid[r][c] = colour


def _sandbox(**kw):
    grid = kw.pop("grid", None)
    if grid is None:
        grid = _blank()
        _mark(grid, 4, 4, 5)          # something must be able to move
    env = FakeEnv(grid, reactive=kw.pop("reactive", True))
    sb = Sandbox(env, max_seconds=2)
    sb.adopt(FakeFrame(env.grid_state), level=0)
    return sb, env


# --------------------------------------------------------------------- tests

def test_batch_of_identical_presses_executes():
    """`action(['DOWN','DOWN'])` is a normal batch, not a repeat."""
    sb, _ = _sandbox()
    result = sb.run("action(['DOWN', 'DOWN'])")
    assert result["ok"], result
    assert result["executed_now"] == 2, result


def test_same_action_twice_in_one_snippet_is_refused():
    """Only when the board did NOT move in between -- otherwise it is a normal
    batch, and `test_batch_of_identical_presses_executes` covers that."""
    sb, _ = _sandbox(reactive=False)
    result = sb.run("action('DOWN')\naction('DOWN')")
    assert not result["ok"]
    assert result["stop_reason"] == "repeated_action", result


def test_noop_refusal_is_overridable():
    """A hard block makes the counter-evidence unreachable, so insist must work."""
    sb, _ = _sandbox(reactive=False)
    first = sb.run("action('DOWN')")
    assert first["ok"], first
    assert sb.refusals["noop"] == 0

    blocked = sb.run("action('DOWN')")
    assert not blocked["ok"]
    assert blocked["stop_reason"] == "known_noop", blocked

    insisted = sb.run("action('DOWN')")
    assert insisted["ok"], insisted
    assert insisted["executed_now"] == 1


def test_noop_evidence_is_dropped_when_the_board_proves_it_wrong():
    """Once the same action does move things, the record must go -- otherwise the
    refusal fires forever on a board that has already changed."""
    sb, env = _sandbox(reactive=False)
    start = interior_hash(sb.current._grid, 1)
    sb.run("action('DOWN')")                       # learned as inert from `start`
    assert "ACTION2" in sb._noop.get(start, set())
    assert sb.run("action('DOWN')")["stop_reason"] == "known_noop"

    env.reactive = True                            # the probe now moves the marker
    insisted = sb.run("action('DOWN')")
    assert insisted["ok"] and insisted["executed_now"] == 1, insisted
    assert "ACTION2" not in sb._noop.get(start, set()), sb._noop


def test_action_after_game_over_is_refused():
    sb, env = _sandbox()
    env.pending_state = "GAME_OVER"
    sb.run("action('DOWN')")
    assert sb._terminal == "game_over"
    nxt = sb.run("action('DOWN')")
    assert not nxt["ok"]
    assert nxt["stop_reason"] == "terminal", nxt


def test_unavailable_action_is_rejected_before_stepping():
    sb, env = _sandbox()
    result = sb.run("action('MOUSE')")
    assert not result["ok"]
    assert result["stop_reason"] == "bad_action", result
    assert env.calls == 0


def test_retained_function_survives_into_the_next_snippet():
    sb, _ = _sandbox()
    first = sb.run("def colour(frame):\n    return frame.ascii[0]\nprint(colour(current_frame))")
    assert first["ok"], first
    assert first["retained_functions"] == ["colour"], first
    # no redefinition here: it must already exist
    second = sb.run("print(colour(current_frame))")
    assert second["ok"], second
    assert second["stdout"].strip()


def test_retained_functions_are_dropped_on_a_new_level():
    sb, _ = _sandbox()
    sb.run("def colour(frame):\n    return 1")
    sb.reset_level_scope()
    result = sb.run("print(colour(current_frame))")
    assert not result["ok"]
    assert "NameError" in result["error"], result


def test_click_coordinates_are_part_of_action_identity():
    specs = normalise_actions([{"action": "MOUSE", "row": 3, "col": 4},
                               {"action": "MOUSE", "row": 3, "col": 5}])
    assert specs[0]["row"] == 3 and specs[1]["col"] == 5
    assert specs[0] != specs[1], "two clicks on different cells are two actions"


def test_compact_click_form_is_parsed():
    spec, = normalise_actions("ACTION6@12,30")
    assert (spec["row"], spec["col"]) == (12, 30), spec


# --------------------------------------------------------------------- views

def test_segment_finds_one_node_per_object():
    g = _blank(10, 10)
    _mark(g, 1, 1, 5)
    _mark(g, 1, 2, 5)
    _mark(g, 5, 5, 7)
    nodes = segment(g)
    assert len(nodes) == 2, nodes
    assert {n.colour for n in nodes} == {5, 7}
    assert nodes[0].pixels == 2


def test_frame_diff_reports_a_move_not_a_disappearance():
    before = _blank(8, 8)
    _mark(before, 2, 2, 5)
    _mark(before, 2, 3, 5)
    after = _blank(8, 8)
    _mark(after, 4, 2, 5)
    _mark(after, 4, 3, 5)
    d = frame_diff(FrameView(before), FrameView(after))
    assert d["changed_cell_count"] == 4, d
    assert len(d["moved"]) == 1, d
    assert d["moved"][0]["delta"] == [2, 0], d
    assert not d.get("appeared"), d
    assert not d.get("disappeared"), d


def test_frame_diff_survives_many_moves():
    """The original died on the first call: lists inside a set."""
    before = _blank(8, 8)
    _mark(before, 1, 1, 5)
    after = _blank(8, 8)
    _mark(after, 2, 2, 5)
    d = frame_diff(FrameView(before), FrameView(after))
    assert d["moved"][0]["delta"] == [1, 1]


def test_interior_hash_ignores_the_border():
    """A timer bar on the rim must not make every state unique."""
    a = _blank(6, 6)
    _mark(a, 3, 3, 5)
    b = [row[:] for row in a]
    for c in range(6):
        b[0][c] = 9          # rim changes, interior identical
    assert interior_hash(a, 1) == interior_hash(b, 1)
    c = [row[:] for row in a]
    c[3][3] = 7
    assert interior_hash(a, 1) != interior_hash(c, 1)


def test_refusals_are_base_exceptions():
    """A defensive `except Exception` around action() must not swallow them."""
    for exc in (KnownNoOpActionError, TerminalStateActionError, RepeatedActionInStateError):
        assert issubclass(exc, BaseException), exc
        assert not issubclass(exc, Exception), f"{exc.__name__} is catchable"


if __name__ == "__main__":
    import traceback

    fns = [n for n in sorted(dir()) if n.startswith("test_")]
    ok = 0
    for name in fns:
        try:
            globals()[name]()
            print("PASS", name)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print("FAIL", name, "->", type(exc).__name__, exc)
            traceback.print_exc()
    print(f"{ok}/{len(fns)} passed")
    raise SystemExit(0 if ok == len(fns) else 1)
