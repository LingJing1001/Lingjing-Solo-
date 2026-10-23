"""Unit test for the Lingjing v27 no-progress give-up (solver.py).

taaf/arcengine are not installable locally, so we stub them and import the
real solver module from the vendored duck-harness tree.
"""
import os
import sys
import types
import time
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "duck-harness" / "ARC3-Inference"


# ---------------------------------------------------------------- stub taaf
def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    # make the submodule reachable as an attribute of its parent too
    if "." in name:
        parent, _, leaf = name.rpartition(".")
        if parent in sys.modules:
            setattr(sys.modules[parent], leaf, m)
    return m


class _FakeSolverBase:
    """Minimal stand-in for taaf.solver.Solver (dataclass-ish base)."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


_stub("taaf")
_stub("taaf.game", Game=object, GameState=object)
_stub("taaf.solver", Solver=_FakeSolverBase)
_stub("arcengine", GameState=type("GameState", (), {
    "WIN": "WIN", "GAME_OVER": "GAME_OVER", "PLAYING": "PLAYING"}))

# stub the heavy inference.* deps so the REAL solver module can be imported
def _pkg(name, path):
    m = types.ModuleType(name)
    m.__path__ = [str(path)]
    sys.modules[name] = m
    return m


_inf = SRC / "inference"
_pkg("inference", _inf)
_pkg("inference.agent", _inf / "agent")
_pkg("inference.framework", _inf / "framework")
_stub("inference.agent.action_names",
      to_engine_action=lambda *a, **k: None,
      to_model_action=lambda *a, **k: None,
      to_model_actions=lambda *a, **k: [])
_stub("inference.agent.runtime_state",
      Frame=object, HistoryEntry=object, RUNTIME_STATE_FILENAME="state.json",
      write_runtime_state=lambda *a, **k: None)
_stub("inference.agent.tool_agent", ToolAgent=object)
_stub("inference.framework.kaggle",
      DEFAULT_QWEN_MODEL_DATASET_SOURCE="stub",
      DEFAULT_SERVED_MODEL_NAME="stub",
      DEFAULT_VLLM_MAX_MODEL_LEN=32768,
      DEFAULT_VLLM_PORT=1234,
      DEFAULT_VLLM_TENSOR_PARALLEL_SIZE=1,
      DEFAULT_VLLM_WHEELHOUSE_DATASET_SOURCE="stub",
      DEFAULT_WHEELHOUSE_STAMP_TEXT="stub",
      DuckKaggleVllmConfig=type("DuckKaggleVllmConfig", (), {}),
      duck_kaggle_dataset_sources=lambda *a, **k: [],
      duck_kaggle_setup_command=lambda *a, **k: "",
      duck_kaggle_teardown_command=lambda *a, **k: "")

sys.path.insert(0, str(SRC))

import inference.framework.solver as S  # noqa: E402

FAIL = []


def check(label, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label} {extra}")
    if not cond:
        FAIL.append(label)


class FakeRun:
    def __init__(self):
        self.history = []
        self.state = "playing"
        self.final_score = None
        self.solver_note = None


class FakeGame:
    number_of_levels = 8

    def __init__(self, levels=0):
        self.game_run = FakeRun()
        self._levels = levels

    class _State:
        def __init__(self, levels):
            self.levels_completed = levels
            self.won = False
            self.over = False
            self.data = None
            self.raw = types.SimpleNamespace(state="PLAYING")

    @property
    def current_state(self):
        return FakeGame._State(self._levels)

    def finish_game(self):
        if self.game_run.final_score is None:
            self.game_run.final_score = 0.0


def make_session(solver, game):
    tmp = Path(tempfile.mkdtemp())
    s = S._HarnessGameSession(
        solver=solver,
        game=game,
        analyzer=types.SimpleNamespace(_timeout=240, reported_tokens=0),
        game_index=0,
        pass_index=0,
        state_path=tmp / "state.json",
        transcript_path=tmp / "t.txt",
        analysis_html_relpath="a.html",
        stop_event=__import__("threading").Event(),
        viewer_data_path=tmp / "v.json",
    )
    return s


def solver_with(**kw):
    return S.HarnessSolver(**kw)


print("=== v27 no-progress give-up ===")

# 1) disabled by default -> no give-up
g = FakeGame()
s = make_session(solver_with(), g)
s._lj_last_progress_at = time.monotonic() - 99999
check("默认关闭：长时间无进展也不放弃", s.no_progress_reached() is False)

# 2) enabled via solver field -> gives up
g = FakeGame()
s = make_session(solver_with(no_progress_giveup_s=60.0), g)
s._lj_last_progress_at = time.monotonic() - 120
check("开启后：超时判定为放弃", s.no_progress_reached() is True)
check("放弃原因写入", bool(s._lj_gave_up_reason) and "no_progress" in s._lj_gave_up_reason,
      s._lj_gave_up_reason or "")

# 3) not yet expired
g = FakeGame()
s = make_session(solver_with(no_progress_giveup_s=600.0), g)
s._lj_last_progress_at = time.monotonic() - 10
check("未到时限不放弃", s.no_progress_reached() is False)

# 4) progress resets the clock (new action)
g = FakeGame()
s = make_session(solver_with(no_progress_giveup_s=600.0), g)
s._lj_last_progress_at = time.monotonic() - 500
s._lj_last_action_count = 0
g.game_run.history.append(object())          # action_count becomes 1
s._note_progress()
check("有新动作 -> 重置无进展计时", (time.monotonic() - s._lj_last_progress_at) < 1.0)

# 5) progress resets the clock (new level)
g = FakeGame(levels=0)
s = make_session(solver_with(no_progress_giveup_s=600.0), g)
s._lj_last_progress_at = time.monotonic() - 500
s._lj_last_level = 0
g._levels = 1
s._note_progress()
check("有新关卡 -> 重置无进展计时", (time.monotonic() - s._lj_last_progress_at) < 1.0)

# 6) should_stop() wires the give-up and records the note
g = FakeGame()
s = make_session(solver_with(no_progress_giveup_s=60.0), g)
s._lj_last_progress_at = time.monotonic() - 300
stopped = s.should_stop()
check("should_stop() 触发放弃", stopped is True)
check("solver_note 记录放弃原因",
      g.game_run.solver_note is not None and "no_progress" in g.game_run.solver_note,
      repr(g.game_run.solver_note))

# 7) env override
os.environ["LINGJING_NO_PROGRESS_GIVEUP_S"] = "45"
try:
    g = FakeGame()
    s = make_session(solver_with(), g)
    s._lj_last_progress_at = time.monotonic() - 100
    check("环境变量可覆盖阈值", s.no_progress_reached() is True)
finally:
    os.environ.pop("LINGJING_NO_PROGRESS_GIVEUP_S", None)

# 8) a game that keeps progressing must never be cut
g = FakeGame()
s = make_session(solver_with(no_progress_giveup_s=60.0), g)
for i in range(5):
    g.game_run.history.append(object())
    s._note_progress()
    time.sleep(0.01)
check("持续推进的局不会被误杀", s.no_progress_reached() is False)

print()
if FAIL:
    print(f"FAILED: {len(FAIL)} -> {FAIL}")
    sys.exit(1)
print("ALL PASS")
