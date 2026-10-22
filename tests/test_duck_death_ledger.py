"""死亡账本：同一盘面上已经证明会终结本局的动作，不许第二次悄悄花掉。

无动作守卫（`_noop`）保的是一步：按了没反应，白费一个动作预算。
死亡账本保的是一整局：同一个动作按下去直接 game_over，整次尝试归零。
两者是同一套机制的两个方向，所以这里的每个断言都对照 `test_duck.py` 里
noop 的那几条来写 —— 包括"必须留出反悔的出口"，因为"这个动作会死"终究
是对方程的一次观测，而不是一条定律。

同样不需要引擎和网络，FakeEnv 由测试直接指挥。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "arc_adaptor"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_duck import _blank, _mark, FakeEnv, FakeFrame   # noqa: E402

from duck.sandbox import Sandbox                          # noqa: E402


def _fresh():
    """一个 12x12 的空板 + 一个标记，env 由测试决定它是否会死。"""
    grid = _blank()
    _mark(grid, 4, 4, 5)
    env = FakeEnv(grid, reactive=True)
    sb = Sandbox(env, max_seconds=2)
    sb.adopt(FakeFrame(env.grid_state), level=0)
    return sb, env


def _die_once(sb, env):
    """让下一次 step 报 game_over，然后花掉它。"""
    env.pending_state = "GAME_OVER"
    result = sb.run("action('DOWN')")
    assert result["ok"], result
    assert sb._terminal == "game_over", sb._terminal
    # 引擎在 game_over 之后会把这一关重置；驱动层清掉 terminal，板子回到起点。
    sb.clear_terminal()
    env.pending_state = "NOT_FINISHED"
    env.grid_state = _blank()
    _mark(env.grid_state, 4, 4, 5)
    sb.adopt(FakeFrame(env.grid_state), level=0)
    return result


def test_fatal_action_is_remembered_against_its_start_board():
    sb, env = _fresh()
    start = sb._here()
    _die_once(sb, env)
    assert "ACTION2" in sb._deadly.get(start, set()), sb._deadly


def test_replaying_it_costs_nothing():
    sb, env = _fresh()
    _die_once(sb, env)
    calls_before = env.calls
    result = sb.run("action('DOWN')")
    assert not result["ok"]
    assert result["stop_reason"] == "known_deadly", result
    assert result["executed_now"] == 0, result
    assert env.calls == calls_before          # nothing reached the engine
    assert sb.refusals["deadly"] == 1


def test_the_model_can_still_insist():
    """反悔出口和 noop 一样：连续第二次就放行，否则反证永远做不出来。"""
    sb, env = _fresh()
    _die_once(sb, env)
    first = sb.run("action('DOWN')")
    assert first["stop_reason"] == "known_deadly"
    second = sb.run("action('DOWN')")
    assert second["ok"] and second["executed_now"] == 1, second


def test_a_new_level_forgets_the_ledger():
    sb, env = _fresh()
    _die_once(sb, env)
    sb.reset_level_scope()                    # 换关：死法不再适用
    sb.adopt(FakeFrame(env.grid_state), level=1)
    result = sb.run("action('DOWN')")
    assert result["ok"] and result["executed_now"] == 1, result


def test_a_harmless_noop_does_not_land_in_the_ledger():
    """不动的动作进 noop 账，不进死亡账 —— 否则守卫会开始拒绝安全动作。"""
    grid = _blank()
    _mark(grid, 4, 4, 5)
    env = FakeEnv(grid, reactive=False)       # 按了完全没反应
    sb = Sandbox(env, max_seconds=2)
    sb.adopt(FakeFrame(env.grid_state), level=0)
    result = sb.run("action('DOWN')")
    assert result["ok"], result
    assert sb._noop, "the no-op guard should have recorded it"
    assert not sb._deadly, sb._deadly


def test_surviving_the_action_refutes_the_ledger_entry():
    """同一盘面重开后活下来了，那条"会死"的假设就该被推翻。

    和 noop 守卫的规则一致（板子真的动了就撤销记录）：否则一次偶发死亡会让
    这个动作在同一盘面上被永久拒绝。
    """
    sb, env = _fresh()
    _die_once(sb, env)
    start = sb._here()
    assert "ACTION2" in sb._deadly.get(start, set())

    # 模型坚持了一次，这次没死（引擎这次不报 game_over）
    assert sb.run("action('DOWN')")["stop_reason"] == "known_deadly"
    survived = sb.run("action('DOWN')")
    assert survived["ok"] and survived["executed_now"] == 1, survived

    # 回到起始板：账本里那条应当已经失效，不再拒绝
    env.grid_state = _blank()
    _mark(env.grid_state, 4, 4, 5)
    sb.adopt(FakeFrame(env.grid_state), level=0)
    again = sb.run("action('DOWN')")
    assert again["ok"] and again["executed_now"] == 1, again


# --------------------------------------------------- 可暂停驱动的那条路径


def test_deferred_path_records_the_fatal_action():
    """`DeferredSandbox` 自己重写了 `_do_action`，所以账本要在两处都成立。

    这里走的是真正会被 harness 用的路径：动作先交出去，结果最后从 `deliver`
    回来 —— 记录必须发生在 deliver 里，因为只有那时才知道死没死。
    """
    from duck.deferred import DeferredEnv, DeferredSandbox
    from duck.env import StepResult

    grid = _blank()
    _mark(grid, 4, 4, 5)

    class Env(DeferredEnv):
        def __init__(self):
            super().__init__("fake")

    sb = DeferredSandbox(Env())
    sb.adopt(FakeFrame(grid), level=0)
    start = sb._here()

    first = sb.run("action('DOWN')")
    pending = first.get("pending")
    assert pending is not None, first

    # harness 花掉这个动作，引擎报 game_over
    sb.deliver(StepResult(FakeFrame(grid), grid, {"game_over": True},
                          "GAME_OVER", 0), "ACTION2", {"action": "DOWN"}, start)
    assert "ACTION2" in sb._deadly.get(start, set()), sb._deadly

    # 重开回同一盘面：这一次必须在交出去之前就被拒绝
    sb.clear_terminal()
    sb.adopt(FakeFrame(grid), level=0)
    second = sb.run("action('DOWN')")
    assert not second["ok"], second
    assert second["stop_reason"] == "known_deadly", second
    assert "pending" not in second, second


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS {name}")
            except Exception as exc:  # noqa: BLE001
                fails += 1
                print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'FAILED' if fails else 'OK'} ({fails} failure(s))")
    raise SystemExit(1 if fails else 0)
