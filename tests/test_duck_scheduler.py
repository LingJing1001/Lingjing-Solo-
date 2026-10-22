"""调度器的语义：钱花在还有人产出的一局上。

这套测试全部用假引擎和假 driver，理由和 `test_duck.py` 一样：要测的是"什么时候
把预算给谁"，不是引擎。真引擎每一步都要跑游戏逻辑，一步几十毫秒，二十几次分配
根本看不出来；假引擎可以把"这一局在什么步数上升级/死亡"直接写成剧本。

钉住的六件事：
1. 每条 lane 先各看一眼（探索），没人被饿死；
2. 有产出的 lane 后续拿到更多步数（利用）；
3. 总步数不超过预算，一局崩溃不拖垮全局；
4. 决策不含随机，输入相同则 trace 逐条相同；
5. 局结束（赢/死/驱动耗尽）后的处理各不相同；
6. reset_hook 真的挂上了，且拿到的是新棋盘。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "arc_adaptor"))

from duck.scheduler import Lane, PriorityScheduler   # noqa: E402


# --------------------------------------------------------------------- fixtures

class FakeRaw:
    """引擎原始帧的最小替身：`.frame` 是层栈，duck 的 `grid_of` 从这里读。"""

    def __init__(self, grid=None):
        self.frame = [grid or [[0, 0, 0, 0]] * 4]
        self.available_actions = [1, 2, 3, 4]
        self.levels_completed = 0
        self.state = type("S", (), {"name": "NOT_FINISHED"})()


class FakeResult:
    __slots__ = ("frame", "flags", "level", "state")

    def __init__(self, level=0, flags=None):
        self.frame = FakeRaw()
        self.flags = dict(flags or {})
        self.level = level
        self.state = "NOT_FINISHED"


class FakeEnv:
    """按剧本回放：`script[i]` 决定第 i 步的结果；脚本用完后重复最后一项。"""

    def __init__(self, game, script=None):
        self.game = game
        self.script = list(script or [])
        self.steps = 0
        self.resets = 0
        self.closed = False
        self.level = 0

    def step(self, name, spec=None):
        idx = min(self.steps, max(0, len(self.script) - 1))
        outcome = self.script[idx] if self.script else {}
        self.steps += 1
        if "level" in outcome:
            self.level = int(outcome["level"])
        return FakeResult(level=self.level, flags=outcome.get("flags"))

    def reset(self):
        self.resets += 1
        return FakeResult(level=self.level)

    def close(self):
        self.closed = True


class FakeDriver:
    """永远有下一个动作，直到被 retire。"""

    def __init__(self, env, *, action="ACTION2", limit=None, fail_on=None):
        self.env = env
        self.action = action
        self.limit = limit
        self.fail_on = fail_on          # 第 n 次 poll 抛异常
        self.polls = 0
        self.fed = []
        self.reset_hook = None

    def poll(self):
        self.polls += 1
        if self.fail_on is not None and self.polls == self.fail_on:
            raise RuntimeError("driver exploded")
        if self.limit is not None and self.polls > self.limit:
            return None
        return self.action, {}

    def feed(self, result):
        self.fed.append(result)

    def stats(self):
        return {"turns": self.polls, "tokens": 0, "refusals": {}}


def _lane(index, game, script=None, **driver_kw):
    env = FakeEnv(game, script)
    return Lane(index=index, game_id=game, env=env, driver=FakeDriver(env, **driver_kw))


def _steps_per_lane(scheduler):
    return [lane.steps for lane in scheduler.lanes]


# --------------------------------------------------------------------- 探索期

def test_every_lane_is_looked_at_before_any_lane_dominates():
    """四条同样贫瘠的 lane：步数应当基本均分，没有一条被饿死。"""
    lanes = [_lane(i, f"g{i}") for i in range(4)]
    sched = PriorityScheduler(lanes, total_steps=40, log=lambda s: None)
    sched.run()
    counts = _steps_per_lane(sched)
    assert min(counts) > 0, counts
    assert max(counts) - min(counts) <= 2, counts


def test_budget_is_never_exceeded():
    lanes = [_lane(i, f"g{i}") for i in range(3)]
    sched = PriorityScheduler(lanes, total_steps=17, log=lambda s: None)
    report = sched.run()
    assert report["steps"] <= 17, report
    assert sum(_steps_per_lane(sched)) == report["steps"]


# --------------------------------------------------------------------- 利用期

def test_productive_lane_gets_the_remaining_budget():
    """一局每两步升一级，其余毫无产出：它必须拿到明显更多的步数。"""
    good = _lane(0, "good", [{"level": i // 2} for i in range(400)])
    idle = [_lane(i, f"idle{i}") for i in range(1, 4)]
    sched = PriorityScheduler([good] + idle, total_steps=60, log=lambda s: None)
    sched.run()
    counts = _steps_per_lane(sched)
    assert counts[0] > max(counts[1:]) * 1.5, counts
    assert good.gained >= 2, good.stats()


def test_steps_are_split_by_priority_not_by_clock():
    """把墙钟推快十年，决策序列也必须一模一样。

    这是横评可复现的前提：同一份代码在慢机器上跑出的动作数不能变。
    """
    def run_once(drift):
        lanes = [_lane(0, "hot", [{"level": i // 2} for i in range(400)]),
                 _lane(1, "cold"), _lane(2, "cold2")]
        sched = PriorityScheduler(lanes, total_steps=40, log=lambda s: None)
        sched.started -= drift
        sched.run()
        return [(r["game"], r["action"], r["level"]) for r in sched.trace]

    assert run_once(0) == run_once(3600 * 24 * 365 * 10)


def test_two_identical_runs_produce_identical_traces():
    def run_once():
        lanes = [_lane(i, f"g{i}", [{"level": i} for _ in range(50)]) for i in range(3)]
        sched = PriorityScheduler(lanes, total_steps=30, log=lambda s: None)
        sched.run()
        return sched.trace

    first, second = run_once(), run_once()
    assert len(first) == len(second)
    assert [r["game"] for r in first] == [r["game"] for r in second]


# --------------------------------------------------------------------- 结束

def test_finished_lane_is_not_picked_again():
    lanes = [_lane(0, "quit", limit=2), _lane(1, "stay")]
    sched = PriorityScheduler(lanes, total_steps=20, log=lambda s: None)
    sched.run()
    assert lanes[0].finished and lanes[0].reason == "driver finished"
    assert lanes[0].steps <= 3, lanes[0].steps
    assert lanes[1].steps > lanes[0].steps


def test_win_marks_the_lane_finished():
    lane = _lane(0, "winner", [{"flags": {"run_complete": True}}])
    other = _lane(1, "other")
    sched = PriorityScheduler([lane, other], total_steps=20, log=lambda s: None)
    sched.run()
    assert lane.wins == 1 and lane.finished and lane.reason == "win"
    assert lane.steps == 1, lane.steps


def test_one_crashing_lane_does_not_take_down_the_others():
    broken = _lane(0, "broken", fail_on=2)
    ok = [_lane(i, f"ok{i}") for i in range(1, 3)]
    sched = PriorityScheduler([broken] + ok, total_steps=24, log=lambda s: None)
    report = sched.run()
    assert broken.finished and "RuntimeError" in broken.reason
    assert report["steps"] > len(ok), report
    assert all(lane.steps > 0 for lane in ok), [l.steps for l in ok]


def test_game_over_is_counted_without_ending_the_lane():
    lane = _lane(0, "dies", [{"flags": {"game_over": True}}, {"level": 1}])
    sched = PriorityScheduler([lane], total_steps=6, log=lambda s: None)
    sched.run()
    assert lane.deaths >= 1, lane.stats()
    assert lane.gained >= 1, lane.stats()


# --------------------------------------------------------------------- 接线

def test_reset_hook_is_wired_and_returns_a_fresh_frame():
    lane = _lane(0, "g")
    PriorityScheduler([lane], total_steps=4, log=lambda s: None)
    assert lane.driver.reset_hook is not None

    fresh = lane.driver.reset_hook()
    # 必须是引擎原始帧，不是 StepResult：给错一层，sandbox.adopt 会去索引包装
    # 对象本身，报 "'FrameDataRaw' object is not subscriptable"。
    assert isinstance(fresh, FakeRaw), type(fresh)
    assert fresh.frame and fresh.frame[-1], fresh.frame
    assert lane.env.resets == 1


def test_lane_level_tracks_the_hook():
    lane = _lane(0, "g")
    lane.env.level = 3
    PriorityScheduler([lane], total_steps=4, log=lambda s: None)
    lane.driver.reset_hook()
    assert lane.level == 3, lane.level


# --------------------------------------------------------------------- 落盘

def test_trace_is_dumped_and_replayable():
    lanes = [_lane(i, f"g{i}", [{"level": i} for _ in range(30)]) for i in range(2)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.json"
        sched = PriorityScheduler(lanes, total_steps=12, log=lambda s: None,
                                  trace_path=str(path))
        sched.run()
        assert path.exists()
        import json
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["budget"] == 12
        assert len(payload["trace"]) == payload["steps"] <= 12
        assert payload["trace"][0]["game"].startswith("g")


def test_scheduler_closes_every_engine():
    lanes = [_lane(i, f"g{i}") for i in range(2)]
    sched = PriorityScheduler(lanes, total_steps=4, log=lambda s: None)
    sched.run()
    assert all(lane.env.closed for lane in lanes)


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
