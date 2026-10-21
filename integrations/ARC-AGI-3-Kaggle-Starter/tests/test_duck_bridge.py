"""duck 接进路由的那道缝：把契约逐条钉死。

这些断言每一条都对应一个我们只会在跑完之后才发现的问题：

* `off` 模式下连 bridge 对象都不该留下 —— 否则 44.57 那次的执行路径就变了；
* `auto` 模式下端点连不上必须是**安静的死路**，而不是等一次 120 秒超时；
* 没有 R3 的失败请求（或请求是别的关卡的）时，一次模型调用都不许发生；
* poll → feed → poll 的握手错一步，动作结果会静默丢失（deferred.py 的模块头
  解释过为什么），所以这里必须真的按 tick 推进，不能一次跑完；
* 模型挂了 / 预算烧完，只能是"退回原流程"，不能把异常抛回 `choose_action`。

不需要引擎、不需要网络：chat 由脚本喂，帧由测试给。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT.parents[1]
for p in (str(ROOT), str(CANON), str(ROOT / "vendor"),
          str(ROOT / "vendor" / "ARC-AGI-3-Agents"), str(ROOT / "agent")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("LINGJING_SRC", str(CANON / "lingjing_solo"))

GRID = [[0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]]


class _State:
    def __init__(self, name: str):
        self.name = name


class FakeFrame:
    """`FrameData` 在 `wrap_raw` / `DeferredEnv` 眼里的样子。"""

    def __init__(self, grid=None, level=0, state="NOT_FINISHED"):
        self.frame = [grid or GRID]
        self.levels_completed = level
        self.state = _State(state)
        self.available_actions = [1, 2, 3, 4, 6]


class ScriptChat:
    """按脚本返回带 tool_call 的 assistant 消息。"""

    def __init__(self, script):
        self.script = list(script)
        self.i = 0
        self.calls = 0

    def __call__(self, messages, tools=None):
        self.calls += 1
        if self.i >= len(self.script):
            return {"role": "assistant", "content": "", "tool_calls": []}
        item = self.script[self.i]
        self.i += 1
        return item

    def tokens(self) -> int:
        return 0


def _snippet(code: str) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": "call_1", "type": "function",
        "function": {"name": "python", "arguments": json.dumps({"code": code})}}]}


def _bridge(script=(), **env):
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update({str(k): str(v) for k, v in env.items()})
    try:
        import duck_bridge
        chat = ScriptChat(script) if script else None
        return duck_bridge.DuckBridge("fake", chat=chat, log=lambda *_: None)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ------------------------------------------------------------------ 开关语义


def test_off_leaves_no_object():
    bridge = _bridge(LINGJING_DUCK="off")
    assert bridge.enabled is False
    assert bridge.reason == "LINGJING_DUCK=off"
    assert bridge.tick(FakeFrame(), 0, request=0) is None


def test_default_mode_is_off():
    """不设环境变量时必须是 off：已知分数 44.57 不能因为一条新线而变。"""
    saved = os.environ.pop("LINGJING_DUCK", None)
    try:
        import duck_bridge
        bridge = duck_bridge.DuckBridge("fake", chat=None, log=lambda *_: None)
    finally:
        if saved is not None:
            os.environ["LINGJING_DUCK"] = saved
    assert bridge.enabled is False
    assert "off" in bridge.reason


def test_auto_without_endpoint_is_inert_and_quiet():
    bridge = _bridge(LINGJING_DUCK="auto",
                     LINGJING_LLM_URL="http://127.0.0.1:9/v1",
                     LINGJING_DUCK_PROBE="0.2")
    assert bridge.enabled is False
    assert "unreachable" in bridge.reason
    assert bridge.tick(FakeFrame(), 0, request=0) is None


def test_forced_mode_does_not_probe():
    # on = 本机已确认服务在跑；连不上也要接（第一次调用失败会自己退回）。
    bridge = _bridge(LINGJING_DUCK="on", LINGJING_LLM_URL="http://127.0.0.1:9/v1")
    assert bridge.enabled is True


# ----------------------------------------------------------------- 门禁语义


def test_no_request_means_no_model_call():
    chat = ScriptChat([_snippet("action('DOWN')")])
    bridge = _bridge(LINGJING_DUCK="on")
    bridge._chat = chat                      # 注入，绕开端点
    assert bridge.tick(FakeFrame(), 0, request=None) is None
    assert chat.calls == 0


def test_request_for_another_level_is_ignored():
    chat = ScriptChat([_snippet("action('DOWN')")])
    bridge = _bridge(LINGJING_DUCK="on")
    bridge._chat = chat
    # R3 是在 L2 失败的，当前已在 L3：那条结论不作数。
    assert bridge.tick(FakeFrame(level=3), 3, request=2) is None
    assert chat.calls == 0
    assert bridge.enabled is True


# ------------------------------------------------------------- 每 tick 一动作


def test_one_action_per_tick_and_result_is_delivered():
    # 一次 `action([...])` 就是一批：duck 会在后续每个 tick 里各吐一个。
    chat = ScriptChat([_snippet("action(['DOWN', 'RIGHT'])")])
    bridge = _bridge(LINGJING_DUCK="on", LINGJING_DUCK_TURNS="2",
                     LINGJING_DUCK_STEPS="8")
    bridge._chat = chat

    first = bridge.tick(FakeFrame(), 0, request=0)
    assert first is not None and first[0] == "ACTION2", first

    # 引擎这一步的回报：板子变了、还是同一关。
    moved = [[0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0]]
    second = bridge.tick(FakeFrame(moved), 0)
    assert second is not None and second[0] == "ACTION4", second
    assert bridge.actions == 2
    # 两个动作来自同一次模型调用：批处理不能被拆成两次问模型
    assert chat.calls == 1


def test_click_carries_row_and_col():
    chat = ScriptChat([_snippet(
        "action({'action': 'MOUSE', 'row': 2, 'col': 4})")])
    bridge = _bridge(LINGJING_DUCK="on")
    bridge._chat = chat
    nxt = bridge.tick(FakeFrame(), 0, request=0)
    assert nxt is not None and nxt[0] == "ACTION6"
    assert nxt[1]["row"] == 2 and nxt[1]["col"] == 4


# ------------------------------------------------------------------- 兜底


def test_model_failure_falls_back_without_raising():
    def boom(messages, tools=None):
        raise RuntimeError("endpoint down")

    bridge = _bridge(LINGJING_DUCK="on")
    bridge._chat = boom
    assert bridge.tick(FakeFrame(), 0, request=0) is None
    assert bridge.enabled is False
    assert "finished" in bridge.reason or "start" in bridge.reason
    # 放弃之后每 tick 都是零开销
    assert bridge.tick(FakeFrame(), 0) is None


def test_budget_guard_stops_before_the_call():
    chat = ScriptChat([_snippet("action('DOWN')")])
    bridge = _bridge(LINGJING_DUCK="on")
    bridge._chat = chat
    bridge._budget = 0
    assert bridge.tick(FakeFrame(), 0, request=0) is None
    # 预算为 0 时闸门在调用前就关掉，脚本里的动作没被消耗
    assert chat.calls == 0


# ------------------------------------------------------- my_agent 侧接线


def _agent(game_id="zz99"):
    from my_agent import MyAgent

    return MyAgent(card_id="t", game_id=game_id, agent_name="t",
                   ROOT_URL="", record=False, arc_env=None)


def test_agent_never_routes_protected_games_to_duck():
    agent = _agent("ar25")            # 校准 plan 局：已有解，别拿模型去换
    agent._duck_request = 0
    assert agent._duck_tick(FakeFrame(), "ar25") is None
    assert agent._duck is None        # 连对象都没构造


def test_agent_without_request_does_nothing():
    agent = _agent()
    assert agent._duck_request == -1
    assert agent._duck_tick(FakeFrame(), "zz99") is None
    assert agent._duck is None


def test_agent_turns_bridge_output_into_a_game_action():
    from arcengine import GameAction

    agent = _agent()

    class FakeBridge:
        enabled = True

        def __init__(self):
            self.seen = []

        def tick(self, frame, levels, request=None):
            self.seen.append((levels, request))
            return ("ACTION6", {"row": 3, "col": 5})

        def stats(self):
            return {}

    bridge = FakeBridge()
    agent._duck = bridge
    agent._duck_request = 0

    action = agent._duck_tick(FakeFrame(), "zz99")
    assert action is not None
    assert str(getattr(action, "name", "")) == GameAction.ACTION6.name
    assert action.action_data.x == 5 and action.action_data.y == 3   # 引擎要 x=col / y=row
    assert bridge.seen == [(0, 0)]


def test_agent_clears_request_when_bridge_gives_up():
    agent = _agent()

    class DeadBridge:
        enabled = False

        def tick(self, frame, levels, request=None):
            return None

        def stats(self):
            return {}

    agent._duck = DeadBridge()
    agent._duck_request = 0
    assert agent._duck_tick(FakeFrame(), "zz99") is None
    assert agent._duck_request == -1     # 本局不再起它


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
