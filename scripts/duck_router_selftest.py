# -*- coding: utf-8 -*-
"""证明 duck 真的接进了路由，而不是只接进了单元测试。

`tests/test_duck_bridge.py` 钉的是缝里的每一处判断，但它不碰引擎。这里补的是
只会在真引擎上才暴露的部分：

  1. `choose_action` 在 duck 命中时**提前 return**，动作没被 v14/AOP 改写；
  2. 交回来的动作能在真引擎上执行，并且被计分卡记数（duck 的合法性前提）；
  3. poll → feed → poll 在真实帧上走通：第二步决策看到的是动作之后的板子；
  4. 脚本用完之后 duck 自己退出，`_duck_request` 归 -1，路由回到原流程。

模型不参与：`ScriptChat` 按顺序吐 snippet，这和 `tests/test_duck.py` 验证非
可恢复循环时用的是同一套办法。

    ../../.venv312/Scripts/python.exe scripts/duck_router_selftest.py

游戏默认 `su15`：25 局里唯一既不在校准 plan 里、也不在 engine_solver 白名单里
的局 —— 于是 `choose_action` 前面那些"已知局抢占"的入口都不会替我们答题。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STARTER = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"
for p in (str(ROOT), str(STARTER), str(ROOT / "arc_adaptor"),
          str(STARTER / "vendor" / "ARC-AGI-3-Agents"), str(STARTER / "agent")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("LINGJING_SRC", str(ROOT / "lingjing_solo"))
os.environ["LINGJING_DUCK"] = "on"
os.environ.setdefault("LINGJING_LLM", "off")     # 不让 LLM 规划器抢在 duck 前面

from arc_agi import Arcade, OperationMode        # noqa: E402
from arcengine import GameAction                 # noqa: E402

from duck_bridge import DuckBridge               # noqa: E402
from my_agent import MyAgent                     # noqa: E402

ACTION_BY_NAME = {f"ACTION{i}": getattr(GameAction, f"ACTION{i}") for i in range(1, 8)}


class ScriptChat:
    def __init__(self, script):
        self.script = list(script)
        self.i = 0
        self.calls = 0

    def __call__(self, messages, tools=None):
        self.calls += 1
        if self.i >= len(self.script):
            return {"role": "assistant", "content": "", "tool_calls": []}
        code = self.script[self.i]
        self.i += 1
        return {"role": "assistant", "content": "", "tool_calls": [{
            "id": f"call_{self.i}", "type": "function",
            "function": {"name": "python", "arguments": json.dumps({"code": code})},
        }]}

    def tokens(self) -> int:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="su15")
    ap.add_argument("--actions", type=int, default=3)
    args = ap.parse_args()

    arc = Arcade(operation_mode=OperationMode.NORMAL,
                 environments_dir=str(STARTER / "environment_files"))
    env = arc.make(args.game, render_mode=None)
    frame = env.reset()

    os.environ["LINGJING_DUCK"] = "on"
    agent = MyAgent(card_id="selftest", game_id=args.game, agent_name="selftest",
                    ROOT_URL="", record=False, arc_env=None)
    chat = ScriptChat(["action('MOUSE@20,20')",
                       "action(['MOUSE@30,30', 'MOUSE@10,10'])"])
    agent._duck = DuckBridge(args.game, chat=chat, log=print)
    agent._duck_request = 0          # 假装 R3 已经在 L0 上失败

    ok = True
    spent = 0
    duck_actions = 0
    for i in range(args.actions):
        fd = agent._convert_raw_frame_data(frame)
        agent.frames = [fd]
        action = agent.choose_action([fd], fd)
        note = (action.reasoning or {}).get("text", "") if action.reasoning else ""
        if "duck" not in note:
            print(f"[router] step {i}: not duck ({action.name}, reasoning={note!r})")
            ok = False
            break
        duck_actions += 1
        data = {}
        if action.name == "ACTION6":
            data = {"x": int(action.action_data.x), "y": int(action.action_data.y)}
        frame = env.step(action, data=data, reasoning={"note": "duck router selftest"})
        spent += 1
        print(f"[router] step {i}: {action.name} -> level={frame.levels_completed} "
              f"state={getattr(frame.state, 'name', frame.state)}")

    # duck 的脚本用完了：下一次它应该自己退出，把控制权交回原流程。
    fd = agent._convert_raw_frame_data(frame)
    agent.frames = [fd]
    fallback = agent.choose_action([fd], fd)
    note = (fallback.reasoning or {}).get("text", "") if fallback.reasoning else ""
    if agent._duck_request != -1:
        print(f"[router] FAIL: duck exhausted but _duck_request={agent._duck_request}")
        ok = False
    if "duck" in note:
        print(f"[router] FAIL: still routing to a dead duck ({note!r})")
        ok = False
    print(f"[router] after duck: {fallback.name} reasoning={note!r}")

    try:
        cards = list(getattr(env.scorecard_manager, "scorecards", {}).values())
        counted = sum(int(getattr(c, "total_actions", 0) or 0) for c in cards)
    except Exception as exc:  # noqa: BLE001
        print(f"[router] could not read scorecard: {type(exc).__name__}: {exc}")
        counted = -1
        ok = False
    print(f"[router] harness spent={spent} duck_actions={duck_actions} "
          f"scorecard_counted={counted} chat_calls={chat.calls}")

    if duck_actions == 0:
        print("[router] FAIL: duck never took a turn")
        ok = False
    if counted >= 0 and counted < spent:
        print(f"[router] FAIL: scorecard counted {counted} < spent {spent}")
        ok = False

    print("[router] " + ("PASS" if ok else "FAIL"))
    try:
        env.close()
    except AttributeError:          # LocalEnvironmentWrapper has no close()
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
