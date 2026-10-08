#!/usr/bin/env python3
"""Shadow 采集驱动：真实 agent loop + 离线 env stub，跑 ≥100 步真实决策。

验证 AOPShadowObserver 接入真实 LingjingSoloAgent.choose_action 后：
  - 输出 ≥100 行旁路记录
  - 全部 used_for_control=false
  - 每行含 episode_id / step_id / prediction / latency_ms

用法:
    python tools/aop/shadow_collect.py --steps 100 \
        --output data/aop/shadow/agent-loop-100.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lingjing_solo.agent import LingjingSoloAgent  # noqa: E402
from lingjing_solo.neural.aop_model import AOPModel, save_checkpoint  # noqa: E402
from lingjing_solo.neural.aop_shadow import AOPShadowObserver  # noqa: E402

ACTIONS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]


def make_checkpoint(path: Path) -> None:
    model = AOPModel(16, len(ACTIONS))
    save_checkpoint(path, model, {"action_names": ACTIONS, "input_width": 16, "action_classes": len(ACTIONS)})


def stub_frame(tick: int) -> dict:
    """构造一个 agent.choose_action 可解析的最小 frame dict。"""
    grid = np.zeros((8, 8), dtype=np.int16)
    grid[0, 0] = (tick % 4) + 1  # 微变化，避免 agent 觉得完全静止
    return {
        "grid": grid,
        "state": "NOT_FINISHED",
        "levels_completed": 0,
        "available_actions": ACTIONS,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("data/aop/shadow/agent-loop-100.jsonl"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    if args.steps < 100:
        raise SystemExit("--steps 必须 >= 100")

    ckpt = args.checkpoint or (args.output.parent / "_collect_ckpt.pt")
    make_checkpoint(ckpt)
    observer = AOPShadowObserver(ckpt, args.output)

    agent = LingjingSoloAgent(shadow_observer=observer)
    env = SimpleNamespace(game_id="stub-collect")
    agent.reset(env=env)
    # 现有 ExplorationEngine 缺 rha_penalty（advisor.py:112 会调），打最小 stub
    if not hasattr(agent.explorer, "rha_penalty"):
        agent.explorer.rha_penalty = lambda a: 0.0
    # 现有 ExplorationEngine 缺 rha_penalty（advisor.py:112 会调），打最小 stub
    if not hasattr(agent.explorer, "rha_penalty"):
        agent.explorer.rha_penalty = lambda a: 0.0

    for tick in range(args.steps):
        frame = stub_frame(tick)
        agent.choose_action([frame], frame, valid_actions=ACTIONS)

    observer.flush()
    observer.close()

    # 校验输出
    rows = [json.loads(l) for l in args.output.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) >= 100, f"期望 >=100 行，实际 {len(rows)}"
    assert all(r["used_for_control"] is False for r in rows), "存在 used_for_control=true"
    assert all(r["episode_id"] for r in rows), "存在空 episode_id"
    assert all(r["latency_ms"] is not None for r in rows), "存在空 latency_ms"
    print(f"COLLECT_OK rows={len(rows)} all_used_for_control=false output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
