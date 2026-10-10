"""用新代码（engine_solver + click_proposer）跑 lp85。"""
import os, sys, time
from pathlib import Path
import numpy as np

os.environ["OPERATION_MODE"] = "offline"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter" / "vendor"))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
from lingjing_solo.agent import LingjingSoloAgent
from lingjing_solo.engine_solver import EngineSolver

MAX_STEPS = 200

def run_lp85(use_engine_solver=False, use_click_proposer=False):
    arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    env = arc.make("lp85")
    frame = env.reset()

    engine_solver = EngineSolver(environments_dir="environment_files") if use_engine_solver else None
    click_proposer = None
    if use_click_proposer:
        try:
            from arc_adaptor.click_heatmap_pro2 import propose_clicks
            click_proposer = propose_clicks
        except Exception:
            pass

    agent = LingjingSoloAgent(engine_solver=engine_solver, click_proposer=click_proposer)
    agent.reset(env=env)
    # 手动设 game_id（env 可能没暴露 game_id 属性）
    agent.transfer.set_game_id("lp85")
    if engine_solver:
        engine_solver.reset_episode("lp85")
        # 手动测 get_action
        test_result = engine_solver.get_action("lp85")
        print(f"  [debug] engine_solver.get_action('lp85') = {test_result}", flush=True)
        engine_solver.reset_episode("lp85")  # 重置回 0
        # 手动测 get_action
        test_result = engine_solver.get_action("lp85")
        print(f"  [debug] engine_solver.get_action('lp85') = {test_result}", flush=True)
        engine_solver.reset_episode("lp85")  # 重置回 0
    if not hasattr(agent.explorer, "rha_penalty"):
        agent.explorer.rha_penalty = lambda a: 0.0

    frames = []
    t0 = time.time()
    for step in range(MAX_STEPS):
        valid_ids = list(frame.available_actions or [6])
        valid_names = [f"ACTION{i}" for i in valid_ids]
        frame_dict = {
            "grid": np.array(frame.frame[-1], dtype=np.int16),
            "state": str(frame.state).replace("GameState.", ""),
            "levels_completed": frame.levels_completed,
            "available_actions": valid_names,
        }
        action_str = agent.choose_action(frames, frame_dict, valid_actions=valid_names)
        if step < 3 and use_engine_solver:
            print(f"  step {step}: action={action_str} rationale={agent.last_rationale} click={agent.last_click}", flush=True)
        if step < 3 and use_engine_solver:
            print(f"  step {step}: action={action_str} rationale={agent.last_rationale} click={agent.last_click}", flush=True)
        aid = int(action_str.replace("ACTION", "")) if action_str.startswith("ACTION") else 6
        data = {}
        if aid == 6 and agent.last_click:
            data = {"x": int(agent.last_click[0]), "y": int(agent.last_click[1])}
        action = GameAction.from_id(aid)
        action.reasoning = {"text": agent.last_rationale}
        frames.append(frame_dict)
        try:
            frame = env.step(action, data=data, reasoning={"text": agent.last_rationale})
        except Exception as e:
            print(f"  step {step} err: {e}")
            break
        if not getattr(frame, "frame", None):
            # frame 空 → RESET
            action = GameAction.from_id(0)
            frame = env.step(action, data={}, reasoning={"text": "reset"})
            continue
        if str(frame.state).upper() == "WIN":
            break
    state = str(frame.state).replace("GameState.", "")
    return frame.levels_completed, step + 1, state, round(time.time() - t0, 1)


def main():
    print("=== 基线: 无 engine_solver 无 click_proposer ===")
    lv0, ac0, st0, t0 = run_lp85(use_engine_solver=False, use_click_proposer=False)
    print(f"  levels={lv0}  actions={ac0}  state={st0}  time={t0}s")

    print("\n=== 新代码: engine_solver（有引擎代码就读）===")
    lv1, ac1, st1, t1 = run_lp85(use_engine_solver=True, use_click_proposer=False)
    print(f"  levels={lv1}  actions={ac1}  state={st1}  time={t1}s")

    print("\n=== 新代码: engine_solver + click_proposer ===")
    lv2, ac2, st2, t2 = run_lp85(use_engine_solver=True, use_click_proposer=True)
    print(f"  levels={lv2}  actions={ac2}  state={st2}  time={t2}s")

    print("\n=== 对比 ===")
    print(f"  基线:                {lv0} 关 / {ac0} 步")
    print(f"  engine_solver:       {lv1} 关 / {ac1} 步")
    print(f"  engine+click_prop:   {lv2} 关 / {ac2} 步")


if __name__ == "__main__":
    main()
