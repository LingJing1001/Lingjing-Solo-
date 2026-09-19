"""G2 复盘探针：L3 为什么没能靠「绕吃补给」(Ls20Solver._refill_detour) 活下来。

用法：
    python probe_ls20_g2_refill.py [recording.jsonl]

按 recording 里记的动作序列在干净引擎上重放，但**同时照常调用 solver.plan()**
（和 run_ls20_r2r3.py 的 _solver_action 一条路径，含 agent.py 的两处补救），
这样才能看到真实的 _phase / mod_task_idx / refill_target / queue，
再逐 tick 打印 _refill_detour 的全部判据（剩余可动作数、这一段要花几步、
画面里有几个补给格）。plan() 的返回值与 recording 动作不一致时打 DIFF。

注意：script_for_level 被钉成 None，否则全程 phase=script，在线分支根本不可达。
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # arc_adaptor
from paths import add_to_sys_path, environments_dir, state_dir  # noqa: E402

add_to_sys_path()

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402

import r2_ls20 as r2  # noqa: E402
import lingjing_solo.planning.ls20_solver as solver_mod  # noqa: E402
from lingjing_solo.planning.ls20_solver import (  # noqa: E402
    Ls20Solver, STEP, find_refill_cells, read_step_bar,
)

# 与 G2 闸门同一前提：关掉离线罐头，逼 Ls20Solver 走在线 _queue_next/_refill_detour 分支
solver_mod.script_for_level = lambda *a, **k: None
if "--fix-color-residue" in sys.argv:
    # 复现 run_ls20_r2r3.py --no-scripts --fix-color-residue 那一轮：颜色台加可达性过滤
    from run_ls20_r2r3 import _install_color_residue_fix  # noqa: E402
    _install_color_residue_fix()
    print("已打桩：_find_color_pad 先按 _reachable_from 过滤（同 rot/shape）")

ACT = {n: getattr(GameAction, f"ACTION{n}") for n in (1, 2, 3, 4)}
LEGAL = [f"ACTION{n}" for n in (1, 2, 3, 4)]


def default_recording() -> Path:
    cands = sorted(state_dir().glob("ls20_r2r3_noscript_*/recording.jsonl"))
    if not cands:
        raise SystemExit("找不到 state/ls20_r2r3_noscript_*/recording.jsonl，请先跑 G2 闸门")
    return cands[-1]


def solver_action(solver: Ls20Solver, grid):
    """复刻 R3Planner._solver_action（含 agent.py:234 的 reset_level 与 layout_wait 补救）。"""
    act = solver.plan(LEGAL)
    if act is None and not solver._await_level_layout \
            and not solver._layout_wait and not solver.active:
        solver.reset_level(grid)
        act = solver.plan(LEGAL)
    if act is None and solver._layout_wait:
        return "r3_layout_wait", "ACTION1"
    return ("r3_ls20_solver", act) if act else (None, None)


def main() -> int:
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(positional[0]) if positional else default_recording()
    lines = [json.loads(raw) for raw in path.open(encoding="utf-8") if raw.strip()]
    actions = [r["data"]["requested_action"]["name"] for r in lines[1:]]
    recorded_phase = [ (r.get("plan") or {}).get("solver_phase") for r in lines[1:] ]
    watch = max(int(r["data"]["levels_completed"]) for r in lines[1:])
    print(f"recording = {path}")
    print(f"重放 {len(actions)} 步，重点看 L{watch + 1}（levels_completed={watch}）")

    arcade = Arcade(environments_dir=environments_dir("ls20"), operation_mode=OperationMode.OFFLINE)
    gid = [e for e in arcade.get_environments() if e.game_id.startswith("ls20")][0].game_id
    env = arcade.make(gid)
    solver = Ls20Solver()

    frame = env.reset()
    grid = r2.to_grid(frame)
    solver.observe(None, grid, None, int(env._game._current_level_index))
    prev_grid, prev_act = grid, None
    printed_entry = False
    diffs = 0

    for tick, name in enumerate(actions, 1):
        frame = env.step(ACT[int(name[-1])], data=None)
        grid = r2.to_grid(frame)
        lvl = int(env._game._current_level_index)
        solver.observe(prev_grid, grid, prev_act, lvl)
        planner, proposed = solver_action(solver, grid)
        prev_grid, prev_act = grid, name
        if lvl != watch:
            continue
        if not printed_entry:
            print(f"\n进 L{watch + 1}: tick={tick} 玩家={solver.player_xy} "
                  f"目标={solver.goals} 修饰任务={solver.mod_tasks} 补给格="
                  f"{len(find_refill_cells(grid, (solver.player_xy[0] % STEP, solver.player_xy[1] % STEP)))}")
            printed_entry = True
        px = solver.player_xy
        residue = (px[0] % STEP, px[1] % STEP) if px else (0, 0)
        refills = find_refill_cells(grid, residue)
        bar = read_step_bar(grid)
        budget = solver._moves_left(grid)
        leg = solver._next_leg_cost(grid)
        detour = solver._refill_detour(grid)
        cur = (solver.mod_tasks[solver.mod_task_idx]
               if solver.mod_task_idx < len(solver.mod_tasks) else None)
        prog = {"rot": solver.pad_entries, "shape": solver.shape_toggles,
                "color": solver.color_toggles}.get(cur[0] if cur else "", None)
        diff = ""
        if proposed != name:
            diffs += 1
            diff = f"  <<DIFF 罐头记的是{name}"
        print(f"  t{tick:>3} 实际={name} 建议={proposed}({planner}) 玩家={str(px):<9} "
              f"HUD条={str(bar):<9} 可动作数={budget} 这一段={leg} 补给={len(refills)} "
              f"绕行={detour} 补给目标={solver.refill_target} phase={solver._phase} "
              f"mod={solver.mod_task_idx}/{len(solver.mod_tasks)} 任务={cur} 已踩={prog} "
              f"queue={list(solver.queue)[:3]} stall={solver.stall} "
              f"active={solver.active} 目标={solver._current_goal()} goals={solver.goals} "
              f"wait={solver._layout_wait}/{solver._await_level_layout} "
              f"记录phase={recorded_phase[tick - 1]}{diff}")
    planner, proposed = solver_action(solver, prev_grid)
    px = solver.player_xy
    print(f"\n最后一次询问(t{len(actions) + 1}): 建议={proposed}({planner}) 玩家={px} "
          f"phase={solver._phase} mod={solver.mod_task_idx}/{len(solver.mod_tasks)} "
          f"任务={(solver.mod_tasks[solver.mod_task_idx] if solver.mod_task_idx < len(solver.mod_tasks) else None)} "
          f"rot={solver.pad_entries} color={solver.color_toggles} "
          f"可动作数={solver._moves_left(prev_grid)} 这一段={solver._next_leg_cost(prev_grid)} "
          f"绕行={solver._refill_detour(prev_grid)} 补给目标={solver.refill_target} "
          f"active={solver.active} 目标={solver._current_goal()} goals={solver.goals} "
          f"wait={solver._layout_wait}/{solver._await_level_layout} stall={solver.stall}")
    print(f"\n重放结束: lvl={lvl} state={env._game._state} "
          f"steps_left={getattr(env._game._step_counter_ui, 'current_steps', None)} DIFF={diffs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
