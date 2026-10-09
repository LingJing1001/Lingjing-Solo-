"""优化 solver: 预计算路径，BFS 纯计算不调引擎。

状态 = (goal_pos, goal-o_pos)。状态转移纯计算:
  点按钮: 如果 goal/goal-o 在该路径上，数字 ±1 → 新位置。
不调 chmfaflqhy/ttawusezqc/set_position，速度快 1000x。
"""
import os, sys, importlib.util, time
from collections import deque
from itertools import product
from pathlib import Path

os.environ["OPERATION_MODE"] = "offline"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter" / "vendor"))

import arc_agi
from arc_agi import OperationMode

spec = importlib.util.spec_from_file_location("lp85mod", "environment_files/lp85/305b61c3/lp85.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
SCALE = mod.crxpafuiwp


def build_path_index(game):
    """预计算: 位置 → [(letter, num), ...] 和 letter → {num: pos}。"""
    maps = game.uopmnplcnv[game.ucybisahh]
    pos_to_letters = {}  # (x,y) → [(letter, num)]
    letter_to_pos = {}   # letter → {num: (x,y)}
    for letter, data in maps.items():
        L = data["oxbwsencfv"]
        if L <= 1:
            continue
        letter_to_pos[letter] = {}
        for num, pos in data["qcmzcjocmj"].items():
            px, py = pos.x * SCALE, pos.y * SCALE
            letter_to_pos[letter][num] = (px, py)
            pos_to_letters.setdefault((px, py), []).append((letter, num))
    return pos_to_letters, letter_to_pos


def get_targets(game):
    """过关目标: (goal_targets, goal-o_targets) 区分两类。"""
    goal_targets = []
    goal_o_targets = []
    for s in game.current_level._sprites:
        if s.tags and "bghvgbtwcb" in s.tags[0]:
            goal_targets.append((s.x + 1, s.y + 1))
        elif s.tags and "fdgmtkfrxl" in s.tags[0]:
            goal_o_targets.append((s.x + 1, s.y + 1))
    return (tuple(sorted(goal_targets)), tuple(sorted(goal_o_targets)))


def get_initial_goals(game):
    """初始 (goal_positions, goal-o_positions) 区分两类。"""
    goals = []
    goal_os = []
    for s in game.current_level._sprites:
        if s.tags:
            if s.tags[0] == "goal":
                goals.append((s.x, s.y))
            elif s.tags[0] == "goal-o":
                goal_os.append((s.x, s.y))
    return (tuple(sorted(goals)), tuple(sorted(goal_os)))


def solve_level_fast(game, time_limit=60):
    """纯计算 BFS: 状态 = goal 位置组合，不调引擎。"""
    t0 = time.time()
    pos_to_letters, letter_to_pos = build_path_index(game)
    targets = get_targets(game)
    init_goals = get_initial_goals(game)

    if not init_goals or not targets:
        return None

    # 按钮按位置分组（同位置多个按钮点一次同时触发）
    btn_groups = {}  # (x,y) → [(letter, direction), ...]
    for s in game.current_level._sprites:
        if s.tags and "button" in s.tags[0]:
            parts = s.tags[0].split("_")
            if len(parts) == 3:
                btn_groups.setdefault((s.x, s.y), []).append((parts[1], parts[2] == "R"))
    btn_list = list(btn_groups.values())

    # 目标状态
    target_state = targets  # (goal_targets, goal-o_targets)

    # 初始状态
    init_state = init_goals  # (goal_positions, goal-o_positions)

    if init_state == target_state:
        return []

    def apply_button(state, letter, direction):
        """纯计算: 点按钮后 goal/goal-o 位置变化。"""
        goal_pos, goal_o_pos = state
        new_goal = list(goal_pos)
        new_goal_o = list(goal_o_pos)
        for positions, new_list in [(goal_pos, new_goal), (goal_o_pos, new_goal_o)]:
            for i, pos in enumerate(positions):
                entries = pos_to_letters.get(pos, [])
                for l, num in entries:
                    if l == letter:
                        L = len(letter_to_pos[letter])
                        new_num = num + 1 if direction else num - 1
                        if new_num > L: new_num = 1
                        if new_num < 1: new_num = L
                        new_list[i] = letter_to_pos[letter][new_num]
                        break
        return (tuple(sorted(new_goal)), tuple(sorted(new_goal_o)))

    # BFS
    seen = {init_state}
    queue = deque([(init_state, [])])
    expansions = 0
    while queue and time.time() - t0 < time_limit:
        state, path = queue.popleft()
        for btn_group in btn_list:
            # 点一次触发该位置所有按钮字母
            new_state = state
            for letter, direction in btn_group:
                new_state = apply_button(new_state, letter, direction)
            expansions += 1
            if new_state == target_state:
                print(f"  fast BFS: expansions={expansions} states={len(seen)} time={time.time()-t0:.1f}s")
                return path + [btn_group]
            if new_state not in seen:
                seen.add(new_state)
                if len(path) < 80:
                    queue.append((new_state, path + [btn_group]))
    print(f"  fast BFS: expansions={expansions} states={len(seen)} (超时)")
    return None


def click(game, letter, direction):
    """调引擎执行按钮点击（用于验证）。"""
    moves = mod.chmfaflqhy(game.ucybisahh, letter, direction, game.uopmnplcnv)
    pending = []
    for frm, to in moves:
        sp = game.ttawusezqc(frm.x * SCALE, frm.y * SCALE)
        if sp: pending.append((sp, to.x * SCALE, to.y * SCALE))
    for sp, nx, ny in pending: sp.set_position(nx, ny)


def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    env = arc.make("lp85")
    env.reset()
    game = env._game

    total = 0
    for lvl in range(8):
        sc = game.current_level.get_data("StepCounter")
        print(f"\n=== Level {lvl} ===  StepCounter={sc}")
        t0 = time.time()
        solution = solve_level_fast(game, time_limit=120)
        if solution is None:
            print(f"  未搜出解"); break
        print(f"  解: {len(solution)} 步, 耗时 {time.time()-t0:.1f}s")
        for i, btn_group in enumerate(solution):
            letters = ", ".join(f"{l}_{'R' if d else 'L'}" for l, d in btn_group)
            if i < 10: print(f"    {i+1}. [{letters}]")
        if len(solution) > 10: print(f"    ... ({len(solution)-10} more)")
        # 验证 + 执行
        for i, btn_group in enumerate(solution):
            for letter, d in btn_group:
                click(game, letter, d)  # d 已经是 bool (True=R, False=L)
            if i < 3:
                goals = [(s.x,s.y) for s in game.current_level._sprites if s.tags and "goal" in s.tags[0]]
                print(f"    step {i+1} 后: goals={goals} win={game.khartslnwa()}")
        win = game.khartslnwa()
        print(f"  验证 win={win}")
        total += len(solution)
        if not win:
            print("  验证失败!"); break
        if str(game._state).upper() == "WIN":
            print("  *** 全部通关! ***"); break
        game.set_level(lvl + 1)
        if hasattr(game, "on_set_level"):
            game.on_set_level(game.current_level)
        print(f"  -> Level {lvl+1}")

    print(f"\n最终: level={game.level_index} state={game._state} total_steps={total}")


if __name__ == "__main__":
    main()
