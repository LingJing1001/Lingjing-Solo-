"""优化 BFS: 利用字母独立性做笛卡尔积搜索。

关键洞察: 每个字母(A/B/C)控制独立的环形路径，互不影响。
  点 n 次 R 等价于点 (L-n) 次 L（L=循环长度）。
  所以每个字母只需搜 0..L-1 次，总空间 = ∏ L_i（如 20×10×10=2000）。
"""
import os, sys, importlib.util, time, hashlib
from itertools import product
from collections import deque
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


def snap(game):
    return tuple((s.x, s.y) for s in game.current_level._sprites)

def restore(game, s):
    for sp, (x, y) in zip(game.current_level._sprites, s):
        sp.set_position(x, y)

def click(game, letter, direction):
    """手动模拟点击按钮: 先收集所有 sprite 再一次性移动（避免链式推动）。"""
    moves = mod.chmfaflqhy(game.ucybisahh, letter, direction, game.uopmnplcnv)
    # 先收集所有要移动的 sprite 和目标位置
    pending = []
    for frm, to in moves:
        sp = game.ttawusezqc(frm.x * SCALE, frm.y * SCALE)
        if sp:
            pending.append((sp, to.x * SCALE, to.y * SCALE))
    # 一次性移动（避免链式推动导致 ttawusezqc 找到错误的 sprite）
    for sp, nx, ny in pending:
        sp.set_position(nx, ny)

def get_letters(game):
    """获取当前关的字母和循环长度。"""
    level_name = game.ucybisahh
    maps = game.uopmnplcnv
    if level_name not in maps:
        return []
    letters = []
    for letter, data in maps[level_name].items():
        L = data["oxbwsencfv"]  # 循环长度
        if L > 1:
            letters.append((letter, L))
    return letters


def solve_level_cartesian(game, time_limit=60):
    """笛卡尔积搜索: 每个字母独立搜 0..L-1 次 R，用差量组合。"""
    t0 = time.time()
    letters = get_letters(game)
    if not letters:
        return None
    print(f"  字母: {[(l, L) for l, L in letters]}")

    initial = snap(game)

    # 对每个字母，预计算点 0..L-1 次 R 后的差量（只含该字母影响的 sprite）
    letter_diffs = {}
    for letter, L in letters:
        restore(game, initial)
        diffs = []  # diffs[n] = [(sprite_idx, new_x, new_y), ...]
        for n in range(L):
            if n > 0:
                click(game, letter, True)  # 累积点 R
            current = snap(game)
            diff = [(i, current[i][0], current[i][1])
                    for i in range(len(current)) if current[i] != initial[i]]
            diffs.append(diff)
        letter_diffs[letter] = diffs

    # 笛卡尔积: 应用各字母的差量（不覆盖其他字母的 sprite）
    ranges = [range(L) for _, L in letters]
    total = 1
    for r in ranges:
        total *= len(r)
    print(f"  搜索空间: {total}")

    best = None
    for combo in product(*ranges):
        if time.time() - t0 > time_limit:
            print(f"  超时 ({time_limit}s)")
            break
        # 从 initial 出发，应用各字母的差量
        restore(game, initial)
        for i, (letter, L) in enumerate(letters):
            n = combo[i]
            for sprite_idx, nx, ny in letter_diffs[letter][n]:
                game.current_level._sprites[sprite_idx].set_position(nx, ny)
        if game.khartslnwa():
            steps = sum(combo)
            if best is None or steps < sum(best):
                best = combo
                print(f"  找到解: {dict(zip([l for l, _ in letters], combo))} = {steps} 步")

    if best:
        # 展开为逐步动作（每个字母点 n 次 R）
        result = []
        for (letter, _), n in zip(letters, best):
            for _ in range(n):
                result.append((letter, "R", f"button_{letter}_R"))
        return result
    return None


def solve_level_bfs(game, max_depth=60, time_limit=300):
    """通用 BFS: 用 goal 位置做状态去重（状态空间 ≈ goal 位置组合）。"""
    t0 = time.time()
    btns = []
    for s in getattr(game, "afhycvvjg", []) or []:
        if s.tags and "button" in s.tags[0]:
            parts = s.tags[0].split("_")
            if len(parts) == 3:
                btns.append((parts[1], parts[2] == "R", s.tags[0]))
    if not btns:
        return None

    def goal_key(g):
        """状态 = goal 位置组合（bghvgbtwcb 不动，过关只看 goal 相对位置）。"""
        goals = tuple(sorted((s.x, s.y) for s in g.current_level._sprites
                             if s.tags and "goal" in s.tags[0]))
        return goals

    initial = snap(game)
    seen = {goal_key(game)}
    queue = deque([(initial, [])])
    expansions = 0
    while queue and time.time() - t0 < time_limit:
        state, path = queue.popleft()
        for letter, direction, tag in btns:
            restore(game, state)
            click(game, letter, direction)
            expansions += 1
            if game.khartslnwa():
                print(f"  BFS expansions={expansions} states={len(seen)}")
                return path + [(letter, "R" if direction else "L", tag)]
            gk = goal_key(game)
            if gk not in seen:
                seen.add(gk)
                if len(path) < max_depth:
                    queue.append((snap(game), path + [(letter, "R" if direction else "L", tag)]))
    print(f"  BFS expansions={expansions} states={len(seen)} (超时/耗尽)")
    return None


def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    env = arc.make("lp85")
    env.reset()
    game = env._game

    total_steps = 0
    for lvl_idx in range(8):
        sc = game.current_level.get_data("StepCounter")
        letters = get_letters(game)
        print(f"\n=== Level {lvl_idx} ===  letters={len(letters)}  StepCounter={sc}")
        if not letters:
            print("  无字母"); break
        t0 = time.time()
        # 先试笛卡尔积（快），失败用 BFS 兜底
        solution = solve_level_cartesian(game, time_limit=30)
        if solution is None:
            print("  笛卡尔积未找到，试 BFS（300s）...")
            solution = solve_level_bfs(game, max_depth=sc or 30, time_limit=300)
        if solution is None:
            print(f"  未搜出解"); break
        print(f"  解: {len(solution)} 步, 耗时 {time.time()-t0:.1f}s")
        for i, (letter, d, _) in enumerate(solution):
            print(f"    {i+1}. {letter}_{d}")
        # 执行解法
        for letter, d, _ in solution:
            click(game, letter, d == "R")
        total_steps += len(solution)
        if str(game._state).upper() == "WIN":
            print("  *** 全部通关! ***"); break
        # 切换关卡
        if hasattr(game, "set_level"):
            game.set_level(lvl_idx + 1)
        else:
            game.next_level()
        if hasattr(game, "on_set_level"):
            game.on_set_level(game.current_level)
        print(f"  -> 进入 Level {lvl_idx+1}")

    print(f"\n最终: level={game.level_index} state={game._state} total_steps={total_steps}")


if __name__ == "__main__":
    main()
