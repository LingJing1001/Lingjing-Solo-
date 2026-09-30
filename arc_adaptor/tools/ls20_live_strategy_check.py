"""用 ARC 侧**生产策略类**驱动离线引擎，检查 Lingjing 侧 solver 改动有没有打回线上 LS20。

为什么需要它：`Ls20Solver` 在 ARC checkout 里是线上真正 import 的那个模块
（`agents/strategies/ls20.py:8 from lingjing_solo.planning.ls20_solver import Ls20Solver`），
而 Lingjing 侧的 editable 安装把它直接指向本仓库
（`.venv/Lib/site-packages/_editable_impl_lingjing_solo.pth` → 本仓库 `lingjing_solo/`）。
所以离线 runner 跑通不代表线上路径跑通：这里刻意**不经过**任何自研兜底，
只按官方适配层的调用顺序喂 `LS20Strategy`，成绩差就是 solver 本身的差。

用法（在 ARC checkout 内、用该 checkout 的 venv）：
    ./.venv/Scripts/python.exe <Lingjing>/arc_adaptor/tools/ls20_live_strategy_check.py
环境变量：
    LINGJING_ARC_HOME   官方 ARC-AGI-3-Agents checkout 根（缺省时取本仓库同级的同名目录）
    --max-steps N       步数上限（默认 600）
    --tag LABEL         报告里标注这轮跑的是哪个版本的 solver（对照 HEAD / 工作树时用）
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # arc_adaptor
from paths import PROJECT_ROOT, add_to_sys_path, environments_dir  # noqa: E402

add_to_sys_path()

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

STRATEGY_REL = Path("agents") / "strategies" / "ls20.py"
ACT_BY_NAME = {f"ACTION{n}": getattr(GameAction, f"ACTION{n}") for n in (1, 2, 3, 4)}


def arc_root() -> Path:
    """官方 checkout 根：环境变量优先，否则本仓库同级同名目录；解析不到就 fail-closed。"""
    env = os.environ.get("LINGJING_ARC_HOME", "").strip()
    cands = ([Path(env).expanduser()] if env else []) + [PROJECT_ROOT.parent / "ARC-AGI-3-Agents"]
    for cand in cands:
        if (cand / STRATEGY_REL).is_file():
            return cand.resolve()
    raise LookupError(
        "找不到 ARC 生产策略模块 " + STRATEGY_REL.as_posix() + "，已尝试:\n  "
        + "\n  ".join(str(c) for c in cands)
        + "\n请设 LINGJING_ARC_HOME=<官方 ARC-AGI-3-Agents checkout 根>"
    )


def load_live_strategy(root: Path):
    """按文件路径加载生产策略模块，绕开 `agents/__init__.py`（它会拉起官方全套 agent）。"""
    path = root / STRATEGY_REL
    spec = importlib.util.spec_from_file_location("lingjing_live_ls20_strategy", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def solver_source() -> Path | None:
    try:
        from lingjing_solo.planning import ls20_solver
    except Exception:
        return None
    return Path(ls20_solver.__file__).resolve()


def main() -> int:
    max_steps = 600
    tag = "worktree"
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--max-steps" and i + 1 < len(args):
            max_steps = int(args[i + 1])
        elif a == "--tag" and i + 1 < len(args):
            tag = args[i + 1]

    root = arc_root()
    live = load_live_strategy(root)
    src = solver_source()
    print(f"arc_root        = {root}")
    print(f"solver_loaded   = {src}")
    print(f"tag             = {tag}")

    arcade = Arcade(environments_dir=environments_dir("ls20"),
                    operation_mode=OperationMode.OFFLINE)
    gid = [e.game_id for e in arcade.get_environments() if e.game_id.startswith("ls20")][0]
    env = arcade.make(gid)
    frame = env.reset()

    strategy = live.LS20Strategy()
    strategy.reset(frame)
    grid = live._frame_grid(frame)
    levels = int(getattr(frame, "levels_completed", 0) or 0)
    level_steps: dict[int, int] = {levels: 0}
    step = 0
    none_streak = 0
    started = time.monotonic()

    while step < max_steps and frame.state not in (GameState.WIN, GameState.GAME_OVER):
        legal = [f"ACTION{n}" for n in sorted(getattr(frame, "available_actions", []) or [])
                 if f"ACTION{n}" in ACT_BY_NAME]
        if not legal:
            print(f"{step:4d} 没有可用动作 → 停止 (state={frame.state})")
            break
        chosen = strategy.choose_action([frame], frame, grid, legal, levels)
        if chosen is None:
            none_streak += 1
            if none_streak >= 10:
                print(f"{step:4d} 策略连续 {none_streak} 次返回 None → 停止 "
                      f"(state={frame.state} levels={levels})")
                break
            chosen = "ACTION1" if "ACTION1" in legal else legal[0]   # 线上兜底：legal[0]
            print(f"{step:4d} 策略返回 None，本轮临时用 {chosen}")
            continue
        if chosen not in legal:
            print(f"{step:4d} 策略给出非法动作 {chosen}，legal={legal} → 停止")
            break
        none_streak = 0
        step += 1
        frame = env.step(ACT_BY_NAME[chosen], data=None)
        grid = live._frame_grid(frame)
        new_levels = int(getattr(frame, "levels_completed", 0) or 0)
        if new_levels != levels:
            print(f"{step:4d} L{levels + 1} → L{new_levels + 1}  (state={frame.state})")
            levels = new_levels
            level_steps.setdefault(levels, 0)
        level_steps[levels] = level_steps.get(levels, 0) + 1

    elapsed = time.monotonic() - started
    won = frame.state is GameState.WIN
    print("-" * 64)
    for lv in sorted(level_steps):
        print(f"  L{lv + 1}: {level_steps[lv]:4d} 步")
    print(f"RESULT tag={tag} state={frame.state.name} levels={levels}/7 "
          f"steps={step} won={won} elapsed={elapsed:.1f}s")
    print(f"verdict={'PASS' if won else 'FAIL'}")
    return 0 if won else 1


if __name__ == "__main__":
    raise SystemExit(main())
