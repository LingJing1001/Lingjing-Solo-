"""AR25 通用求解器 — 不依赖官方引擎的纯 Python AR25 求解工具包。

架构:
    1. AR25Simulator — 独立状态模拟器 (从 ar25.py 逆向规则)
    2. 启发函数 — 镜像对感知 + 全局轴代价 + 覆盖盈余惩罚
    3. 双策略搜索 — A* (简单关卡) + 宏动作搜索 (复杂关卡)

Public API:
    load_levels(path)        → 加载关卡数据
    solve(level, strategy)   → 求解单个关卡
    solve_all(levels)        → 求解所有关卡
    verify(level, actions)   → 验证解法
    AR25Simulator            → 状态模拟器类

Example:
    >>> from ar25_solver import load_levels, solve
    >>> levels = load_levels()
    >>> actions = solve(levels[0], strategy="auto")
    >>> from ar25_solver import verify
    >>> verify(levels[0], actions)
    {'won': True, 'steps': 15, ...}

CLI:
    python -m ar25_solver                      # 求解所有关卡
    python -m ar25_solver --level 3            # 求解L3
    python -m ar25_solver --strategy macro     # 指定宏搜索
    python -m ar25_solver --save out.json      # 保存解法
"""
from .simulator import (
    AR25Simulator,
    ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT, ACT_SWITCH,
    BOARD_SIZE, MAX_CASCADE_DEPTH,
    AXIS_TAG, AXIS_TAG_VERTICAL, AXIS_TAG_HORIZONTAL,
    IMMOVABLE_TAG, MOVABLE_TAG,
)
from .solver import (
    load_levels,
    solve,
    solve_all,
    solve_astar,
    solve_macro,
    verify,
)
from .advisor import (
    Advisor,
    Reflection,
    SearchSnapshot,
)

__version__ = "1.1.0"
__all__ = [
    "AR25Simulator",
    "Advisor",
    "Reflection",
    "SearchSnapshot",
    "load_levels",
    "solve",
    "solve_all",
    "solve_astar",
    "solve_macro",
    "verify",
    "ACT_UP", "ACT_DOWN", "ACT_LEFT", "ACT_RIGHT", "ACT_SWITCH",
    "BOARD_SIZE", "MAX_CASCADE_DEPTH",
    "AXIS_TAG", "AXIS_TAG_VERTICAL", "AXIS_TAG_HORIZONTAL",
    "IMMOVABLE_TAG", "MOVABLE_TAG",
    "__version__",
]
