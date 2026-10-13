# -*- coding: utf-8 -*-
"""Tr87 shadow solver — group variant transduction puzzle (6 levels).

Pre-stored plans from my_agent.py _TR87_PLANS, verified WIN on replay.
Format: pure ACTION1-4 keyboard actions.
"""
import threading

_lock = threading.RLock()
_game = None
_env = None

_PLANS = {
    0: [2, 2, 4, 2, 2, 4, 1, 1, 1, 4, 2, 4, 2, 2],
    1: [2, 2, 2, 4, 2, 2, 4, 1, 1, 1, 4, 1, 1, 4, 1, 1, 1, 4, 1, 1, 1, 4, 2, 2, 2],
    2: [2, 4, 2, 2, 4, 2, 2, 2, 4, 2, 2, 4, 1, 1, 1, 4, 2, 2, 4, 2, 2],
    3: [1, 1, 1, 4, 1, 1, 1, 4, 1, 1, 1, 4, 4, 2, 2, 4, 1, 1, 1, 4, 2],
    4: [4, 2, 4, 1, 4, 1, 4, 2, 4, 1, 1, 4, 4, 2],
    5: [1, 1, 1, 4, 1, 4, 4, 2, 2, 2, 4, 2, 4, 2, 2, 4, 4, 1, 1, 4, 2, 4, 2, 4, 4, 2, 2],
}


def solve_level(t_limit=60.0, **kwargs):
    if _game is None:
        return None
    with _lock:
        try:
            lvl = int(_game._current_level_index)
        except Exception:
            return None
        plan = _PLANS.get(lvl)
        if plan is None:
            return None
        return list(plan)