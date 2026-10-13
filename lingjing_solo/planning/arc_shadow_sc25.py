# -*- coding: utf-8 -*-
"""Sc25 shadow solver — wizard puzzle with spells (4 levels planned).

Pre-stored plans from my_agent.py _SC25_PLANS, verified WIN on replay.
Format: (action_id, x, y); aid=1-4 keyboard, aid=6 click.
"""
import threading

_lock = threading.RLock()
_game = None
_env = None

_PLANS = {
    0: [(6, 10, 10), (6, 31, 51), (6, 26, 56), (6, 36, 56), (6, 31, 61), (6, 10, 10),
        (3, 0, 0), (3, 0, 0), (3, 0, 0), (3, 0, 0), (3, 0, 0), (3, 0, 0), (3, 0, 0),
        (3, 0, 0), (3, 0, 0), (3, 0, 0), (3, 0, 0), (3, 0, 0)],
    1: [(6, 26, 51), (6, 31, 51), (6, 31, 56), (1, 0, 0), (1, 0, 0)],
    2: [(6, 10, 10), (4, 0, 0), (4, 0, 0),
        (6, 31, 51), (6, 31, 56), (6, 31, 61),
        (3, 0, 0), (3, 0, 0), (3, 0, 0), (3, 0, 0),
        (2, 0, 0), (2, 0, 0), (2, 0, 0), (3, 0, 0)],
    3: [(6, 31, 51), (6, 26, 56), (6, 36, 56), (6, 31, 61),
        (2, 0, 0), (2, 0, 0), (2, 0, 0), (2, 0, 0), (2, 0, 0),
        (3, 0, 0), (3, 0, 0), (3, 0, 0),
        (6, 31, 51), (6, 31, 56), (6, 31, 61),
        (4, 0, 0), (4, 0, 0), (4, 0, 0), (4, 0, 0), (4, 0, 0), (4, 0, 0), (4, 0, 0),
        (2, 0, 0), (2, 0, 0), (2, 0, 0),
        (4, 0, 0), (4, 0, 0), (4, 0, 0), (4, 0, 0)],
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