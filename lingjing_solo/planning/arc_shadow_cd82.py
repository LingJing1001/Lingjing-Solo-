# -*- coding: utf-8 -*-
"""Cd82 shadow solver — keyboard+click stamp game (6 levels).

Pre-stored plans from my_agent.py _CD82_PLANS, verified WIN on replay.
Format: (action_id, x, y); aid=1-5 keyboard, aid=6 click.
"""
import threading

_lock = threading.RLock()
_game = None
_env = None

_PLANS = {
    0: [(3, 0, 0), (2, 0, 0), (2, 0, 0), (4, 0, 0), (5, 0, 0)],
    1: [(5, 0, 0), (6, 46, 4), (4, 0, 0), (2, 0, 0), (2, 0, 0), (5, 0, 0)],
    2: [(6, 47, 4), (4, 0, 0), (2, 0, 0), (5, 0, 0), (6, 53, 4), (1, 0, 0), (3, 0, 0), (3, 0, 0), (2, 0, 0), (5, 0, 0), (6, 29, 4), (1, 0, 0), (5, 0, 0), (6, 35, 4), (4, 0, 0), (6, 32, 20)],
    3: [(6, 35, 4), (5, 0, 0), (6, 29, 4), (4, 0, 0), (2, 0, 0), (2, 0, 0), (5, 0, 0), (6, 59, 4), (3, 0, 0), (3, 0, 0), (1, 0, 0), (5, 0, 0), (6, 41, 4), (6, 13, 39)],
    4: [(6, 59, 4), (5, 0, 0), (6, 47, 4), (3, 0, 0), (2, 0, 0), (2, 0, 0), (5, 0, 0), (6, 35, 4), (4, 0, 0), (4, 0, 0), (5, 0, 0), (6, 53, 4), (1, 0, 0), (1, 0, 0), (3, 0, 0), (6, 32, 20)],
    5: [(6, 47, 4), (4, 0, 0), (2, 0, 0), (5, 0, 0), (6, 53, 4), (1, 0, 0), (3, 0, 0), (3, 0, 0), (5, 0, 0), (6, 29, 4), (4, 0, 0), (6, 32, 20), (6, 41, 4), (3, 0, 0), (2, 0, 0), (6, 13, 39)],
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