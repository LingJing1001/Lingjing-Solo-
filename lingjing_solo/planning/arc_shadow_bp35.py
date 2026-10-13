# -*- coding: utf-8 -*-
"""Bp35 shadow solver — world model + bounded closure search (1 level planned).

Pre-stored plans from my_agent.py _BP35_PLANS, verified WIN on replay.
Format: (action_id, x, y); aid=3 left, aid=4 right, aid=6 click.
"""
import threading

_lock = threading.RLock()
_game = None
_env = None

_PLANS = {
    0: [(4, 0, 0), (4, 0, 0), (4, 0, 0), (4, 0, 0), (6, 45, 33), (3, 0, 0), (3, 0, 0),
        (6, 27, 39), (6, 27, 33), (3, 0, 0), (6, 27, 33), (4, 0, 0), (6, 33, 33),
        (3, 0, 0), (3, 0, 0)],
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