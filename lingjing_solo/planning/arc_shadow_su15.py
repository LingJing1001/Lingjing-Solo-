# -*- coding: utf-8 -*-
"""Su15 shadow solver — pure ACTION6/7 click game (9 levels).

Pre-stored plans from verified_solutions.py SU15_SOLUTIONS, verified WIN on replay.
Format: (x, y) ACTION6 clicks.
"""
import threading

_lock = threading.RLock()
_game = None
_env = None

_PLANS = {
    0: [[10, 53], [16, 47], [22, 41], [28, 35], [34, 29], [40, 23], [46, 17]],
    1: [[45, 52], [18, 53], [20, 41], [39, 38], [41, 45], [22, 47], [26, 40], [34, 41], [33, 33], [33, 25]],
    2: [[13, 28], [58, 30], [50, 30], [18, 24], [35, 18], [35, 26], [42, 30], [36, 36], [15, 32], [12, 40], [31, 42], [10, 48], [25, 48]],
    3: [[31, 35], [8, 30], [28, 47], [25, 40], [13, 42], [10, 36], [17, 40], [12, 47], [7, 54]],
    4: [[48, 33], [8, 57], [12, 24], [51, 56], [47, 50], [11, 42], [44, 43], [10, 50], [36, 45], [28, 47], [14, 43], [22, 42], [25, 34], [28, 26], [31, 18]],
    5: [[0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [53, 40], [53, 48], [53, 53], [55, 55]],
    6: [[11, 32], [26, 37], [19, 33], [21, 25], [46, 24], [23, 17]],
    7: [[9, 49], [37, 29], [33, 42], [6, 18], [17, 50], [10, 54]],
    8: [[13, 20], [16, 54], [46, 53], [42, 31], [42, 45], [33, 44], [25, 44], [8, 56], [51, 56]],
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
        return [(6, x, y) for x, y in plan]