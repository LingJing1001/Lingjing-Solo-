# -*- coding: utf-8 -*-
"""Generic shadow solver template for games without pre-stored plans.

Uses r3_generic_search from lingjing_solo.planning.search.generic_shadow
for online BFS/A* state-space search. Falls back to None on failure.
"""
import threading
import time

_lock = threading.RLock()
_game = None
_env = None
_AVAILABLE_ACTIONS = []
_CLICK_PROPOSER = None


def _inject_env(env_ref):
    global _env, _game
    _env = env_ref
    _game = getattr(env_ref, "_game", None)


def solve_level(t_limit=60.0, **kwargs):
    if _game is None or _env is None:
        return None
    with _lock:
        try:
            from arcengine import GameAction, ActionInput, GameState
            from lingjing_solo.planning.search.generic_shadow import r3_generic_search

            act_map = {}
            for n in _AVAILABLE_ACTIONS:
                name = f"ACTION{n}"
                if hasattr(GameAction, name):
                    act_map[n] = getattr(GameAction, name)
            if not act_map:
                return None

            def make_action(aid):
                return ActionInput(id=aid, data={}, reasoning=None)

            path = r3_generic_search(
                _env,
                t_limit=t_limit,
                max_nodes=20000,
                max_depth=80,
                act_map=act_map,
                make_action=make_action,
                game_over_state=GameState.GAME_OVER,
                win_state=GameState.WIN,
            )
            return path
        except Exception:
            return None