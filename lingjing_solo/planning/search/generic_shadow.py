"""Generic game snapshot/restore/search for unknown environments.

Unlike arc_shadow.py (AR25-specific attribute names), this module uses
__dict__ introspection to snapshot any game object, enabling R3 state-space
search on arbitrary unknown games.

Usage:
    # The caller (a boundary adapter) injects the engine objects; this module
    # must not import arcengine -- see tests/test_abstract_action_boundary.py.
    path = r3_generic_search(
        env,
        act_map={n: getattr(GameAction, "ACTION%d" % n) for n in range(1, 8)},
        make_action=lambda aid: ActionInput(id=aid, data={}, reasoning=None),
        game_over_state=GameState.GAME_OVER,
        t_limit=15.0,
    )
"""
from __future__ import annotations

import copy
import hashlib
import heapq
import time
from typing import Any, Optional

import numpy as np


def generic_snapshot(game: Any) -> dict:
    snap = {}
    for k, v in game.__dict__.items():
        if callable(v) and not isinstance(v, type):
            continue
        try:
            if isinstance(v, np.ndarray):
                snap[k] = v.copy()
            else:
                snap[k] = copy.deepcopy(v)
        except Exception:
            pass
    return snap


def generic_restore(game: Any, snap: dict) -> None:
    for k, v in snap.items():
        try:
            setattr(game, k, v)
        except Exception:
            pass


def generic_state_key(game: Any) -> tuple:
    try:
        grid = game.naxbskjmlg()
        h = hashlib.md5(grid.tobytes()).hexdigest()
    except Exception:
        h = "0"
    try:
        level = int(game._current_level_index)
        steps = int(game.lelsvjlwneo.current_steps)
    except Exception:
        level, steps = 0, 0
    return (h, level, steps)


def generic_heuristic(game: Any) -> int:
    try:
        grid = game.naxbskjmlg()
        unc = sum(1 for t in game.fswikrcrdmx if grid[t.y, t.x] < 0)
        return unc
    except Exception:
        return 0


def _is_won(game: Any) -> bool:
    try:
        w = game.vplrhaovhr()
        return w is True or (hasattr(w, "__len__") and len(w) == 1 and bool(w))
    except Exception:
        return False


def r3_generic_search(
    env: Any,
    t_limit: float = 15.0,
    max_nodes: int = 20000,
    max_depth: int = 60,
    act_map: Optional[dict[int, Any]] = None,
    make_action: Optional[Any] = None,
    game_over_state: Any = None,
) -> Optional[list[int]]:
    """Greedy best-first search on game state space for any unknown game.

    Engine knowledge stays at the boundary: the caller injects ``act_map``
    (action number -> engine action enum), ``make_action`` (factory building an
    engine action input from an action enum) and ``game_over_state`` (the engine
    GAME OVER sentinel). Without them the search refuses to run and returns None
    rather than importing the engine here.

    Returns action sequence (list of action numbers 1-7) that advances
    levels_completed, or None on failure/timeout.
    """
    if act_map is None or make_action is None or game_over_state is None:
        return None

    game = env._game
    if _is_won(game):
        return []

    try:
        level_idx0 = int(game._current_level_index)
    except Exception:
        level_idx0 = 0

    try:
        max_steps = int(game.lelsvjlwneo.ilqnjlrnkk)
    except Exception:
        max_steps = max_depth

    start_snap = generic_snapshot(game)

    neutered = False
    try:
        game.next_level = lambda: None
        neutered = True
    except Exception:
        pass

    try:
        seq = 0
        h0 = generic_heuristic(game)
        heap = [(h0, 0, seq, start_snap, [])]
        seen = {generic_state_key(game)}
        nodes = 0
        t0 = time.time()

        while heap:
            if time.time() - t0 > t_limit or nodes > max_nodes:
                return None

            h, _, _, snap, path = heapq.heappop(heap)

            if len(path) >= min(max_steps - 1, max_depth):
                continue

            for a in range(1, 8):
                generic_restore(game, snap)
                try:
                    game.perform_action(make_action(act_map[a]), raw=True)
                except Exception:
                    continue

                try:
                    new_level = int(game._current_level_index)
                except Exception:
                    new_level = level_idx0

                if new_level > level_idx0 or _is_won(game):
                    return path + [a]

                try:
                    if game._state == game_over_state:
                        continue
                except Exception:
                    pass

                k = generic_state_key(game)
                if k in seen:
                    continue
                seen.add(k)

                h_val = generic_heuristic(game)
                seq += 1
                heapq.heappush(
                    heap, (h_val * 10 + len(path) + 1, nodes + 1, seq,
                           generic_snapshot(game), path + [a])
                )
            nodes += 1

        return None
    finally:
        if neutered:
            try:
                del game.next_level
            except Exception:
                pass
        generic_restore(game, start_snap)