"""Generic game snapshot/restore/search for unknown environments.

Unlike ``arc_shadow.py`` (whose attribute names belong to AR25 only), every
signal read here comes from the surface the engine base class gives *all*
games. Verified on this box against three different games (ar25 / ls20 / sc25):

    common: _state, _current_level_index, level_index, is_last_level,
            _action_count, _available_actions, __dict__

The previous revision of this module looked up ``naxbskjmlg``,
``lelsvjlwneo``, ``fswikrcrdmx`` and ``vplrhaovhr`` — AR25's obfuscated
attributes — each inside a bare ``except Exception``. On any other game those
lookups failed *silently*, the state key collapsed to the constant ``"0"``, the
first seven children were all judged duplicates, and the search gave up after
expanding a single node in ~0.1 s. Nothing raised, which is why the collapse
went unnoticed. Now:

* the state key is a digest over the same ``__dict__`` snapshot the search
  already stores, so it is a real state signature on any game;
* win / game-over / action construction are **injected by the boundary**; the
  planning layer still must not import ``arcengine``;
* if those are missing, or no state signature can be built, the search reports
  why via :func:`support_reason` instead of pretending to search.

Usage:
    # The caller (a boundary adapter) injects the engine objects.
    path = r3_generic_search(
        env,
        act_map={n: getattr(GameAction, "ACTION%d" % n) for n in range(1, 8)},
        make_action=lambda aid: ActionInput(id=aid, data={}, reasoning=None),
        game_over_state=GameState.GAME_OVER,
        win_state=GameState.WIN,
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
    """Copy everything reachable through ``__dict__`` (no engine names)."""
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


def _encode(obj: Any, _depth: int = 0) -> bytes:
    """Deterministic, **identity-free** encoding for snapshot payloads.

    Opaque engine objects (Sprite, Level, …) must not fall back to ``repr``:
    that prints a memory address, so a deep-copied twin of an identical sprite
    would hash differently and the search's dedup would never fire. Recurse
    into the object's own ``__dict__`` instead, with a depth cap for cycles.
    """
    if isinstance(obj, np.ndarray):
        return b"A" + repr(obj.shape).encode() + str(obj.dtype).encode() + obj.tobytes()
    if isinstance(obj, np.generic):
        return _encode(obj.item(), _depth)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return b"P" + repr(obj).encode()
    if isinstance(obj, dict):
        return b"D" + b"".join(
            _encode(k, _depth + 1) + _encode(v, _depth + 1)
            for k, v in sorted(obj.items(), key=lambda kv: repr(kv[0]))
        )
    if isinstance(obj, (list, tuple)):
        return b"L" + b"".join(_encode(v, _depth + 1) for v in obj)
    if isinstance(obj, (set, frozenset)):
        return b"E" + b"".join(
            _encode(v, _depth + 1) for v in sorted(obj, key=repr)
        )
    inner = getattr(obj, "__dict__", None)
    if inner and _depth < 4:
        return (b"S" + type(obj).__name__.encode() + b"|"
                + _encode(dict(inner), _depth + 1))
    # Last resort: the class name only — never an address.
    return b"C" + type(obj).__name__.encode()


def _level_index(game: Any) -> int:
    """Current level, from whichever name the engine exposes."""
    for name in ("_current_level_index", "level_index"):
        try:
            return int(getattr(game, name))
        except Exception:
            continue
    return 0


def generic_state_key(game: Any) -> Optional[tuple]:
    """State signature = (digest of the whole snapshot, level index).

    ``_action_count`` rides inside the snapshot, so two paths of different
    length that land on the same board are *not* merged — that costs pruning
    but can never merge two genuinely different states, which is the failure
    mode this function had before. Returns ``None`` when no signature can be
    built, so callers fail loudly instead of searching blind.
    """
    snap = generic_snapshot(game)
    if not snap:
        return None
    payload = b"".join(_encode(k) + _encode(v) for k, v in sorted(snap.items()))
    return (hashlib.md5(payload).hexdigest(), _level_index(game))


def generic_heuristic(game: Any) -> int:
    """Always 0: no game-independent distance-to-goal exists in pixels alone.

    With h == 0 the priority below degenerates into breadth-first by depth,
    which is the honest default for an unknown game. Claiming a heuristic that
    silently returned 0 on every non-ar25 game was worse than admitting this.
    """
    return 0


def _is_won(game: Any, win_state: Any) -> bool:
    try:
        return game._state == win_state
    except Exception:
        return False


def support_reason(
    game: Any,
    act_map: Optional[dict[int, Any]] = None,
    make_action: Optional[Any] = None,
    game_over_state: Any = None,
    win_state: Any = None,
) -> Optional[str]:
    """Why the search cannot run on ``game``, or ``None`` if it can.

    Kept public so the boundary can log the reason instead of watching the
    search return ``None`` for an unexplained third time.
    """
    missing = [
        name for name, val in (("act_map", act_map), ("make_action", make_action),
                               ("game_over_state", game_over_state),
                               ("win_state", win_state))
        if val is None
    ]
    if missing:
        return "missing injection: " + ", ".join(missing)
    try:
        if not generic_snapshot(game):
            return "empty snapshot: __dict__ gives nothing to restore"
    except Exception as exc:  # noqa: BLE001
        return f"snapshot failed: {type(exc).__name__}"
    if generic_state_key(game) is None:
        return "no state signature: search would blind-collapse"
    return None


def r3_generic_search(
    env: Any,
    t_limit: float = 15.0,
    max_nodes: int = 20000,
    max_depth: int = 60,
    act_map: Optional[dict[int, Any]] = None,
    make_action: Optional[Any] = None,
    game_over_state: Any = None,
    win_state: Any = None,
) -> Optional[list[int]]:
    """Breadth-first state-space search for any game the engine exposes.

    Engine knowledge stays at the boundary: the caller injects ``act_map``
    (action number -> engine action enum), ``make_action`` (factory building an
    engine action input from an action enum), ``game_over_state`` and
    ``win_state``. Without them the search refuses to run and returns ``None``
    rather than importing the engine here — see :func:`support_reason` for the
    diagnosable reason.

    Returns the action sequence (numbers 1-7) that advances
    ``_current_level_index``, or ``None`` on failure/timeout.
    """
    reason = support_reason(
        env._game if env is not None else None,
        act_map, make_action, game_over_state, win_state,
    ) if env is not None else "no env"
    if reason is not None:
        return None

    game = env._game
    if _is_won(game, win_state):
        return []

    level_idx0 = _level_index(game)
    start_snap = generic_snapshot(game)

    # Reaching the last level's goal ends the game; stop the engine from
    # advancing underneath the search while it explores.
    neutered = False
    try:
        game.next_level = lambda: None
        neutered = True
    except Exception:
        pass

    try:
        seq = 0
        heap = [(generic_heuristic(game), 0, seq, start_snap, [])]
        root_key = generic_state_key(game)
        if root_key is None:
            return None
        seen = {root_key}
        nodes = 0
        t0 = time.time()

        while heap:
            if time.time() - t0 > t_limit or nodes > max_nodes:
                return None

            h, _, _, snap, path = heapq.heappop(heap)

            if len(path) >= max_depth:
                continue

            for a in sorted(act_map):
                # One deadline check per pop is not enough: expanding a single
                # node costs ~1 s on a slow game, which overshot t_limit=12.0
                # to 13.8 s in measurement. Check per child instead.
                if time.time() - t0 > t_limit:
                    return None
                generic_restore(game, snap)
                try:
                    game.perform_action(make_action(act_map[a]), raw=True)
                except Exception:
                    continue

                if _level_index(game) > level_idx0 or _is_won(game, win_state):
                    return path + [a]

                try:
                    if game._state == game_over_state:
                        continue
                except Exception:
                    pass

                k = generic_state_key(game)
                if k is None or k in seen:
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
