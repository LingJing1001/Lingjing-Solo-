"""A forward model built from learned rules, so the agent can plan instead of flinch.

`planner.RulePlanner` scores one action at a time. That is enough to notice "this
changes something", but it cannot see past a two-state oscillation: if A1 and A2 undo
each other, one-step lookahead rates both equally and the agent ping-pongs until the
level's action budget is gone.

This module turns the same rules into a *search*:

    SymState  --action-->  SymState'          (apply the predicted effects)
    search()  --BFS----->  a plan that reaches a board we have never stood on

Two honest limits, both surfaced rather than hidden:

* Only `move` / `recolor` / `vanish` are applied exactly. `resize` / `reshape` /
  `turn` / `appear` change an object's geometry in ways the `code` round-trip would
  have to invent, so a state touched by one of them is marked `sure=False` and is
  never used as a search goal.
* When no rule fires for an action we do not know what happens. Such a successor is
  kept (the action is still legal) but flagged `sure=False`.

Unsure states can be traversed, never trusted. That keeps the search from confidently
walking into a hallucinated board while still letting it cross regions the rules do
not explain yet.
"""
from __future__ import annotations

import collections
from typing import Optional

from . import rules as R

# effects whose geometry we can reproduce exactly
_EXACT = ("move", "recolor", "vanish", "move+recolor")


class Obj:
    """One placed object: its name, anchor, and the cells+colours relative to the anchor."""

    __slots__ = ("sym", "at", "local", "container", "hash")

    def __init__(self, sym, at, local, container=False):
        self.sym = sym
        self.at = (int(at[0]), int(at[1]))
        self.local = tuple(sorted(local, key=lambda kv: kv[0]))   # (((dr,dc), colour), ...)
        self.container = bool(container)
        self.hash = hash(self.local)

    @property
    def cells(self):
        return frozenset((self.at[0] + dr, self.at[1] + dc) for (dr, dc), _ in self.local)

    @property
    def size(self):
        return len(self.local)

    def moved(self, d):
        return Obj(self.sym, (self.at[0] + d[0], self.at[1] + d[1]), self.local, self.container)

    def recoloured(self, mapping):
        loc = tuple(((dr, dc), mapping.get(col, col)) for (dr, dc), col in self.local)
        return Obj(self.sym, self.at, loc, self.container)

    def key(self):
        return (self.sym, self.at, self.hash)

    def __repr__(self):
        return f"<{self.sym}@{self.at} n={self.size}>"


class SymState:
    """A board as objects only: no pixels, cheap to clone and compare."""

    __slots__ = ("objs", "shape", "bg", "lexicon", "sure", "_key", "_state", "_code")

    def __init__(self, objs, shape, bg, lexicon=None, sure=True):
        self.objs = tuple(objs)
        self.shape = (int(shape[0]), int(shape[1]))
        self.bg = int(bg)
        self.lexicon = lexicon
        self.sure = sure
        self._key = None
        self._state = None
        self._code = None

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_code(cls, code):
        grid = code["native"]
        objs = []
        for p in code["placed"]:
            at = (int(p["at"][0]), int(p["at"][1]))
            local = []
            for rc in p["cells"]:
                r, c = int(rc[0]), int(rc[1])
                col = int(grid[r][c]) if 0 <= r < len(grid) and 0 <= c < len(grid[r]) else code["background"]
                local.append(((r - at[0], c - at[1]), col))
            objs.append(Obj(p["sym"], at, local, p.get("container", False)))
        return cls(objs, code["shape"], code["background"], code.get("lexicon"))

    # ------------------------------------------------------------------ views

    def key(self):
        if self._key is None:
            self._key = tuple(sorted(o.key() for o in self.objs))
        return self._key

    def fingerprint(self):
        """Fingerprint that ignores exact colours — what `planner` counts as a board."""
        return tuple(sorted((o.sym, o.at) for o in self.objs if not o.container))

    def to_code(self):
        if self._code is not None:
            return self._code
        h, w = self.shape
        grid = [[self.bg] * w for _ in range(h)]
        placed = []
        for o in self.objs:
            cells = []
            for (dr, dc), col in o.local:
                r, c = o.at[0] + dr, o.at[1] + dc
                if 0 <= r < h and 0 <= c < w:
                    grid[r][c] = col
                    cells.append((r, c))
            if not cells:
                continue
            placed.append({"sym": o.sym, "at": o.at, "size": len(cells), "cells": cells,
                           "depth": 1, "container": o.container})
        self._code = {"scale": 1, "offset": (0, 0), "shape": self.shape,
                      "background": self.bg, "residue": [], "placed": placed,
                      "lexicon": self.lexicon, "native": grid, "locked": False}
        return self._code

    def to_state(self):
        if self._state is None:
            try:
                self._state = R.State(self.to_code())
            except Exception:
                return None
        return self._state

    def out_of_bounds(self):
        h, w = self.shape
        for o in self.objs:
            for (dr, dc), _ in o.local:
                r, c = o.at[0] + dr, o.at[1] + dc
                if not (0 <= r < h and 0 <= c < w):
                    return True
        return False

    def overlapping(self):
        seen = set()
        for o in self.objs:
            if o.container:
                continue
            for rc in o.cells:
                if rc in seen:
                    return True
                seen.add(rc)
        return False

    def __eq__(self, other):
        return isinstance(other, SymState) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def __repr__(self):
        return f"<SymState {len(self.objs)} objs sure={self.sure}>"


# --------------------------------------------------------------------------- transition

def successors(ss: SymState, rules, actions) -> list:
    """[(action, SymState)] — what the learned rules say each action does here.

    The state is returned even when no rule fires (the action is still legal); it is
    flagged unsure so it can be crossed but not trusted.
    """
    st = ss.to_state()
    if st is None:
        return []
    out = []
    for a in actions:
        try:
            preds, _click = R.predict(rules, st, a)
        except Exception:
            preds = []
        if not preds:
            out.append((a, SymState(ss.objs, ss.shape, ss.bg, ss.lexicon, sure=False)))
            continue
        nxt = _apply(ss, preds)
        if nxt is None:
            continue
        out.append((a, nxt))
    return out


def _apply(ss: SymState, preds) -> Optional[SymState]:
    """Apply one step's predicted effects. None if the result is not representable."""
    objs = list(ss.objs)
    dead = set()
    shift = collections.defaultdict(lambda: [0, 0])
    recolor = {}
    sure = ss.sure

    for _rule, i, eff in preds:
        kind = eff[0]
        if kind == "move":
            if i is None or i >= len(objs):
                continue
            shift[i][0] += eff[1][0]
            shift[i][1] += eff[1][1]
        elif kind == "vanish":
            if i is not None:
                dead.add(i)
        elif kind == "recolor":
            if i is None or i >= len(objs):
                continue
            mp = dict(eff[1])
            recolor.setdefault(i, {}).update(mp)
        else:
            # resize / reshape / turn / appear: geometry we cannot reproduce exactly
            sure = False

    new = []
    for idx, o in enumerate(objs):
        if idx in dead:
            continue
        if idx in shift and (shift[idx][0] or shift[idx][1]):
            o = o.moved(tuple(shift[idx]))
        if idx in recolor:
            o = o.recoloured(recolor[idx])
        new.append(o)

    nxt = SymState(new, ss.shape, ss.bg, ss.lexicon, sure=sure)
    if nxt.out_of_bounds():
        return None
    if nxt.overlapping():
        # our model has no notion of pushing / stacking; treat as unknown
        return None
    return nxt


# --------------------------------------------------------------------------- search

def search(start: SymState, rules, actions, visits: dict, recent=(), dead_actions=frozenset(),
           depth: int = 8, width: int = 40, deadline: float = 1.5, clock=None):
    """Look `depth` actions ahead through the learned model; return the first action
    of the shortest plan that reaches a board we have never stood on.

    Ranking is by novelty, because the goal is unknown. `visits` counts how often the
    real agent has stood on a board; `recent` marks boards from the last few steps, so
    a plan that merely walks back is not mistaken for progress.
    """
    import time
    t0 = clock() if clock else time.time()
    recent = set(recent)
    best = None                      # (score_key, first_action, path_len)
    seen = {start.key()}
    frontier = [(start, ())]
    explored = 0

    for d in range(depth):
        nxt_frontier = []
        for ss, path in frontier:
            if explored > 4000:
                break
            if (clock() if clock else time.time()) - t0 > deadline:
                break
            for a, nxt in successors(ss, rules, actions):
                explored += 1
                k = nxt.key()
                if k in seen:
                    continue
                seen.add(k)
                p = path + (a,)
                frm = ss.fingerprint()
                if (frm, a) in dead_actions:
                    continue
                nxt_frontier.append((nxt, p))

                # only states the model actually understands may be goals
                if not nxt.sure:
                    continue
                fp = nxt.fingerprint()
                n = visits.get(fp, 0)
                tabu = 1 if fp in recent else 0
                # prefer: never visited, not recently stood on, fewer actions
                rank = (n, tabu, len(p))
                if best is None or rank < best[0]:
                    best = (rank, p[0], len(p), a)
        if not nxt_frontier:
            break
        # keep the most promising frontier: unvisited first, then shallower
        nxt_frontier.sort(key=lambda item: (visits.get(item[0].fingerprint(), 0),
                                            item[0].fingerprint() in recent,
                                            len(item[1])))
        frontier = nxt_frontier[:width]
        # a good-enough plan: reach a board never visited, in few actions
        if best is not None and best[0][0] == 0 and best[0][1] == 0:
            break

    if best is None:
        return None, None
    return best[1], best


def describe(sym: SymState, limit: int = 6):
    return " ".join(repr(o) for o in sym.objs[:limit])
