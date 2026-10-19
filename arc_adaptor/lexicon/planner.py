"""Planning from learned rules, for games we have no hand-written solver for.

`ar25_solver` searches a true physical model, so it plans exactly. On an unseen game
we have no such model — but the rule grammar gives us an *approximate* forward model
learned from play: `predict` says what each action would do. This module turns that
into decisions:

    score(action) = does anything change?  +  is the predicted board new?
                    -  has this action done nothing / killed us here before?

Novelty is the honest objective when the goal is unknown: an action whose predicted
board we have never stood on is worth more than one that returns us somewhere known.
When the game does tell us something — a level completed, a death — `record_outcome`
feeds it back, so the planner stops repeating what did not work.

This is deliberately weaker than a true model. It is the floor for games with no
solver, not a replacement for one.
"""
from __future__ import annotations

from . import worldsim as W
from .perceive import ACTIONS, Perception, frame_to_grid


def _fingerprint(state, preds):
    """The board as (symbol, position) after applying the predicted effects."""
    place = [(p["sym"], tuple(p["at"])) for p in state.objs]
    for _rule, i, eff in preds:
        if i is None or i >= len(place):
            continue
        sym, at = place[i]
        kind = eff[0]
        if kind == "move":
            place[i] = (sym, (at[0] + eff[1][0], at[1] + eff[1][1]))
        elif kind in ("vanish", "reshape"):
            place[i] = None
        elif kind in ("recolor", "turn", "resize"):
            place[i] = (sym + "*", at)
    return tuple(sorted(p for p in place if p is not None))


class RulePlanner:
    """One-step lookahead over learned rules, with visit counts and action guards."""

    def __init__(self, perception: Perception, actions=ACTIONS,
                 w_novel: float = 6.0, w_change: float = 2.0, w_repeat: float = 1.0,
                 w_tabu: float = 5.0, tabu_len: int = 8,
                 depth: int = 8, width: int = 40, deadline: float = 1.5):
        self.p = perception
        self.actions = tuple(actions)
        self.w_novel = w_novel
        self.w_change = w_change
        self.w_repeat = w_repeat
        self.w_tabu = w_tabu        # penalty for returning to a recently seen board
        self.tabu_len = tabu_len
        self.visits: dict = {}
        self.recent: list = []      # last few boards, newest last: stops A1/A2 oscillation
        self.dead: set = set()      # (fingerprint, action) that killed us
        self.noop: set = set()      # (fingerprint, action) that changed nothing
        self.progress: list = []    # actions that led to a level being completed
        # ---- search over the learned forward model ----
        self.depth = depth
        self.width = width
        self.deadline = deadline
        self._plan: list = []       # remaining actions of the current plan
        self._plan_key = None       # board the plan was computed for
        self._searched: dict = {}   # board key -> first action (memo, avoids re-search)
        self.last_plan_len = 0

    # ---------------------------------------------------------------- bookkeeping

    def note_visit(self, grid):
        fp = self.fingerprint_of(grid)
        self.visits[fp] = self.visits.get(fp, 0) + 1
        self.recent.append(fp)
        if len(self.recent) > self.tabu_len:
            self.recent = self.recent[-self.tabu_len:]
        return fp

    def fingerprint_of(self, grid):
        try:
            code = self.p.encode_current(grid)
            from . import rules as R
            st = R.State(code)
            return _fingerprint(st, [])
        except Exception:
            return ()

    def record_outcome(self, fp, action, outcome):
        """outcome: 'death' | 'nothing' | 'progress'."""
        if fp is None:
            return
        if outcome == "death":
            self.dead.add((fp, action))
        elif outcome == "nothing":
            self.noop.add((fp, action))
        elif outcome == "progress":
            self.progress.append((fp, action))

    # ---------------------------------------------------------------- choice

    def scores(self, grid):
        """{action: (score, predicted fingerprint, n predicted changes)}."""
        out = {}
        try:
            code = self.p.encode_current(grid)
            from . import rules as R
            st = R.State(code)
        except Exception:
            return out
        base_fp = _fingerprint(st, [])
        for a in self.actions:
            if (base_fp, a) in self.dead or (base_fp, a) in self.noop:
                out[a] = (float("-inf"), base_fp, 0)
                continue
            try:
                preds = self.p.predict(grid, a)
            except Exception:
                preds = []
            fp = _fingerprint(st, preds)
            seen = self.visits.get(fp, 0)
            score = 0.0
            score += self.w_change * len(preds)
            score += self.w_novel if fp != base_fp and seen == 0 else 0.0
            score -= self.w_repeat * seen
            if fp in self.recent:
                # recently stood here: an action that only takes us back is how A1/A2
                # oscillation burns a level's action budget
                score -= self.w_tabu
            if fp == base_fp and preds:
                score -= self.w_repeat * 2          # rules say: nothing actually moves
            out[a] = (score, fp, len(preds))
        return out

    # ---------------------------------------------------------------- search

    def symstate(self, grid=None):
        """The board as a symbolic state, from the last frame the Perception read."""
        try:
            code = self.p.encode_current(grid) if grid is not None else self.p.last_code
            if code is None:
                return None
            return W.SymState.from_code(code)
        except Exception:
            return None

    def plan(self, grid=None):
        """Search the learned forward model for a plan that reaches an unvisited board.

        Returns (first_action, plan, symstate) or (None, None, symstate).
        """
        ss = self.symstate(grid)
        if ss is None:
            return None, None, None
        fp = ss.fingerprint()
        if self._plan_key == fp and self._plan:
            return self._plan.pop(0), None, ss
        hit = self._searched.get(ss.key())
        if hit is not None:
            self._plan_key = fp
            self.last_plan_len = 1
            return hit, (hit,), ss
        try:
            first, best = W.search(ss, self.p.rules(), self.actions, self.visits,
                                   recent=self.recent, dead_actions=self.dead,
                                   depth=self.depth, width=self.width,
                                   deadline=self.deadline)
        except Exception:
            first, best = None, None
        if first is None:
            return None, None, ss
        self._searched[ss.key()] = first
        self._plan_key = fp
        self.last_plan_len = best[2] if best else 1
        return first, (first,), ss

    def best_action(self, grid):
        """The action worth trying next, or None when nothing is known to do anything.

        Tries the forward-model search first: it can see past an A1/A2 oscillation,
        which one-step scoring cannot. Falls back to the one-step score when the
        model is too thin to plan with.
        """
        a, _plan, _ss = self.plan(grid)
        if a is not None:
            return a
        sc = self.scores(grid)
        if not sc:
            return None
        ranked = sorted(sc.items(), key=lambda kv: kv[1][0], reverse=True)
        top_a, (top_s, _fp, top_n) = ranked[0]
        if top_s == float("-inf"):
            return None
        if top_n == 0:
            # the rules cannot tell the actions apart yet -> fall back to informed probing
            return None
        return top_a
