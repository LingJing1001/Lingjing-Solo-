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


def _fingerprint(state, preds, ignore=frozenset()):
    """The board as (symbol, position) after applying the predicted effects.

    `ignore` drops decorative objects so this matches `worldsim.SymState.fingerprint`.
    """
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
    return tuple(sorted(p for i, p in enumerate(place)
                        if p is not None and i not in ignore))


class RulePlanner:
    """One-step lookahead over learned rules, with visit counts and action guards."""

    def __init__(self, perception: Perception, actions=ACTIONS,
                 w_novel: float = 6.0, w_change: float = 2.0, w_repeat: float = 1.0,
                 w_tabu: float = 5.0, w_try: float = 1.5, tabu_len: int = 8,
                 depth: int = 8, width: int = 40, deadline: float = 1.5):
        self.p = perception
        self.actions = tuple(actions)
        self.w_novel = w_novel
        self.w_change = w_change
        self.w_repeat = w_repeat
        self.w_tabu = w_tabu        # penalty for returning to a recently seen board
        self.w_try = w_try          # penalty for re-pressing the same key here
        self.tabu_len = tabu_len
        self.visits: dict = {}
        self.recent: list = []      # last few boards, newest last: stops A1/A2 oscillation
        self.dead: set = set()      # (fingerprint, action) that killed us
        self.noop: set = set()      # (fingerprint, action) that changed nothing
        self.tries: dict = {}       # (fingerprint, action) -> how often we pressed it
        self.progress: list = []    # actions that led to a level being completed
        # ---- search over the learned forward model ----
        self.depth = depth
        self.width = width
        self.deadline = deadline
        self._plan: list = []       # remaining actions of the current plan
        self._plan_key = None       # board the plan was computed for
        self._searched: dict = {}   # board key -> first action (memo, avoids re-search)
        self.last_plan_len = 0
        # Symbols seen in the last few frames. Anything whose name changes from frame
        # to frame is a footer / timer, not part of the board's identity.
        self._sym_hist: list = []
        self._stable: frozenset | None = None
        self._moved_syms: set = set()   # symbols a rule translates: always identity

    # ---------------------------------------------------------------- bookkeeping

    def _observe_symbols(self, grid):
        """Track which symbols persist across frames.

        In ARC-AGI-3 the footer / step counter is rendered as objects whose symbol is
        derived from their pixels, so the name changes on every move (`W9717` ->
        `W23c7` -> `W423c`). Such symbols never survive the window, so they never
        enter the board's identity, and novelty stops being drowned by furniture.
        """
        try:
            code = self.p.encode_current(grid) if grid is not None else self.p.last_code
            if not code:
                return
            syms = frozenset(p["sym"] for p in code["placed"])
        except Exception:
            return
        self._sym_hist.append(syms)
        if len(self._sym_hist) > 4:
            self._sym_hist = self._sym_hist[-4:]
        inter = frozenset.intersection(*self._sym_hist)
        if not inter:
            return
        # Monotone shrinking: a symbol once judged furniture stays out for good.
        # Letting the set swing back and forth (4 -> 5 -> 4 symbols) changes the
        # fingerprint's length, which silently invalidates every stored visit.
        self._stable = inter if self._stable is None else (self._stable & inter)

    def note_visit(self, grid):
        self._observe_symbols(grid)
        fp = self.fingerprint_of(grid)
        self.visits[fp] = self.visits.get(fp, 0) + 1
        self.recent.append(fp)
        if len(self.recent) > self.tabu_len:
            self.recent = self.recent[-self.tabu_len:]
        return fp
    def fingerprint_of(self, grid):
        """Board identity for visit counting.

        Routed through `symstate` so it excludes the same decorative objects the
        search does. Two fingerprints computed by different code paths is exactly
        how novelty silently stops working.
        """
        ss = self.symstate(grid)
        return ss.fingerprint() if ss is not None else ()

    def note_try(self, fp, action):
        """Count that `action` was pressed on board `fp`.

        Without this, a board where one action has *any* prediction keeps returning
        that action for ever: the plan returns it, `best_action` trusts the plan, and
        the other primitives are never sampled — the agent sits in a dead end
        re-pressing a wall.
        """
        if fp is None or action is None:
            return
        self.tries[(fp, action)] = self.tries.get((fp, action), 0) + 1

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
        # Denoise first. A footer that resizes on every action used to make every
        # action look eventful and novel, so the ranking was dominated by decoration
        # rather than by what the action does to the board.
        per, _noise, decor = ({}, set(), frozenset())
        try:
            per, _noise, decor = W.action_noise(W.SymState.from_code(code),
                                                self.p.rules(), self.actions)
        except Exception:
            per = {}

        def _keep(fp):
            if self._stable is None:
                return fp
            return tuple(p for p in fp if p[0].rstrip("*") in self._stable)

        base_fp = _keep(_fingerprint(st, [], decor))

        for a in self.actions:
            if (base_fp, a) in self.dead or (base_fp, a) in self.noop:
                out[a] = (float("-inf"), base_fp, 0)
                continue
            if per:
                preds = per.get(a, [])
            else:
                try:
                    preds = self.p.predict(grid, a)
                except Exception:
                    preds = []
            fp = _keep(_fingerprint(st, preds, decor))
            seen = self.visits.get(fp, 0)
            score = 0.0
            score += self.w_change * len(preds)
            score += self.w_novel if fp != base_fp and seen == 0 else 0.0
            score -= self.w_repeat * seen
            # pressing the same key on the same board again buys less each time
            score -= self.w_try * self.tries.get((base_fp, a), 0)
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
        """The board as a symbolic state, from the last frame the Perception read.

        Decorative objects (footers, timers) are marked so they drop out of the
        fingerprint — see `worldsim.action_noise`.
        """
        try:
            code = self.p.encode_current(grid) if grid is not None else self.p.last_code
            if code is None:
                return None
            ss = W.SymState.from_code(code)
            try:
                per, _noise, decor = W.action_noise(ss, self.p.rules(), self.actions)
                # Whatever a rule translates is by definition part of the board's
                # identity: if the player's symbol ever fell out of the stable set,
                # the fingerprint would stop distinguishing positions and novelty
                # ranking would collapse to "pick the first action".
                moved = set()
                for preds in per.values():
                    for _r, i, eff in preds:
                        if eff[0] == "move" and i is not None and i < len(ss.objs):
                            moved.add(ss.objs[i].sym)
                self._moved_syms |= moved
                keep = ((self._stable | self._moved_syms) if self._stable else None)
                if decor or keep is not None:
                    ss = ss.with_hints(ignore=decor or frozenset(), keep_syms=keep)
            except Exception:
                pass
            return ss
        except Exception:
            return None

    def plan(self, grid=None):
        """Search the learned forward model for a plan that reaches an unvisited board.

        Deliberately *not* memoised. A cached first action stays cached after the real
        engine proves it a no-op or a death, and the agent then walks into the same
        wall forever while the board never changes. Re-searching every step is the
        MPC contract: the model proposes, the engine disposes.

        Returns (first_action, plan, symstate) or (None, None, symstate).
        """
        ss = self.symstate(grid)
        if ss is None:
            return None, None, None
        try:
            first, best = W.search(ss, self.p.rules(), self.actions, self.visits,
                                   recent=self.recent, dead_actions=self.dead,
                                   depth=self.depth, width=self.width,
                                   deadline=self.deadline)
        except Exception:
            first, best = None, None
        if first is None:
            return None, None, ss
        self.last_plan_len = best[2] if best else 1
        return first, (first,), ss

    def best_action(self, grid):
        """The action worth trying next, or None when nothing is known to do anything.

        Tries the forward-model search first: it can see past an A1/A2 oscillation,
        which one-step scoring cannot. Falls back to the one-step score when the
        model is too thin to plan with.
        """
        a, _plan, _ss = self.plan(grid)
        fp = self.fingerprint_of(grid)
        if a is not None and self.tries.get((fp, a), 0) < 2:
            return a
        # Either the model is silent, or it keeps proposing a key we have already
        # pressed here twice without anything new happening. Both mean: stop
        # exploiting, start sampling the primitives we have not tried.
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
