"""Adapter: an ARC-AGI-3 frame (numpy or FrameData) -> the lexicon's structured view + rules.

Why this layer exists
---------------------
`ar25_solver` reaches 8/8 on ar25, but only because we read that engine's internals
(sprite/axis/target attributes). Nothing of that transfers to another game. The
ARCAID3 lexicon + rule grammar is the game-agnostic half of the champion solution:
it reads **only the colour grid**, so it works on any of the 25 games, and it learns
`when <action> [if/unless <relation>] then <verb>(<who>, <how>)` from the level's own
steps — the "induction" half of "trial, induction, planning".

This module keeps that machinery but holds state across steps (the champion's sandbox
recomputes everything per call because it starts a fresh process for every tool call).
Holding state makes the learner incremental and the per-level scale lock exact.

Usage from an agent
-------------------
    p = Perception()
    p.reset_level(level)                      # on entering / re-entering a level
    p.observe(grid, action="ACTION1")         # call once per step, with the grid you see
    p.rules_text()                            # what has been learned, as text
    p.predict(grid, "ACTION1")                # closed-world: [(rule, obj, effect)]
    p.code_text()                             # the board as objects, for logs or a model

`grid` is a 2-D array of colour values 0-15 (64x64 as the games draw it).
"""
from __future__ import annotations

import time

from . import lexicon as LX
from . import rules as R
from . import rules_view as RV

# The games' action names as the rule grammar sees them. ACTION6 carries a click.
ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7")


def frame_to_grid(frame):
    """FrameData / numpy / nested list -> list[list[int]] of colour values."""
    if frame is None:
        return []
    layers = getattr(frame, "frame", None)
    if layers:
        last = layers[-1]
        return [[int(v) for v in row] for row in last]
    arr = getattr(frame, "tolist", None)
    if arr is not None:
        return [[int(v) for v in row] for row in arr()]
    return [[int(v) for v in row] for row in frame]


class Perception:
    """Structured observation + rule learning for one level at a time.

    Per level it keeps: the Lexicon (stable names, so a picture is called the same
    thing in every frame and every process), the chain of encodings (so objects that
    merged on touching are cut apart again — object permanence), the agreed block size
    (`scale_lock`), and the incremental `Learner`.
    """

    def __init__(self, window: int = 80, budget: float = 8.0, naming: str = "stable"):
        self.window = window
        self.budget = budget          # seconds; the learner is re-scored, not re-run, so this is a guard
        self.naming = naming
        self.level = None
        self.reset_level(None)

    # ------------------------------------------------------------------ level lifecycle

    def reset_level(self, level=None):
        """Start a level: fresh lexicon, learner, scale lock and frame chain."""
        self.level = level
        self.lexicon = LX.Lexicon(naming=self.naming)
        self.learner = R.Learner()
        self.lock = None
        self.lock_grids = []          # distinct grids of this level, for scale_lock
        self.chain = []               # encodings of the level's distinct predecessor frames
        self.seq = []                 # [(action, grid)] for this level, oldest first
        self.last_code = None
        self.last_state = None
        self.events = []              # events of the most recent step
        self.step_count = 0

    # ------------------------------------------------------------------ observation

    def _encode(self, grid, prev=None):
        return LX.encode(grid, self.lexicon, lock=self.lock, prev=prev,
                         seen_at=len(self.seq))

    def encode_current(self, grid):
        """Encode `grid` in the level's chain (used to look at a board without stepping)."""
        prev = self.chain[-1] if self.chain else None
        return self._encode(grid, prev)

    def observe(self, grid, action=None, click=None):
        """Read `grid` as this level's next frame and learn from the change.

        `action` is the action that produced this frame (the first frame of a level
        has none). Returns the events the engine's verbs describe: move / recolor /
        turn / resize / reshape / vanish / appear.
        """
        grid = frame_to_grid(grid)
        if not grid or not grid[0]:
            self.events = []
            return self.events

        # per-level scale lock: one block size agreed across the level's frames
        if self.lock is None and len(self.lock_grids) >= 1:
            try:
                self.lock = LX.scale_lock(self.lock_grids + [grid])
            except Exception:
                self.lock = None
        if not self.lock_grids or self.lock_grids[-1] != grid:
            self.lock_grids.append(grid)
            if len(self.lock_grids) > 256:                 # keep the level's first frames
                self.lock_grids = self.lock_grids[:2] + self.lock_grids[-254:]

        prev = self.chain[-1] if self.chain else None
        code = self._encode(grid, prev)
        self.chain.append(code)
        if len(self.chain) > 8:                            # carve needs only recent predecessors
            self.chain = [self.chain[0]] + self.chain[-7:]
        state = R.State(code)

        self.events = []
        if action and self.last_state is not None:
            a, b = self.last_state, state
            if (a.code["scale"], a.code["offset"]) == (b.code["scale"], b.code["offset"]):
                try:
                    self.events = R.observe(a, b)
                except Exception:
                    self.events = []
                if self.events or action:
                    try:
                        self.learner.observe(a, action, click, self.events)
                    except Exception:
                        pass
        self.seq.append((action or "", grid))
        if len(self.seq) > self.window + 1:
            self.seq = [self.seq[0]] + self.seq[-self.window:]
        self.last_code = code
        self.last_state = state
        self.step_count += 1
        return self.events

    # ------------------------------------------------------------------ what was learned

    def rules(self):
        try:
            return self.learner.rules()
        except Exception:
            return []

    def rules_text(self, chars=None, max_rules: int = 10) -> str:
        """The learned rules, and the last step's changes that no rule explains."""
        rs = self.rules()
        where = f"level {self.level}" if self.level is not None else "this level"
        lines = [f"RULES for {where} (when <action> [if/unless <relation>] then <change>; "
                 f"counts: right / wrong on this level):"]
        lines += ["  " + r.line() for r in rs[:max_rules]]
        if len(rs) > max_rules:
            lines.append(f"  ... and {len(rs) - max_rules} more")
        if not rs:
            lines.append("  (none yet: no change has repeated often enough)")
        try:
            resid = self.learner.residual()
        except Exception:
            resid = []
        if resid:
            lines.append(f"UNEXPLAINED by any rule ({len(resid)} of the level's changes):")
            for _, e in resid[-4:]:
                lines.append("  " + RV.describe(e))
        return RV.with_chars("\n".join(lines), chars)

    def code_text(self, grid=None) -> str:
        """The board as objects: true cell size, legend of pictures, placements."""
        code = self.encode_current(grid) if grid is not None else self.last_code
        if code is None:
            return "(no frame read yet)"
        return LX.to_text(code)

    def native_ascii(self, grid=None) -> str:
        code = self.encode_current(grid) if grid is not None else self.last_code
        return LX.native_ascii(code) if code is not None else ""

    # ------------------------------------------------------------------ prediction

    def predict(self, grid, action, click=None):
        """Closed-world prediction of `action` on `grid`: [(rule, object index or None, effect)].

        If no rule fires, nothing is predicted to change — which is exactly the signal an
        agent needs to stop repeating an action that has never done anything.
        """
        code = self.encode_current(grid) if grid is not None else self.last_code
        if code is None:
            return []
        try:
            preds, _ = R.predict(self.rules(), R.State(code), action, click)
        except Exception:
            return []
        return preds

    def effects_by_action(self, grid, actions=ACTIONS):
        """For each action, what the learned rules say will change. Dead actions stand out."""
        out = {}
        for a in actions:
            out[a] = [(r.text(), eff) for r, _i, eff in self.predict(grid, a)]
        return out
