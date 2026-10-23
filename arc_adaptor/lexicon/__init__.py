"""ARCAID3 lexicon view, ported into the Lingjing-Solo framework.

Two layers, both **game-agnostic** (they read only the frame's colour grid, never
engine internals or game files), which is what makes them usable on any ARC-AGI-3
game — unlike `ar25_solver`, which is written for ar25's reflection physics.

    lexicon.py     screen -> native grid, objects, stable symbol names, object-level diff
    rules.py       one level's steps -> rules `when <action> [if/unless <rel>] then <verb>(<who>, <how>)`,
                   chosen by minimum-description-length, with a closed-world predictor
    rules_view.py  rendering those rules as text (colours as board characters)
    perceive.py    the thin adapter this framework's agents use: numpy frame -> the above

Provenance: Daniel Franzen's Milestone 2 notebook (public LB 27.89), cell
"ARCAID3 lexicon view" v6, itself from arcaid3/part2/lexicon.py. Sources are kept
verbatim apart from one line in `rules.py` (the sandbox's file loader for `LX`
becomes a normal package import).

Reference: arc-agi-3-champion-solution.ipynb (Daniel Franzen champion-solution
notebook; resolve via arc_adaptor/paths.py if a local copy is needed)
"""
from . import lexicon as LX
from . import rules as R
from . import rules_view as RV
from .lexicon import (
    Lexicon, native_grid, scale_lock, objects, layered_objects,
    encode, diff, to_text, native_ascii, background_colour, display_xy, read_roles,
)
from .rules import State, Step, Learner, Rule, observe, predict, score, select
from .perceive import Perception, frame_to_grid

__all__ = [
    "LX", "R", "RV",
    "Lexicon", "native_grid", "scale_lock", "objects", "layered_objects",
    "encode", "diff", "to_text", "native_ascii", "background_colour", "display_xy", "read_roles",
    "State", "Step", "Learner", "Rule", "observe", "predict", "score", "select",
    "Perception", "frame_to_grid",
]
