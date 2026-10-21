"""A read-only view of what the helper layers believe.

Why this exists
---------------
The champion solution (Tufa Labs Duck + Franzen) has no goal detector: a 128K
model reads the board and reasons about it. We do not have that model, so the
detectors are written by hand. The risk is that they quietly become *deciders*:
once a human heuristic picks the action, the model is demoted to an actuator and
a wrong heuristic can never be corrected by the thing that is actually thinking.

This module takes the opposite position. Every layer that already runs
(goalsim, its GoalPolicy, graphsim, the rule planner) can *report* what it found.
The report joins the model's context as one more section next to BOARD and
NATIVE. The model may use it, ignore it, or contradict it -- everything here is
derived from the same frames it can read itself, so contradicting it costs one
look at NATIVE.

Five rules keep a view from becoming a decider:

1. **State evidence, not instructions.** "the goal mask is 15 cells" is a view.
   "press A2 eleven times" is an order. The single exception is DELTA, which is a
   proposal and says so.
2. **Say how sure.** A layer that has seen two frames is guessing, and a guess
   dressed as a fact is worse than no view at all. Below four frames the mask is
   still being re-picked, so DELTA is withheld entirely: a precise instruction
   derived from a target that may change in the next frame is worse than silence.
3. **Print nothing when there is nothing.** An empty section costs tokens and
   teaches the model that the section is not worth reading.
4. **Never restate the board.** BOARD/NATIVE already carry the pixels; the value
   here is only the *inference* layer on top of them.
5. **Same coordinates as NATIVE.** The layers work in screen pixels; NATIVE and
   BOARD are in world cells. Reporting `r45..53` next to a 21-row map makes the
   section unusable, so every number here is converted through the detected
   block size before it is printed.

Rule zero is in the prompt, not here: the system prompt must name this section.
The champion measured it -- their first view was used 0 times in 1,130 tool calls
because the prompt still said a frame exposes "only" five fields. A view the
prompt does not mention does not exist.
"""
from __future__ import annotations

from typing import Any, Optional

try:  # imported as lexicon.view
    from .goalsim import decompose
except ImportError:  # pragma: no cover - run as a loose script
    from goalsim import decompose  # type: ignore


HEAD = ("INFERRED (from helper layers; read-only, may be wrong -- check against "
        "BOARD/NATIVE/CHANGES before relying on it):")

# A mask that has only been seen once is not evidence. goalsim re-picks the target
# while a level is young; before that settles, any distance derived from it is noise.
_SETTLED_FRAMES = 4


class _Conv:
    """Screen pixels -> world cells, using the block size the board was drawn at."""

    def __init__(self, code: Optional[dict]):
        code = code or {}
        self.k = max(1, int(code.get("scale") or 1))
        oy, ox = code.get("offset") or (0, 0)
        self.oy, self.ox = int(oy), int(ox)
        self.unit = "cell" if code else "px"

    def pt(self, r: int, c: int) -> tuple[int, int]:
        return (int(r - self.oy) // self.k, int(c - self.ox) // self.k)

    def n(self, d: float) -> int:
        return int(round(float(d) / self.k))


def _short(action: str) -> str:
    return str(action).replace("ACTION", "A")


def _dir(d) -> str:
    dy, dx = int(d[0]), int(d[1])
    parts = []
    if dy:
        parts.append("down" if dy > 0 else "up")
    if dx:
        parts.append("right" if dx > 0 else "left")
    return "+".join(parts)


def _span(w0: tuple, w1: tuple) -> str:
    return f"r{w0[0]}..{w1[0]} c{w0[1]}..{w1[1]}"


def _frames(goal_model) -> int:
    return int(getattr(goal_model, "frames", 0) or 0)


def _goal_line(goal_model, conv: _Conv) -> Optional[str]:
    goal = getattr(goal_model, "goal", None)
    if goal is None or not getattr(goal, "cells", None):
        return None
    y0, x0, y1, x1 = goal.bbox
    a = conv.pt(y0, x0)
    b = conv.pt(y1, x1)
    frames = _frames(goal_model)
    if frames >= _SETTLED_FRAMES:
        state = f"mask unchanged for {frames} frames"
    else:
        state = f"PROVISIONAL: picked from {frames} frame(s), still being re-picked"
    return (f"GOAL (hypothesis): colour {goal.colour}, {len(goal.cells)} screen cells, "
            f"{_span(a, b)} in world cells, why={goal.why}; {state}")


def _cover_line(goal_model, grid) -> Optional[str]:
    mask = getattr(goal_model, "goal_mask", None)
    goal = getattr(goal_model, "goal", None)
    if not mask or goal is None or not grid:
        return None
    h, w = len(grid), len(grid[0]) if grid else 0
    ref = getattr(goal, "ref", None) or {}
    changed = 0
    for y, x in mask:
        if y >= h or x >= w:
            continue
        if int(grid[y][x]) != ref.get((y, x), goal.colour):
            changed += 1
    return (f"COVER: {changed}/{len(mask)} goal pixels changed since level start "
            f"({changed / max(1, len(mask)):.0%})")


def _is_wall(piece, w_h: tuple) -> bool:
    """A piece that spans the whole board is a wall or border, not something a key moves."""
    h, w = w_h
    y0, x0, y1, x1 = piece.at
    return (y1 - y0 + 1) >= 0.9 * h or (x1 - x0 + 1) >= 0.9 * w


def _pick(pieces, mask, w_h):
    live = [p for p in pieces if not _is_wall(p, w_h)]
    if not live:
        return None, pieces
    if mask:
        return max(live, key=lambda p: len(p.cells & mask)), live
    return live[0], live


def _mover_line(goal_policy, mask, conv: _Conv, w_h) -> Optional[str]:
    if goal_policy is None:
        return None
    raw = getattr(goal_policy, "cursors", None) or []
    piece, live = _pick(raw, mask, w_h)
    if piece is None:
        return "MOVER: no movable piece found"
    a = conv.pt(piece.at[0], piece.at[1])
    b = conv.pt(piece.at[2], piece.at[3])
    if getattr(goal_policy, "linked_now", False):
        # More than one match is common (ar25 draws the piece and its reflection).
        # Saying which one is tracked matters: on a reflection board the keys may
        # drive the other one directly.
        extra = ("" if len(live) == 1 else
                 f"; {len(live)} pieces share that shape, so one may be a mirror that "
                 f"the keys only move indirectly")
        return (f"MOVER: the goal is shaped like a movable piece; tracked "
                f"{_span(a, b)}, colour {piece.colour}{extra}")
    return (f"MOVER: {len(live)} movable piece(s), none matches the goal shape -- "
            f"the target may not be 'push a copy onto it'; largest at {_span(a, b)}")


def _keys_line(goal_policy, conv: _Conv) -> Optional[str]:
    if goal_policy is None:
        return None
    disp = getattr(goal_policy, "disp", None) or {}
    blocked = getattr(goal_policy, "blocked", None) or set()
    if not disp and not blocked:
        return None
    items = []
    for a, d in sorted(disp.items()):
        step = max(abs(conv.n(d[0])), abs(conv.n(d[1]))) or 1
        items.append(f"{_short(a)}={_dir(d)} {step}{conv.unit}")
    for a in sorted(blocked):
        if a not in disp:
            items.append(f"{_short(a)}=no move")
    return "KEYS (learned from real steps): " + " | ".join(items)


def _summarise(actions) -> str:
    out = []
    for a in actions:
        if out and out[-1][0] == a:
            out[-1][1] += 1
        else:
            out.append([a, 1])
    return " ".join(f"{_short(a)}x{n}" for a, n in out)


def _delta_line(goal_model, goal_policy, mask, conv: _Conv, w_h) -> Optional[str]:
    if goal_policy is None or not mask:
        return None
    if _frames(goal_model) < _SETTLED_FRAMES:
        # Deliberately silent: a number computed from an unstable mask reads as
        # authoritative and is not.
        return None
    raw = getattr(goal_policy, "cursors", None) or []
    piece, _live = _pick(raw, mask, w_h)
    if piece is None:
        return None
    ty = min(c[0] for c in mask)
    tx = min(c[1] for c in mask)
    dy, dx = conv.n(ty - piece.at[0]), conv.n(tx - piece.at[1])
    if (dy, dx) == (0, 0):
        return ("DELTA: the tracked piece already sits on the goal corner -- if the level "
                "did not advance, alignment alone is not the win condition")
    disp = {k: v for k, v in (getattr(goal_policy, "disp", None) or {}).items()
            if k not in (getattr(goal_policy, "blocked", None) or set())}
    plan = decompose((ty - piece.at[0], tx - piece.at[1]), disp) if disp else []
    if plan:
        tail = (f"; a greedy split is {_summarise(plan)} -- a proposal, not something "
                f"you have to follow")
    else:
        tail = "; no learned key expresses this move yet"
    return (f"DELTA: tracked piece needs {dy:+d} rows, {dx:+d} cols to reach the goal "
            f"corner{tail}")


def _graph_line(graph, fp) -> Optional[str]:
    if graph is None:
        return None
    edges = getattr(graph, "edges", None) or {}
    boards = len({k[0] for k in edges})
    if not edges:
        return None
    parts = [f"GRAPH: {boards} boards, {len(edges)} real transitions recorded"]
    if fp is not None:
        try:
            untried = graph.untried(fp)   # a single action, or None
        except Exception:  # noqa: BLE001
            untried = None
        if untried:
            parts.append(f"this board has never tried {_short(untried)}")
        else:
            parts.append("every action has been tried from this board")
    return "; ".join(parts)


def build(goal_model=None, goal_policy=None, graph=None, grid=None,
          fp: Any = None, code: Optional[dict] = None,
          max_chars: int = 720) -> list[str]:
    """Return the INFERRED block as lines, or [] when no layer has an opinion.

    Every argument is optional: layers that are off simply contribute nothing.
    `code` is the perception layer's board description (scale and offset), which
    is what makes these numbers comparable with NATIVE.
    """
    conv = _Conv(code)
    try:
        h = len(grid) if grid else 0
        w = len(grid[0]) if grid and grid[0] else 0
        w_h = (max(1, h // max(1, conv.k)), max(1, w // max(1, conv.k)))
        mask = getattr(goal_model, "goal_mask", None) or frozenset()
        body = [
            _goal_line(goal_model, conv),
            _cover_line(goal_model, grid),
            _mover_line(goal_policy, mask, conv, w_h),
            _keys_line(goal_policy, conv),
            _delta_line(goal_model, goal_policy, mask, conv, w_h),
            _graph_line(graph, fp),
        ]
    except Exception:  # noqa: BLE001
        return []
    lines = [b for b in body if b]
    if not lines:
        return []
    kept: list[str] = []
    used = len(HEAD)
    for ln in lines:
        if used + len(ln) + 1 > max_chars:
            break
        kept.append(ln)
        used += len(ln) + 1
    return [HEAD] + kept if kept else []
