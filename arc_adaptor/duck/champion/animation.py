"""Animation awareness.

``arcengine`` renders a frame after every internal ``step()``, so a single
action can come back as a short animation. ``GameState.frame`` is only the last
of those, and the harness consumed nothing else - so every intermediate frame
was discarded before the agent could see it. On some games the whole effect of
an action lives there: the board before and the board after are identical, and
reading the final frame cannot possibly reveal what happened.

Two shapes occur. Where cells change and change BACK, the information is
genuinely hidden and worth spending prompt tokens on. Where every cell changes
at most once, the frames are motion interpolation toward a state the final
frame already shows, and there is nothing to find - those animations are
reported nowhere, because a line about them would cost tokens to say that
nothing is missing.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from .grid_utils import ARC_COLOR_CHARS, format_grid_ascii

Grid = tuple[tuple[int, ...], ...]

# Retrieval budget.
#
# Why a diff timeline rather than the frames themselves: one 64x64 grid renders
# as roughly 4,100 characters, and a single action can return dozens of frames.
# Whole frames are never affordable against a ~1,024-token tool budget, so
# retrieval collapses identical frames and reports only the cells that changed
# between the rest. A single frame can still be read verbatim, cropped.
# No step limit: the timeline crosses the pipe as data and the model prints
# only what it chooses, so dropping steps served a prompt budget that no longer
# applies. It also cost a real misreading - a step count reported as omitted
# was read as "those steps had no changes", the exact inverse.
ANIMATION_MAX_CELLS_PER_STEP = 24
ANIMATION_MAX_TOTAL_CELLS = 80


def normalize_frames(raw_frames: Any) -> list[Grid]:
    """Engine frame list to hashable, comparable grids."""
    frames: list[Grid] = []
    for frame in raw_frames or ():
        rows = frame.tolist() if hasattr(frame, "tolist") else frame
        frames.append(tuple(tuple(int(cell) for cell in row) for row in rows or ()))
    return frames


def collapse_frames(before: Grid | None, frames: Sequence[Grid]) -> list[Grid]:
    """The pre-action board, then each distinct state the action passed through.

    Runs of identical grids collapse to one entry; non-adjacent repeats stay
    separate, because a grid at positions 2 and 7 with something between them
    is two distinct moments and the timeline should show both transitions.

    ``before`` is prepended and is NOT optional. A cell showing X before the
    action, Y during it and X again at the end arrives from the engine as
    [Y, Y, X] - one transition, which would not register as transient. With the
    board before it is [X, Y, Y, X], which gives two. Without this the very
    case the feature exists for is invisible.
    """
    collapsed: list[Grid] = []
    if before is not None:
        collapsed.append(before)
    for grid in frames:
        if collapsed and collapsed[-1] == grid:
            continue
        collapsed.append(grid)
    return collapsed


def _transient_cells(chain: Sequence[Grid]) -> list[tuple[int, int]]:
    """Cells whose value changes two or more times across the chain.

    A cell that changes once and stays changed is visible in the final frame,
    so the agent loses nothing by never seeing the frames between. A cell that
    changes twice has shown something that is no longer there - a rejected
    click, a consumed attempt, a sprite crossing a background cell. Those are
    exactly the pixels no frame the agent can reach will ever contain.
    """
    changes: defaultdict[tuple[int, int], int] = defaultdict(int)
    for before, after in zip(chain, chain[1:]):
        rows = max(len(before), len(after))
        for row in range(rows):
            before_row = before[row] if row < len(before) else ()
            after_row = after[row] if row < len(after) else ()
            for col in range(max(len(before_row), len(after_row))):
                old = before_row[col] if col < len(before_row) else None
                new = after_row[col] if col < len(after_row) else None
                if old != new:
                    changes[(row, col)] += 1
    return sorted(cell for cell, count in changes.items() if count >= 2)


def summarize_animation(chain: Sequence[Grid]) -> dict[str, Any] | None:
    """Compact metadata for one action, or ``None`` if it did not animate.

    ``None`` for the single-frame case means ordinary actions carry no extra
    tokens at all - the key is simply absent from the action result.

    ``frames`` counts engine frames after collapsing and excludes the prepended
    pre-action board, so it answers "how many distinct states did this action
    pass through" rather than describing the internal representation.
    """
    if len(chain) <= 2:
        return None
    transient = _transient_cells(chain)
    summary: dict[str, Any] = {
        "frames": len(chain) - 1,
        "transient_pixels": len(transient),
    }
    if transient:
        rows = [cell[0] for cell in transient]
        cols = [cell[1] for cell in transient]
        # Inclusive [top, left, bottom, right]: enough to point at the region
        # without spending tokens on a coordinate list.
        summary["transient_bbox"] = [min(rows), min(cols), max(rows), max(cols)]
    return summary


def _bbox_phrase(bbox: Sequence[int] | None) -> str:
    """Where the transient cells are, collapsing degenerate ranges.

    A single cell reads as "at (20,30)", a single row or column as a span in
    the other axis only; "rows 20-20, cols 30-30" is noise the model has to
    parse before noticing it means one cell.
    """
    if not bbox:
        return ""
    top, left, bottom, right = bbox
    if top == bottom and left == right:
        return f" at ({top},{left})"
    if top == bottom:
        return f" in row {top}, cols {left}-{right}"
    if left == right:
        return f" in col {left}, rows {top}-{bottom}"
    return f" around rows {top}-{bottom}, cols {left}-{right}"


def describe_animation(
    summary: dict[str, Any] | None,
    *,
    action_display: str | None,
    gameplay_changed: bool | None,
    game_over: bool = False,
    timeline: bool = True,
) -> str:
    """The opener line, or "" when there is nothing worth saying.

    Silent unless cells changed and changed back. Where every cell changed at
    most once the final frame already shows the outcome, and a line saying so
    would spend tokens to report that nothing is missing.
    """
    if not summary:
        return ""
    transient = int(summary.get("transient_pixels") or 0)
    if transient <= 0:
        return ""
    frames = summary.get("frames")
    cells = "1 cell changed and changed back" if transient == 1 else (
        f"{transient} cells changed and changed back"
    )
    where = _bbox_phrase(summary.get("transient_bbox"))
    name = f" (`{action_display}`)" if action_display else ""
    # point at whichever accessor is actually available
    where_to_look = (
        "`last_animation_timeline`"
        if timeline
        else "`last_animation_frames`"
    )

    if game_over:
        # Follows the BAR RULE block, which has just walked the model through
        # history[-3] and history[-2]. This says what those two frames cannot
        # contain, which is precisely what "transient" means.
        return (
            f"That fatal action animated over {frames} frames, and inside it "
            f"{cells}{where}. Those cells are in neither "
            f"`history[-3].frame` nor `history[-2].frame` - read {where_to_look} to see "
            "what happened between them."
        )
    if gameplay_changed is False:
        return (
            f"The last executed action{name} animated over {frames} frames and left the "
            f"board area unchanged, but it was not a no-op: {cells}{where}. "
            "Whatever this action did is visible only there - "
            f"read {where_to_look} before concluding it failed."
        )
    return (
        f"The last executed action{name} animated over {frames} frames. "
        f"{cells[0].upper()}{cells[1:]}{where}. {where_to_look} shows them."
    )


def _color_char(value: int | None) -> str:
    if value is None:
        return "?"
    return ARC_COLOR_CHARS[max(0, min(15, int(value)))]


def _cell_at(grid: Grid, row: int, col: int) -> int | None:
    if row >= len(grid):
        return None
    line = grid[row]
    return line[col] if col < len(line) else None


def _diff_cells(before: Grid, after: Grid) -> list[tuple[int, int, int | None, int | None]]:
    rows = max(len(before), len(after))
    changes: list[tuple[int, int, int | None, int | None]] = []
    for row in range(rows):
        cols = max(
            len(before[row]) if row < len(before) else 0,
            len(after[row]) if row < len(after) else 0,
        )
        for col in range(cols):
            old = _cell_at(before, row, col)
            new = _cell_at(after, row, col)
            if old != new:
                changes.append((row, col, old, new))
    return changes


def _bbox_text(cells: Sequence[tuple[int, int, Any, Any]]) -> str:
    rows = [cell[0] for cell in cells]
    cols = [cell[1] for cell in cells]
    return f"rows {min(rows)}-{max(rows)}, cols {min(cols)}-{max(cols)}"


def _format_changes(
    cells: Sequence[tuple[int, int, int | None, int | None]], budget: int
) -> list[str]:
    """Changed cells grouped by colour transition, one compact line each.

    Strings rather than nested coordinate lists because the caller renders with
    json.dumps(indent=2), which would put every [row, col] pair on four lines.
    """
    grouped: dict[str, list[str]] = {}
    for row, col, old, new in cells[:budget]:
        key = f"{_color_char(old)}>{_color_char(new)}"
        grouped.setdefault(key, []).append(f"({row},{col})")
    lines = [f"{key} @ {' '.join(points)}" for key, points in grouped.items()]
    if len(cells) > budget:
        lines.append(f"... {len(cells) - budget} further changed cells omitted")
    return lines


def build_animation_frames(record: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Every distinct state the action passed through, as FrameView payloads.

    The pre-action board is dropped here: the model already has it as
    `previous_frame`, and including it would make index 0 mean something
    different from the step numbers in the timeline.

    Sends the grid as well as the ascii so `segmentation` works on the far
    side. That doubles the payload, but it crosses a pipe rather than the
    prompt - what reaches the model is only what its own code prints.
    """
    if not record:
        return []
    chain = list(record.get("chain") or ())
    if len(chain) <= 2:
        return []
    step = record.get("step") or 0
    level = record.get("level") or 0
    return [
        {
            "ascii": format_grid_ascii(grid),
            "step": step,
            "level": level,
            "shape": (len(grid), max((len(row) for row in grid), default=0)),
            "grid": [list(row) for row in grid],
        }
        for grid in chain[1:]
    ]


def build_animation_view(record: dict[str, Any] | None) -> dict[str, Any]:
    """The diff timeline: which cells changed at each step.

    The cheap first look. Reading a frame verbatim is no longer this
    function's job - `last_transition.frames()` hands back FrameView objects
    and the model crops their ascii the same way it crops current_frame, with
    its own helpers rather than a region parameter it has to guess.
    """
    empty = {
        "error": (
            "No animation is available: the last executed action returned a single frame."
        )
    }
    if not record:
        return empty
    chain: list[Grid] = list(record.get("chain") or ())
    if len(chain) <= 2:
        return empty

    header: dict[str, Any] = {
        "action": record.get("action_display"),
        # excludes the prepended pre-action board, matching the metadata
        "frames": len(chain) - 1,
    }

    steps: list[dict[str, Any]] = []
    for position in range(1, len(chain)):
        cells = _diff_cells(chain[position - 1], chain[position])
        if cells:
            steps.append({"_position": position, "_cells": cells})

    rendered: list[dict[str, Any]] = []
    remaining = ANIMATION_MAX_TOTAL_CELLS
    for step in steps:
        cells = step["_cells"]
        entry: dict[str, Any] = {
            "step": step["_position"],
            "changed": len(cells),
            "bbox": _bbox_text(cells),
        }
        budget = min(ANIMATION_MAX_CELLS_PER_STEP, max(0, remaining))
        if len(cells) > ANIMATION_MAX_CELLS_PER_STEP or budget == 0:
            # Too broad to enumerate usefully: a colour-transition census says
            # more per token than an arbitrarily truncated cell list.
            census: dict[str, int] = {}
            for _, _, old, new in cells:
                key = f"{_color_char(old)}>{_color_char(new)}"
                census[key] = census.get(key, 0) + 1
            entry["transitions"] = census
        else:
            entry["changes"] = _format_changes(cells, budget)
            remaining -= min(len(cells), budget)
        rendered.append(entry)

    return {**header, "steps": rendered}
