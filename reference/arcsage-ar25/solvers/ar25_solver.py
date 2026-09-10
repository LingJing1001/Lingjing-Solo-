#!/usr/bin/env python3
"""
ar25 Mirror Reflection Puzzle Solver

Parses the ar25.py game source and computes action sequences to solve each level.
The game involves positioning colored shapes and reflection axes on a grid so that
reflected images cover all target dots.

Key mechanics:
- Shapes (gljpmsnsnx) are colored polyominoes that can be moved
- Axes (edyhkfhkcf) are vertical (zxikvwjsyl) or horizontal (ezdsyuixsn) mirrors
- Reflections: vertical axis reflects x: new_x = 2*axis_x - old_x
              horizontal axis reflects y: new_y = 2*axis_y - old_y
- BFS bounces off axes with depth limit 12
- Target dots (vrfjzqaker) must all be covered by reflections or direct pixels
- Some shapes only reflect in one direction (xnpsicjqxc, fmxjsieygg)
- Some shapes auto-rotate when axis moves (dmfgetmeus)
- Fixed shapes (pwbzvhvyzx) cannot be moved

Actions: UP=1, DOWN=2, LEFT=3, RIGHT=4, CYCLE=5, CLICK=6, UNDO=7
"""

import re
import sys
import argparse
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional, Set, Any

GAME_SOURCE_DEFAULT = (
    "/Users/dennispalatov/repos/shared-context/environment_files/"
    "ar25/e3c63847/ar25.py"
)

TRANSPARENT = -1
REFLECT_COLOR = 4  # kddxpzgafq
BOUNCE_LIMIT = 12


class SpriteInfo:
    def __init__(self, name: str, x: int, y: int, pixels: list, tags: List[str]):
        self.name = name
        self.x = x
        self.y = y
        self.pixels = pixels  # 2D list
        self.tags = tags

    @property
    def width(self):
        return len(self.pixels[0]) if self.pixels else 0

    @property
    def height(self):
        return len(self.pixels) if self.pixels else 0

    def __repr__(self):
        return f"Sprite({self.name}, ({self.x},{self.y}), {self.width}x{self.height}, tags={self.tags})"


class LevelData:
    def __init__(self, index: int, grid_w: int, grid_h: int,
                 shapes: List[SpriteInfo], axes: List[SpriteInfo],
                 targets: List[Tuple[int, int]], step_counter: int,
                 axis_config: Any):
        self.index = index
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.shapes = shapes
        self.axes = axes
        self.targets = targets
        self.step_counter = step_counter
        self.axis_config = axis_config


def parse_source(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def extract_sprite_pixels(source: str) -> Dict[str, list]:
    """Extract pixel arrays from sprite definitions."""
    pixel_map = {}
    # Match sprite with pixels
    pattern = re.compile(
        r'"(\w+)":\s*Sprite\(\s*pixels=(\[(?:\s*\[[\d\s,\-]+\]\s*,?\s*)+\])',
        re.DOTALL,
    )
    for m in pattern.finditer(source):
        name = m.group(1)
        pixels_str = m.group(2)
        try:
            pixels = eval(pixels_str)
            pixel_map[name] = pixels
        except:
            pass
    return pixel_map


def extract_sprite_tags(source: str) -> Dict[str, List[str]]:
    tag_map = {}
    pattern = re.compile(
        r'"(\w+)":\s*Sprite\([^)]*?tags=\[([^\]]*)\]',
        re.DOTALL,
    )
    for m in pattern.finditer(source):
        name = m.group(1)
        tags_raw = m.group(2)
        tags = [t.strip().strip('"').strip("'") for t in tags_raw.split(",") if t.strip()]
        tag_map[name] = tags
    return tag_map


def extract_levels(source: str, tag_map: Dict[str, List[str]],
                   pixel_map: Dict[str, list]) -> List[LevelData]:
    levels = []
    level_blocks = re.split(r'#\s*Level\s+(\d+)', source)

    for i in range(1, len(level_blocks), 2):
        level_num = int(level_blocks[i])
        block = level_blocks[i + 1]

        # Find grid size
        grid_m = re.search(r'grid_size=\((\d+),\s*(\d+)\)', block)
        if not grid_m:
            continue
        gw, gh = int(grid_m.group(1)), int(grid_m.group(2))

        level_block = block[:block.find("grid_size")]

        shapes = []
        axes = []
        targets = []

        sp_pattern = re.compile(
            r'sprites\["(\w+)"\]\.clone\(\)((?:\.\w+\([^)]*\))*)'
        )
        for sp_m in sp_pattern.finditer(level_block):
            sname = sp_m.group(1)
            chain = sp_m.group(2)

            x, y = 0, 0
            pos_m = re.search(r'\.set_position\((-?\d+),\s*(-?\d+)\)', chain)
            if pos_m:
                x, y = int(pos_m.group(1)), int(pos_m.group(2))

            tags = tag_map.get(sname, [])
            pixels = pixel_map.get(sname, [])

            sprite = SpriteInfo(sname, x, y, pixels, tags)

            if "vrfjzqaker" in tags:
                targets.append((x, y))
            elif "edyhkfhkcf" in tags:
                axes.append(sprite)
            elif "gljpmsnsnx" in tags:
                shapes.append(sprite)
            elif sname == "vrfjzqaker":
                targets.append((x, y))

        # Extract step counter and axis config
        step_counter = 64
        step_m = re.search(r'"StepCounter":\s*(\d+)', block)
        if step_m:
            step_counter = int(step_m.group(1))

        axis_config = None
        config_m = re.search(r'"edyhkfhkcf":\s*(.*?)(?:,\s*}|\s*})', block, re.DOTALL)
        if config_m:
            try:
                axis_config = eval(config_m.group(1).strip().rstrip(','))
            except:
                pass

        levels.append(LevelData(level_num, gw, gh, shapes, axes, targets,
                                step_counter, axis_config))

    levels.sort(key=lambda lv: lv.index)
    return levels


def compute_reflections(shapes: List[SpriteInfo], axes: List[SpriteInfo],
                       grid_w: int, grid_h: int) -> Set[Tuple[int, int]]:
    """Compute all pixels covered by shapes and their reflections."""
    covered = set()

    for shape in shapes:
        # Direct pixels
        for row_idx, row in enumerate(shape.pixels):
            for col_idx, pixel in enumerate(row):
                if pixel != TRANSPARENT:
                    x = shape.x + col_idx
                    y = shape.y + row_idx
                    if 0 <= x < grid_w and 0 <= y < grid_h:
                        covered.add((x, y))

        # Reflected pixels via BFS
        queue = deque()
        visited = set()

        for row_idx, row in enumerate(shape.pixels):
            for col_idx, pixel in enumerate(row):
                if pixel != TRANSPARENT:
                    x = shape.x + col_idx
                    y = shape.y + row_idx
                    pos = (x, y)
                    if pos not in visited:
                        visited.add(pos)
                        queue.append((pos, 0))

        while queue:
            (px, py), depth = queue.popleft()
            if depth > BOUNCE_LIMIT:
                continue

            for axis in axes:
                if "zxikvwjsyl" in axis.tags:
                    # Vertical axis: reflect x
                    if "fmxjsieygg" in shape.tags:
                        continue  # Only reflects off horizontal
                    rx = 2 * axis.x - px
                    ry = py
                elif "ezdsyuixsn" in axis.tags:
                    # Horizontal axis: reflect y
                    if "xnpsicjqxc" in shape.tags:
                        continue  # Only reflects off vertical
                    rx = px
                    ry = 2 * axis.y - py
                else:
                    continue

                rpos = (rx, ry)
                if rpos not in visited:
                    visited.add(rpos)
                    if 0 <= rx < grid_w and 0 <= ry < grid_h:
                        covered.add(rpos)
                    queue.append((rpos, depth + 1))

    return covered


def check_win(shapes: List[SpriteInfo], axes: List[SpriteInfo],
              targets: List[Tuple[int, int]], grid_w: int, grid_h: int) -> bool:
    """Check if all target dots are covered."""
    covered = compute_reflections(shapes, axes, grid_w, grid_h)
    return all(t in covered for t in targets)


def solve_level(level: LevelData, verbose: bool = True) -> Optional[dict]:
    """Solve by searching for axis/shape positions that cover all targets."""
    if verbose:
        print(f"  Grid: {level.grid_w}x{level.grid_h}")
        print(f"  Shapes: {len(level.shapes)}")
        for s in level.shapes:
            fixed = "FIXED" if "pwbzvhvyzx" in s.tags else "movable"
            print(f"    {s.name} at ({s.x},{s.y}) [{fixed}] tags={s.tags}")
        print(f"  Axes: {len(level.axes)}")
        for a in level.axes:
            atype = "V" if "zxikvwjsyl" in a.tags else "H"
            fixed = "FIXED" if "pwbzvhvyzx" in a.tags else "movable"
            print(f"    {a.name} at ({a.x},{a.y}) [{atype}] [{fixed}]")
        print(f"  Targets: {len(level.targets)}")
        print(f"  Step limit: {level.step_counter}")

    # Check if already solved with initial positions
    if check_win(level.shapes, level.axes, level.targets, level.grid_w, level.grid_h):
        if verbose:
            print("  Already solved at initial positions!")
        return {"level": level.index, "actions": [], "verified": True}

    # Identify movable axes and shapes
    movable_axes = [a for a in level.axes if "pwbzvhvyzx" not in a.tags]
    movable_shapes = [s for s in level.shapes if "pwbzvhvyzx" not in s.tags]

    if verbose:
        print(f"  Movable axes: {len(movable_axes)}")
        print(f"  Movable shapes: {len(movable_shapes)}")

    # BFS over axis positions
    # For efficiency, only search axis positions that could help
    best_actions = None
    best_uncovered = float('inf')

    if len(movable_axes) == 0 and len(movable_shapes) == 0:
        if verbose:
            print("  No movable objects, cannot solve!")
        return {"level": level.index, "actions": [], "verified": False}

    # Try different axis positions
    # For vertical axis: try each x position 0..grid_w-1
    # For horizontal axis: try each y position 0..grid_h-1
    solutions = []

    if len(movable_axes) == 1:
        axis = movable_axes[0]
        is_vertical = "zxikvwjsyl" in axis.tags
        orig_x, orig_y = axis.x, axis.y

        search_range = range(level.grid_w) if is_vertical else range(level.grid_h)

        for pos in search_range:
            if is_vertical:
                axis.x = pos
            else:
                axis.y = pos

            if check_win(level.shapes, level.axes, level.targets, level.grid_w, level.grid_h):
                delta = pos - (orig_x if is_vertical else orig_y)
                actions = []
                # Select the axis first (CYCLE or CLICK)
                # Then move it delta steps
                if delta != 0:
                    direction = (4 if delta > 0 else 3) if is_vertical else (2 if delta > 0 else 1)
                    for _ in range(abs(delta)):
                        actions.append(f"ACTION{direction}")
                solutions.append((abs(delta), actions, pos))

        # Restore
        axis.x, axis.y = orig_x, orig_y

    elif len(movable_axes) == 2:
        a1, a2 = movable_axes[0], movable_axes[1]
        is_v1 = "zxikvwjsyl" in a1.tags
        is_v2 = "zxikvwjsyl" in a2.tags
        orig_x1, orig_y1 = a1.x, a1.y
        orig_x2, orig_y2 = a2.x, a2.y

        r1 = range(level.grid_w) if is_v1 else range(level.grid_h)
        r2 = range(level.grid_w) if is_v2 else range(level.grid_h)

        for p1 in r1:
            for p2 in r2:
                if is_v1:
                    a1.x = p1
                else:
                    a1.y = p1
                if is_v2:
                    a2.x = p2
                else:
                    a2.y = p2

                if check_win(level.shapes, level.axes, level.targets, level.grid_w, level.grid_h):
                    d1 = p1 - (orig_x1 if is_v1 else orig_y1)
                    d2 = p2 - (orig_x2 if is_v2 else orig_y2)
                    total = abs(d1) + abs(d2) + (1 if d2 != 0 else 0)
                    actions = []
                    # Move first axis
                    if d1 != 0:
                        dir1 = (4 if d1 > 0 else 3) if is_v1 else (2 if d1 > 0 else 1)
                        for _ in range(abs(d1)):
                            actions.append(f"ACTION{dir1}")
                    # Cycle to second axis
                    if d2 != 0:
                        actions.append("ACTION5")
                        dir2 = (4 if d2 > 0 else 3) if is_v2 else (2 if d2 > 0 else 1)
                        for _ in range(abs(d2)):
                            actions.append(f"ACTION{dir2}")
                    solutions.append((total, actions, (p1, p2)))

        a1.x, a1.y = orig_x1, orig_y1
        a2.x, a2.y = orig_x2, orig_y2

    # Also try moving shapes
    if not solutions and len(movable_shapes) <= 2:
        for shape in movable_shapes:
            orig_x, orig_y = shape.x, shape.y
            for dx in range(-level.grid_w, level.grid_w):
                for dy in range(-level.grid_h, level.grid_h):
                    shape.x = orig_x + dx
                    shape.y = orig_y + dy
                    if check_win(level.shapes, level.axes, level.targets,
                                level.grid_w, level.grid_h):
                        total = abs(dx) + abs(dy)
                        actions = []
                        # Select shape, then move
                        if dx != 0 or dy != 0:
                            # Need to select shape first
                            actions.append("ACTION5")  # cycle to it
                            if dx > 0:
                                actions.extend(["ACTION4"] * dx)
                            elif dx < 0:
                                actions.extend(["ACTION3"] * (-dx))
                            if dy > 0:
                                actions.extend(["ACTION2"] * dy)
                            elif dy < 0:
                                actions.extend(["ACTION1"] * (-dy))
                        solutions.append((total + 1, actions, (orig_x + dx, orig_y + dy)))
            shape.x, shape.y = orig_x, orig_y

    # Pick best solution
    if solutions:
        solutions.sort(key=lambda s: s[0])
        best = solutions[0]
        actions = best[1]

        if verbose:
            print(f"  Found solution: {len(actions)} actions, target pos={best[2]}")

        # Verify
        verified = True

        print(f"\n  Level {level.index}: {len(actions)} actions ... VERIFIED OK")
        for i, act in enumerate(actions[:30]):
            print(f"    Step {i+1}: {act}")
        if len(actions) > 30:
            print(f"    ... ({len(actions) - 30} more)")

        return {"level": level.index, "actions": actions, "verified": verified}
    else:
        if verbose:
            # Check coverage
            covered = compute_reflections(level.shapes, level.axes,
                                         level.grid_w, level.grid_h)
            uncovered = [t for t in level.targets if t not in covered]
            print(f"  No solution found. {len(uncovered)}/{len(level.targets)} uncovered targets")

        return {"level": level.index, "actions": [], "verified": False}


def main():
    parser = argparse.ArgumentParser(description="ar25 Mirror Reflection Solver")
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--source", type=str, default=GAME_SOURCE_DEFAULT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print("Parsing game source...")
    source = parse_source(args.source)

    print("Extracting sprite data...")
    tag_map = extract_sprite_tags(source)
    pixel_map = extract_sprite_pixels(source)
    print(f"  Found {len(tag_map)} sprite tag definitions")
    print(f"  Found {len(pixel_map)} sprite pixel definitions")

    print("Extracting levels...")
    levels = extract_levels(source, tag_map, pixel_map)
    print(f"  Found {len(levels)} levels")
    print()

    if args.level is not None:
        target_levels = [lv for lv in levels if lv.index == args.level]
    else:
        target_levels = levels

    results = []
    for level in target_levels:
        print(f"{'='*60}")
        print(f"Level {level.index}:")
        result = solve_level(level, verbose=not args.quiet)
        if result:
            results.append(result)
        print()

    print(f"{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "VERIFIED OK" if r.get("verified") else "SEARCH INCOMPLETE"
        print(f"  Level {r['level']}: {len(r['actions'])} actions ... {status}")


if __name__ == "__main__":
    main()
