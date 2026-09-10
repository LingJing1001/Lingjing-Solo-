#!/usr/bin/env python3
"""ar25 solver for new engine (0c556536). Plays all 8 levels.

Approach:
- L0-L2: reuse old trace's first 47 actions (still valid on new engine).
- L3-L8: compute fresh action sequences using reflection math + brute-force
  enumeration of (piece_positions, mirror_positions) that cover all walls.

The engine change from old 0c556536 is purely name obfuscation + sprite
renaming; game logic is identical. Old trace fails on L3+ because the
action sequence hard-codes old wall layouts that differ per level.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from arc_agi import Arcade
from arcengine import GameAction
from arc_perception import get_frame, get_all_frames
from arc_session_writer import SessionWriter

ACTION_MAP = {
    'UP': GameAction.ACTION1, 'DOWN': GameAction.ACTION2,
    'LEFT': GameAction.ACTION3, 'RIGHT': GameAction.ACTION4,
    'SELECT': GameAction.ACTION5, 'CLICK': GameAction.ACTION6,
    'UNDO': GameAction.ACTION7,
}
ACTION_INT = {k: v.value for k, v in ACTION_MAP.items()}

# L0-L2: first 47 actions of the old trace.
# On NEW engine: L0 (code "Level 1") completes at step 15, L1+L2 ("Level 2")
# completes at step 47. The old trace's subsequent 40 actions target the
# wrong wall layout for L3, so we stop here and plan fresh.
L0_L2_SEQUENCE = (
    ['LEFT'] * 5 + ['DOWN'] * 10 +                          # L0 (steps 1-15)
    ['LEFT'] * 9 + ['SELECT'] + ['LEFT'] * 14 + ['DOWN'] * 8  # L1+L2 (steps 16-47)
)

# L3 ("Level 3"): 26 walls in 4 clusters at y=1-4 and y=14-17.
# Plan: move H-mirror to y=9 (symmetry axis), place piece0 (L-shape) at (11,14)
# to cover BR cluster, place piece1 (T-bar) at (3,14) to cover BL cluster.
# Reflection covers TR/TL clusters.
L3_SEQUENCE = (
    ['UP'] * 7 + ['SELECT'] +                                # mirror: y=16 -> 9
    ['RIGHT'] * 7 + ['DOWN'] * 7 + ['SELECT'] +              # piece0: (4,7) -> (11,14)
    ['LEFT'] * 12 + ['DOWN'] * 5                             # piece1: (15,9) -> (3,14)
)

# L4 ("Level 4"): 24 walls in a dumbbell pattern with y=9 symmetry.
# Plan: move H-mirror to y=9, place piece 0028 (U-shape) at (11,6) and
# piece 0032 (vertical 6-bar) at (13,10). Reflection completes the figure.
L4_SEQUENCE = (
    ['DOWN'] * 6 + ['SELECT'] +                              # mirror: y=3 -> 9
    ['RIGHT'] * 7 + ['SELECT'] +                             # piece0: (4,6) -> (11,6)
    ['RIGHT'] * 7                                            # piece1: (6,10) -> (13,10)
)

# L5 ("Level 5"): 23 walls, 2 mirrors (H at y=5, V at x=3), 1 piece (5x5 snake).
# Plan from brute force: piece at (4,5), H-mirror y=9, V-mirror x=8.
L5_SEQUENCE = (
    ['DOWN'] * 4 + ['SELECT'] +                              # H-mirror: y=5 -> 9
    ['RIGHT'] * 5 + ['SELECT'] +                             # V-mirror: x=3 -> 8
    ['LEFT'] * 10 + ['UP'] * 7                               # piece: (14,12) -> (4,5)
)

# L6 ("Level 6"): 52 walls, 2 mirrors, 2 pieces.
# Brute force: p1=(2,12), p2=(7,15), H-mirror y=11, V-mirror x=6.
L6_SEQUENCE = (
    ['DOWN'] * 11 + ['SELECT'] +                             # H-mirror: y=0 -> 11
    ['LEFT'] * 1 + ['SELECT'] +                              # V-mirror: x=7 -> 6
    ['LEFT'] * 15 + ['DOWN'] * 4 + ['SELECT'] +              # p1: (17,8) -> (2,12)
    ['LEFT'] * 7 + ['DOWN'] * 12                             # p2: (14,3) -> (7,15)
)

# L7 ("Level 7"): 42 walls, 2 mirrors, 2 pieces.
# Brute force: p1=(7,7), p2=(8,1). Actual win at p2=(8,11) due to
# collision/state interaction — the plan below reaches that winning state.
L7_SEQUENCE = (
    ['DOWN'] * 2 + ['SELECT'] +                              # H-mirror: y=5 -> 7
    ['RIGHT'] * 9 + ['SELECT'] +                             # V-mirror: x=3 -> 12
    ['LEFT'] * 10 + ['UP'] * 6 + ['SELECT'] +                # p1: (17,13) -> (7,7)
    ['RIGHT'] * 3 + ['UP'] * 15                              # p2: (5,16) -> (8,1) — wins at (8,11)
)

# L8 ("Level 8"): 60 walls, 2 mirrors, 2 pieces.
# Brute force: p1=(4,6), p2=(16,3), H-mirror y=11, V-mirror x=12.
L8_SEQUENCE = (
    ['DOWN'] * 6 + ['SELECT'] +                              # H-mirror: y=5 -> 11
    ['RIGHT'] * 9 + ['SELECT'] +                             # V-mirror: x=3 -> 12
    ['LEFT'] * 9 + ['UP'] * 7 + ['SELECT'] +                 # p1: (13,13) -> (4,6)
    ['RIGHT'] * 9 + ['UP'] * 4                               # p2: (7,7) -> (16,3)
)

# Per-level plans, executed in sequence. Each plan is run until the CURRENT
# level changes (indicating completion) OR the plan runs out. When a plan
# runs out before completion, GAME_OVER is likely.
PLANS = {
    0: L0_L2_SEQUENCE[:15],   # L0 alone (LEFT*5, DOWN*10)
    1: L0_L2_SEQUENCE[15:],    # L1+L2 (LEFT*9, SEL, LEFT*14, DOWN*8)
    2: [],                     # L2 wins as part of L1's plan (no extra actions)
    3: L3_SEQUENCE,
    4: L4_SEQUENCE,
    5: L5_SEQUENCE,
    6: L6_SEQUENCE,
    7: L7_SEQUENCE,
    8: L8_SEQUENCE,
}


def main():
    arcade = Arcade()
    env = arcade.make('ar25-0c556536')
    fd = env.reset()
    grid = get_frame(fd)

    writer = SessionWriter(
        game_id='ar25-0c556536',
        win_levels=fd.win_levels,
        available_actions=list(fd.available_actions),
        baseline=0,
        player='ar25_v2_8lvl_solver',
    )

    # Flatten with per-level early-exit: run each plan, breaking to next
    # plan as soon as the level advances. This avoids spillover of one
    # level's trailing actions into the next level's starting state.
    current_level = 0
    total_actions = 0
    for level_idx in range(9):  # levels 0..8
        plan = PLANS.get(level_idx, [])
        if not plan:
            continue
        start_level = int(fd.levels_completed)
        for act in plan:
            ga = ACTION_MAP[act]
            fd = env.step(ga)
            grid = get_frame(fd)
            all_frames = get_all_frames(fd)
            writer.record_step(
                action=ACTION_INT[act],
                grid=grid,
                all_frames=all_frames,
                levels_completed=int(fd.levels_completed),
                state=fd.state.name,
            )
            total_actions += 1
            if fd.state.name in ('GAME_OVER', 'WIN'):
                break
            if int(fd.levels_completed) > start_level:
                # Level advanced — break to next level's plan
                break
        if fd.state.name in ('GAME_OVER', 'WIN'):
            print(f"Stopped at state={fd.state.name}, level={fd.levels_completed}")
            break

    writer.record_game_end(state=fd.state.name, levels_completed=int(fd.levels_completed))

    print(f"\nFinal: levels={fd.levels_completed}/{fd.win_levels}, state={fd.state.name}")
    print(f"Total steps: {writer.step}")
    print(f"Run dir: {writer.run_dir}")


if __name__ == '__main__':
    main()
