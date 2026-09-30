"""VC33-5430563c offline runner: R2 perception, then two tiers of replay BFS.

Why this file lives in the repo (2026-09-28): the original scratch runner was never
committed and only survived in an ad-hoc backup snapshot, so its results were not
reproducible. This is the tracked entry point.

Search tiers, tried in order for each level:

  1. `combo.r3_search_click_path` — shortest click path, keyed by picture only. Cheap,
     and it clears the shallow levels (L1 in 3 clicks, L2 in 7).
  2. `closure_bfs` (this file) — cap-free breadth-first closure over reachable pictures.
     No per-button click cap, no ordering assumption; it stops only when the frontier is
     genuinely empty or its time budget runs out.

Two earlier designs were measured and dropped; the reasons are recorded here so nobody
re-tries them:

  * `r4_explore` fallback (the 2026-09-17 runs): clicked the least-used button until the
    level's 64-step bar hit zero, self-inflicting GAME_OVER at 85 clicks and polluting
    the evidence. Removed — a level no search can solve now stops honestly instead.
  * `budget_aware_bfs` (tried 2026-09-28): kept the remaining step budget inside the
    state key, on the theory that VC33 might hide a counter in its top budget row. It
    made L3 *worse* — 756 pictures visited versus 522 for the picture-only key — because
    the picture already changes on every click, so budget is not a hidden dimension.
    The step budget is still honoured, but as a per-node depth cap, not as a state key.

Dated finding that motivated tier 2 (2026-09-29): an enumeration built on saturation
caps measured at the level-entry state (3887 click multisets, zero solutions) was a
FALSE NEGATIVE. VC33 buttons re-charge one another, so a button measured as "one click
then no-op" may need five clicks in a real solution — L3 needs (12,56) x5 and (46,56) x9
against entry-state caps of 1 and 8. Only the cap-free closure found L3, at 23 clicks.
Consequently this runner must never phrase an "unsolvable" claim unless `closure_bfs`
returns `frontier_exhausted`; `time_limit` and `state_limit` mean "unknown".
"""
import argparse
from collections import deque
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import r2r3r4_combo as combo

GAME = 'vc33-5430563c'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environments-dir', required=True)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parents[3] / 'state')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--search-seconds', type=float, default=15,
                        help='deadline for each shortest-path BFS attempt')
    parser.add_argument('--closure-seconds', type=float, default=360,
                        help='deadline for the cap-free closure BFS fallback')
    parser.add_argument('--max-depth', type=int, default=64,
                        help='click cap per level; VC33 grants 64 clicks per level')
    parser.add_argument('--max-states', type=int, default=20000)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--max-seconds', type=float, default=900)
    args = parser.parse_args()
    for key in ('search_seconds', 'closure_seconds', 'max_depth', 'max_states',
                'max_steps', 'max_seconds'):
        if getattr(args, key) <= 0:
            parser.error('budgets must be positive')

    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction

    output = args.output_dir / ('vc33_r234_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger('vc33_r234')
    logger.setLevel(logging.WARNING)
    arc = Arcade(environments_dir=args.environments_dir, logger=logger,
                 recordings_dir=str(output / 'recordings'),
                 operation_mode=OperationMode.OFFLINE)
    env = None

    def new_env():
        nonlocal env
        # Measured 2026-09-28: env.reset() rewinds to level 1 (levels_completed 2 -> 0)
        # and the SDK exposes no state snapshot, so a fresh instance plus a replayed
        # prefix is the only way to reach any given search node.
        env = arc.make(GAME, seed=args.seed, save_recording=False)
        return env.reset()

    def step(action):
        if action['action'] != 6:
            raise ValueError('VC33 only accepts ACTION6')
        data = action['data']
        if any(not isinstance(data[k], int) or not 0 <= data[k] < 64 for k in ('x', 'y')):
            raise ValueError('Click coordinate outside 64x64 display')
        return env.step(GameAction.ACTION6, data=data)

    def replay(path):
        """Frame reached by replaying `path` (click dicts) from a fresh level-1 env."""
        frame = new_env()
        for action in path:
            frame = step(action)
            if frame.state.name in ('WIN', 'GAME_OVER'):
                return frame
        return frame

    def grid_of(frame):
        return frame.frame[-1].tolist()

    def candidates(frame):
        return combo.vc33_click_candidates(grid_of(frame))

    def picture(frame):
        # Row zero is the step bar, not puzzle content, so it stays out of the key.
        return combo.r2_state_hash(grid_of(frame)[1:])

    def budget(frame):
        # Top row is the shrinking step bar: cells equal to its left end = clicks left.
        row = list(frame.frame[-1][0])
        return int(sum(1 for cell in row if cell == row[0])) if row else 0

    def state_key(frame):
        return (int(frame.levels_completed), picture(frame))

    def advanced(frame, base_level):
        return frame.state.name == 'WIN' or int(frame.levels_completed) > base_level

    def closure_bfs(base_path, deadline_at, max_states):
        """Cap-free BFS over the pictures reachable within one level.

        Each node branches on the buttons actually on screen in that node, keeps no
        per-button click limit, and never enqueues a branch that dies or has run out of
        step budget — so a negative result cannot be an artefact of self-inflicted
        GAME_OVER. `frontier_exhausted` means the closure was enumerated.
        """
        entry = replay(base_path)
        blank = {'path': None, 'visited': 1, 'expanded': 0, 'children': 0,
                 'leaves_cut_by_depth_cap': 0, 'seconds': 0.0}
        if entry.state.name in ('WIN', 'GAME_OVER'):
            return {'reason': 'terminal_initial_state', **blank}
        base_level = int(entry.levels_completed)
        queue = deque([[]])
        visited = {state_key(entry)}
        expanded = children = truncated = 0
        start = time.monotonic()

        def result(reason, path=None):
            return {'reason': reason, 'path': path, 'visited': len(visited),
                    'expanded': expanded, 'children': children,
                    'leaves_cut_by_depth_cap': truncated,
                    'seconds': round(time.monotonic() - start, 3)}

        while queue:
            if time.monotonic() >= deadline_at:
                return result('time_limit')
            if len(visited) >= max_states:
                return result('state_limit')
            path = queue.popleft()
            node = replay(base_path + path)
            if node.state.name in ('WIN', 'GAME_OVER') or budget(node) <= 0:
                continue
            # The 64-click bar is respected as a depth cap, never as a state-key field.
            if len(path) >= args.max_depth or len(path) >= budget(node):
                truncated += 1
                continue
            for action in candidates(node):
                if time.monotonic() >= deadline_at:
                    return result('time_limit')
                child = replay(base_path + path + [action])
                children += 1
                if advanced(child, base_level):
                    return result('solved', path + [action])
                if child.state.name == 'GAME_OVER' or budget(child) <= 0:
                    continue
                key = state_key(child)
                if key in visited:
                    continue
                visited.add(key)
                queue.append(path + [action])
            expanded += 1
        # The queue drained: every reachable picture was enumerated. This is the only
        # reason below that licenses an "unsolvable" claim.
        return result('frontier_exhausted')

    prefix = []
    traces = []
    searches = []
    started = time.monotonic()
    frame = replay(prefix)
    stop = 'step_limit'

    while len(prefix) < args.max_steps and frame.state.name not in ('WIN', 'GAME_OVER'):
        if args.max_seconds - (time.monotonic() - started) <= 0:
            stop = 'time_limit'
            break
        base_level = int(frame.levels_completed)

        # 1. Shortest-path BFS (picture-only key). Cheap; clears the shallow levels.
        search = combo.r3_search_click_path(
            lambda: replay(prefix), step, candidates, state_key,
            max_seconds=min(args.search_seconds,
                            args.max_seconds - (time.monotonic() - started)),
            max_depth=min(args.max_depth, args.max_steps - len(prefix)),
            max_states=args.max_states)
        searches.append({'level': base_level + 1, 'flow': 'shortest_bfs', **search})
        print(f'L{base_level+1} shortest: {search["reason"]}, '
              f'states={search.get("states")}, seconds={search["seconds"]}', flush=True)
        path = search['path']
        solver = 'r3_bfs'

        # 2. Cap-free closure BFS.
        if not path:
            remaining = args.max_seconds - (time.monotonic() - started)
            if remaining <= 0:
                stop = 'time_limit'
                break
            csearch = closure_bfs(prefix,
                                  time.monotonic() + min(args.closure_seconds, remaining),
                                  args.max_states)
            searches.append({'level': base_level + 1, 'flow': 'closure_bfs', **csearch})
            print(f'L{base_level+1} closure:   {csearch["reason"]}, '
                  f'visited={csearch.get("visited")}, children={csearch.get("children")}, '
                  f'seconds={csearch.get("seconds")}', flush=True)
            path = csearch['path']
            solver = 'closure_bfs'
            if not path:
                # Only an emptied frontier proves unreachability; anything else is unknown.
                stop = ('no_path_frontier_exhausted'
                        if csearch['reason'] == 'frontier_exhausted'
                        else 'search_budget_exhausted')
                break

        frame = replay(prefix)
        combo.reset_smart_play()
        for action in path:
            if (int(frame.levels_completed) != base_level
                    or frame.state.name in ('WIN', 'GAME_OVER')):
                break
            if time.monotonic() - started >= args.max_seconds:
                stop = 'time_limit'
                break
            frame = step(action)
            prefix.append(action)
            traces.append({'step': len(prefix), 'level_before': base_level + 1,
                           'method': solver, **action, 'state': frame.state.name,
                           'levels_completed': int(frame.levels_completed)})

        print(f'progress: {frame.levels_completed}/{frame.win_levels}, steps={len(prefix)}, '
              f'state={frame.state.name}, budget={budget(frame)}', flush=True)
        if int(frame.levels_completed) == base_level and frame.state.name not in ('WIN', 'GAME_OVER'):
            stop = 'committed_path_stalled'
            break

    if frame.state.name in ('WIN', 'GAME_OVER'):
        stop = frame.state.name

    observed = {'state': frame.state.name, 'levels_completed': int(frame.levels_completed)}
    # Independent replay verifies the executed route, not the search's speculative branches.
    new_env()
    verified = None
    for action in prefix:
        verified = step(action)
    if not prefix:
        replay_matches = True
    else:
        replay_matches = (verified.state.name == observed['state'] and
                          int(verified.levels_completed) == observed['levels_completed'])

    result = {
        'game': GAME, 'mode': 'OFFLINE', 'seed': args.seed,
        'budgets': {k: v for k, v in vars(args).items()
                    if k not in ('output_dir', 'environments_dir')},
        **observed, 'win_levels': int(frame.win_levels),
        'executed_steps': len(prefix), 'stop_reason': stop,
        'elapsed_seconds': round(time.monotonic() - started, 3),
        'replay_matches': replay_matches, 'final_step_budget': budget(frame),
        # A solvability statement is only made when the closure frontier actually emptied.
        'unsolvable_claimed': stop == 'no_path_frontier_exhausted',
        'search_flows_used': sorted({s['flow'] for s in searches}),
        'searches': searches, 'actions': traces,
    }
    (output / 'result.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('searches', 'actions')}, ensure_ascii=False, indent=2))
    print(f'RESULT_FILE={output / "result.json"}')
    if not replay_matches:
        raise RuntimeError('Independent replay did not match execution')


if __name__ == '__main__':
    main()
