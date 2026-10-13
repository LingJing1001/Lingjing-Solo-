#!/usr/bin/env python3
"""Collect balanced AOP counterfactual shards without training.

The collector is intentionally separate from training. Each invocation creates
an immutable shard and updates collection/state.json. Training consumes a
frozen dataset manifest only after collection quotas pass.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter

ROOT = Path('/srv/agent-platform/projects/Lingjing-Solo-')
BASE_LABELS = ROOT / 'data/aop/continuous/labels.jsonl'
OUT = ROOT / 'data/aop/collection'
STATE = OUT / 'state.json'
LOCK = OUT / '.lock'
TARGET_PER_CELL = 500
# vc33 exposes only ACTION6 at reset and cannot form a two-branch
# counterfactual group. Its existing ordinary trajectories remain in the base
# dataset; it is excluded from this counterfactual quota until a dedicated
# single-action collector is implemented.
GAMES = ['ar25', 'ls20', 'dc22', 're86']
ACTIONS = [f'ACTION{i}' for i in range(1, 8)]
SUPPORTED_ACTIONS = {
    'ar25': set(ACTIONS),
    'ls20': {f'ACTION{i}' for i in range(1, 5)},
    'dc22': {'ACTION1', 'ACTION2', 'ACTION3', 'ACTION4', 'ACTION6'},
    're86': {f'ACTION{i}' for i in range(1, 6)},
}
# Coordinates are only trusted where they were explicitly verified previously.
ACTION6_COORDS = {'ar25': (15, 31)}


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]


def counts() -> Counter[tuple[str, str]]:
    result: Counter[tuple[str, str]] = Counter()
    paths = [BASE_LABELS] + list(OUT.glob('shards/*/counterfactual_labels.jsonl'))
    for path in paths:
        for row in read_rows(path):
            episode = str(row.get('episode_id', ''))
            game = episode.split('_', 1)[0]
            action = str((row.get('requested_action') or {}).get('name', ''))
            if game in GAMES and action in ACTIONS:
                result[(game, action)] += 1
    return result


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def shard_status(added: int, failure_count: int) -> str:
    if added:
        return 'shard_complete'
    if failure_count >= 2:
        return 'no_rows_target_quarantined'
    return 'no_rows_retry_with_new_target'


def write_state(payload: dict) -> None:
    STATE.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        print(json.dumps({'status': 'already_running', 'lock': str(LOCK)}))
        return 0
    LOCK.write_text(str(os.getpid()), encoding='utf-8')
    try:
        previous = json.loads(STATE.read_text()) if STATE.exists() else {}
        failures = dict(previous.get('failures', {}))
        current = counts()
        eligible = []
        blocked = []
        for game in GAMES:
            for action in sorted(SUPPORTED_ACTIONS[game]):
                key = f'{game}:{action}'
                deficit = TARGET_PER_CELL - current[(game, action)]
                if deficit <= 0:
                    continue
                if action == 'ACTION6' and game not in ACTION6_COORDS:
                    blocked.append(key)
                    continue
                if failures.get(key, 0) >= 2:
                    blocked.append(key)
                    continue
                eligible.append((deficit, -failures.get(key, 0), key, game, action))
        if not eligible:
            status = 'collection_blocked_targets' if blocked else 'collection_quota_met'
            state = {'status': status, 'target_per_cell': TARGET_PER_CELL,
                     'counts': {f'{g}:{a}': current[(g, a)] for g in GAMES for a in ACTIONS},
                     'blocked_targets': sorted(blocked), 'failures': failures,
                     'heartbeat': time.time(), 'training_performed': False}
            write_state(state)
            print(json.dumps(state, ensure_ascii=False))
            return 0
        _, _, _, game, action = max(eligible)
        deficit = TARGET_PER_CELL - current[(game, action)]

        stamp = time.strftime('%Y%m%d-%H%M%S', time.gmtime())
        shard = OUT / 'shards' / f'{stamp}-{game}-{action.lower()}'
        raw = shard / 'raw'
        raw.mkdir(parents=True)
        cmd = [sys.executable, 'tools/aop/counterfactual_collect.py', '--out', str(raw),
               '--games', game, '--groups-per-game', '10', '--focus-actions', action,
               '--prefix-lengths', '50,75,100,125']
        if action == 'ACTION6':
            x, y = ACTION6_COORDS[game]
            cmd += ['--action6-x', str(x), '--action6-y', str(y)]
        result = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=540)
        (shard / 'collection.log').write_text(result.stdout, encoding='utf-8')
        labels = raw / 'counterfactual_labels.jsonl'
        added = len(read_rows(labels))
        if labels.exists() and added:
            final = shard / 'counterfactual_labels.jsonl'
            final.write_bytes(labels.read_bytes())
            (shard / 'manifest.json').write_text(json.dumps({
                'game': game, 'action': action, 'rows': added,
                'labels_sha256': sha(final), 'collector_exit': result.returncode,
                'created_at': time.time(), 'training_performed': False,
            }, indent=2) + '\n', encoding='utf-8')
        updated = counts()
        key = f'{game}:{action}'
        if added:
            failures.pop(key, None)
        else:
            failures[key] = failures.get(key, 0) + 1
        status = shard_status(added, failures[key])
        state = {'status': status, 'target': key, 'deficit_before': deficit,
                 'added_rows': added, 'collector_exit': result.returncode,
                 'target_per_cell': TARGET_PER_CELL,
                 'counts': {f'{g}:{a}': updated[(g, a)] for g in GAMES for a in ACTIONS},
                 'failures': failures, 'heartbeat': time.time(), 'training_performed': False}
        write_state(state)
        print(json.dumps(state, ensure_ascii=False))
        return 0 if result.returncode in (0, 1) else result.returncode
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
