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

from tools.aop.canonical_ingest import ingest_counterfactual_shard

ROOT = Path('/srv/agent-platform/projects/Lingjing-Solo-')
BASE_LABELS = ROOT / 'data/aop/continuous/labels.jsonl'
OUT = ROOT / 'data/aop/collection'
CANONICAL_OUT = ROOT / 'data/aop/canonical/model/shards'
STATE = OUT / 'state.json'
LOCK = OUT / '.lock'
TARGET_PER_CELL = 500
GROUPS_PER_RUN = int(os.environ.get('AOP_GROUPS_PER_RUN', '10'))
PREFIX_LENGTHS = os.environ.get('AOP_PREFIX_LENGTHS', '20,30,40,50')
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
# dc22:(15,31) is corroborated by 179 prior recorded ACTION6 payloads.
ACTION6_COORDS = {'ar25': (15, 31), 'dc22': (15, 31)}


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]


def semantic_key(row: dict) -> tuple[str, str, str, str, str]:
    action = str((row.get('requested_action') or {}).get('name', ''))
    payload = json.dumps((row.get('requested_action') or {}).get('payload', {}), sort_keys=True, separators=(',', ':'))
    delta = row.get('state_delta') or {}
    before_hash = str(delta.get('before_hash') or row.get('replay_state_hash') or '')
    after_hash = str(delta.get('after_hash') or '')
    group = str(row.get('counterfactual_group_id') or row.get('episode_id') or '')
    return group, action, payload, before_hash, after_hash


def existing_shard_keys() -> set[tuple[str, str, str, str, str]]:
    keys: set[tuple[str, str, str, str, str]] = set()
    for path in OUT.glob('shards/*/counterfactual_labels.jsonl'):
        keys.update(semantic_key(row) for row in read_rows(path))
    return keys


def counts() -> Counter[tuple[str, str]]:
    result: Counter[tuple[str, str]] = Counter()
    for row in read_rows(BASE_LABELS):
        game = str(row.get('episode_id', '')).split('_', 1)[0]
        action = str((row.get('requested_action') or {}).get('name', ''))
        if game in GAMES and action in ACTIONS:
            result[(game, action)] += 1
    seen: set[tuple[str, str, str, str, str]] = set()
    for path in OUT.glob('shards/*/counterfactual_labels.jsonl'):
        for row in read_rows(path):
            key = semantic_key(row)
            if key in seen:
                continue
            seen.add(key)
            game = str(row.get('episode_id', '')).split('_', 1)[0]
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


def source_commit() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


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
        seed_base = time.time_ns() % 2_000_000_000
        cmd = [sys.executable, 'tools/aop/counterfactual_collect.py', '--out', str(raw),
               '--games', game, '--groups-per-game', str(GROUPS_PER_RUN), '--focus-actions', action,
               '--prefix-lengths', PREFIX_LENGTHS, '--seed-base', str(seed_base),
               '--run-id', stamp]
        if action == 'ACTION6':
            x, y = ACTION6_COORDS[game]
            cmd += ['--action6-x', str(x), '--action6-y', str(y)]
        try:
            result = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=540)
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ''
            if isinstance(output, bytes):
                output = output.decode('utf-8', errors='replace')
            result = subprocess.CompletedProcess(cmd, 124, output + '\\nCOLLECTOR_TIMEOUT\\n')
        (shard / 'collection.log').write_text(result.stdout, encoding='utf-8')
        labels = raw / 'counterfactual_labels.jsonl'
        raw_rows = read_rows(labels)
        seen = existing_shard_keys()
        unique_rows = []
        canonical_error = None
        for row in raw_rows:
            semantic = semantic_key(row)
            if semantic in seen:
                continue
            seen.add(semantic)
            unique_rows.append(row)
        added = len(unique_rows)
        if added:
            final = shard / 'counterfactual_labels.jsonl'
            final.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in unique_rows), encoding='utf-8')
            source_manifest = shard / 'manifest.json'
            producer_commit = source_commit()
            source_manifest.write_text(json.dumps({
                'game': game, 'action': action, 'raw_rows': len(raw_rows), 'unique_rows': added,
                'rows': added, 'labels_sha256': sha(final), 'collector_exit': result.returncode,
                'seed_base': seed_base, 'run_id': stamp,
                'created_at': time.time(), 'training_performed': False,
                'schema_version': 'aop-counterfactual-shard-v2',
                'thread': 'model', 'shard_id': shard.name,
                'source': 'remote_counterfactual_v1',
                'source_sha256': sha(final), 'provenance_complete': True,
                'source_commit': producer_commit,
            }, indent=2) + '\n', encoding='utf-8')
            canonical_dir = CANONICAL_OUT / shard.name
            try:
                canonical_result = ingest_counterfactual_shard(
                    final, canonical_dir / 'counterfactual_labels.jsonl', source_manifest, 'model'
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                canonical_result = None
                canonical_error = str(exc)
        else:
            canonical_result = None
        updated = counts()
        key = f'{game}:{action}'
        if added:
            failures.pop(key, None)
        else:
            failures[key] = failures.get(key, 0) + 1
        failure_count = failures.get(key, 0)
        status = shard_status(added, failure_count)
        state = {'status': status, 'target': key, 'deficit_before': deficit,
                 'added_rows': added, 'collector_exit': result.returncode,
                 'target_per_cell': TARGET_PER_CELL,
                 'counts': {f'{g}:{a}': updated[(g, a)] for g in GAMES for a in ACTIONS},
                 'failures': failures, 'heartbeat': time.time(), 'training_performed': False,
                 'canonical': canonical_result,
                 'status_detail': ('canonical_ingest_rejected' if canonical_error
                                   else 'candidate_requires_freeze' if canonical_result else status),
                 'canonical_error': canonical_error}
        write_state(state)
        print(json.dumps(state, ensure_ascii=False))
        if canonical_error:
            return 1
        return 0 if result.returncode in (0, 1) else result.returncode
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
