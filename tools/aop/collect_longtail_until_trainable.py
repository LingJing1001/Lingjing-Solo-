#!/usr/bin/env python3
"""Durable long-tail collector: retry until canonical action support is sufficient."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path('/srv/agent-platform/projects/Lingjing-Solo-')
sys.path.insert(0, str(ROOT))

from tools.aop.canonical_ingest import ingest_counterfactual_shard
from tools.aop.counterfactual_qa import validate_groups

TARGET_ACTIONS = ('ACTION2', 'ACTION3', 'ACTION5', 'ACTION7')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rows_from(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def support(root: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in root.glob('shards/*/raw/counterfactual_labels.jsonl'):
        for row in rows_from(path):
            counts[str((row.get('requested_action') or {}).get('name'))] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT / 'data/aop/collection/longtail-20261014')
    parser.add_argument('--target-rows-per-action', type=int, default=20)
    parser.add_argument('--groups-per-run', type=int, default=1)
    parser.add_argument('--prefix-lengths', default='10')
    parser.add_argument('--retry-seconds', type=int, default=5)
    parser.add_argument('--timeout', type=int, default=480)
    parser.add_argument('--target-actions', default=','.join(TARGET_ACTIONS),
                        help='comma-separated actions to collect')
    parser.add_argument('--games', default='ar25', help='comma-separated games')
    args = parser.parse_args()
    target_actions = tuple(action.strip() for action in args.target_actions.split(',') if action.strip())
    games = tuple(game.strip() for game in args.games.split(',') if game.strip())
    if not target_actions or not games:
        raise SystemExit('target-actions and games must be non-empty')
    if not os.environ.get('ARC_API_KEY'):
        raise SystemExit('ARC_API_KEY is missing')
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / 'shards').mkdir(parents=True, exist_ok=True)
    log = (args.root / 'loop.log').open('a', encoding='utf-8', buffering=1)
    attempt = 0
    try:
        while True:
            counts = support(args.root)
            status = {action: counts[action] for action in target_actions}
            print(json.dumps({'event': 'support', 'status': status}, sort_keys=True), file=log)
            if all(counts[action] >= args.target_rows_per_action for action in target_actions):
                final = {'status': 'trainable_longtail_support', 'support': status, 'target': args.target_rows_per_action}
                print(json.dumps(final, sort_keys=True), file=log)
                print(json.dumps(final, sort_keys=True))
                return 0

            attempt += 1
            run_id = f"longtail-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{os.getpid()}-{attempt}"
            shard = args.root / 'shards' / run_id
            raw = shard / 'raw'
            raw.mkdir(parents=True, exist_ok=False)
            cmd = [
                sys.executable, 'tools/aop/counterfactual_collect.py',
                '--out', str(raw), '--groups-per-game', str(args.groups_per_run),
                '--focus-actions', ','.join(target_actions), '--games', ','.join(games),
                '--prefix-lengths', args.prefix_lengths, '--run-id', run_id,
                '--seed-base', str(time.time_ns() % 2_000_000_000),
            ]
            print(json.dumps({'event': 'attempt', 'attempt': attempt, 'run_id': run_id, 'cmd': cmd}), file=log)
            try:
                result = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, timeout=args.timeout)
                output = result.stdout or ''
                exit_code = result.returncode
            except subprocess.TimeoutExpired as exc:
                output = (exc.stdout or '')
                if isinstance(output, bytes):
                    output = output.decode('utf-8', errors='replace')
                output += '\nCOLLECTOR_TIMEOUT\n'
                exit_code = 124
            (shard / 'collection.log').write_text(output, encoding='utf-8')
            labels = raw / 'counterfactual_labels.jsonl'
            groups = raw / 'groups.json'
            rows = rows_from(labels) if labels.exists() else []
            qa = validate_groups(rows) if rows else {'pass': False, 'errors': ['empty labels']}
            print(json.dumps({'event': 'attempt_result', 'run_id': run_id, 'exit': exit_code, 'rows': len(rows), 'qa': qa}, sort_keys=True), file=log)
            if exit_code == 0 and rows and groups.exists() and qa.get('pass'):
                source_manifest = shard / 'manifest.json'
                source_manifest.write_text(json.dumps({
                    'schema_version': 'aop-counterfactual-shard-v2',
                    'shard_id': run_id,
                    'thread': 'model',
                    'source': 'remote_counterfactual_v1',
                    'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'collector_exit': 0,
                    'rows': len(rows),
                    'labels_sha256': digest(labels),
                    'source_sha256': digest(labels),
                    'provenance_complete': True,
                    'training_performed': False,
                    'run_id': run_id,
                }, indent=2) + '\n', encoding='utf-8')
                output_path = ROOT / 'data/aop/canonical/model/shards' / run_id / 'counterfactual_labels.jsonl'
                result = ingest_counterfactual_shard(labels, output_path, source_manifest, 'model')
                print(json.dumps({'event': 'canonical_ingest', 'run_id': run_id, 'result': result}, sort_keys=True), file=log)
            else:
                print(json.dumps({'event': 'discard_unusable_attempt', 'run_id': run_id, 'reason': 'requires exit=0, groups.json, and QA pass'}, sort_keys=True), file=log)
            time.sleep(args.retry_seconds)
    finally:
        log.close()


if __name__ == '__main__':
    raise SystemExit(main())
