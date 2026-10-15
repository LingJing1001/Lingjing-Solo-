#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

from tools.aop.canonical_ingest import ingest_counterfactual_shard

ROOT = Path('/srv/agent-platform/projects/Lingjing-Solo-')
STATE = ROOT/'data/aop/continuous/state.json'
RAW_CUM = ROOT/'data/aop/continuous/labels.jsonl'
CANONICAL_SHARDS = ROOT/'data/aop/canonical/model/shards'
LOCK = ROOT/'data/aop/continuous/.lock'
TARGETS = [(g,a) for g in ['ar25','vc33','ls20','dc22','re86'] for a in [2,3,4,5,6,7]]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def load_state():
    if STATE.exists(): return json.loads(STATE.read_text())
    return {'next_target':0,'cycles':0,'rows_added':0,'status':'initializing'}

def run(cmd, timeout):
    return subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)

def main():
    if LOCK.exists():
        print(json.dumps({'status':'already_running','lock':str(LOCK)})); return 0
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(str(os.getpid()))
    try:
        state=load_state()
        if not os.environ.get('ARC_API_KEY'):
            state.update(status='blocked_missing_credential', heartbeat=time.time())
            STATE.write_text(json.dumps(state,indent=2)+'\n')
            print(json.dumps({'status':state['status'],'checkpoint':str(STATE)})); return 2
        game,action=TARGETS[state['next_target'] % len(TARGETS)]
        stamp=time.strftime('%Y%m%d-%H%M%S',time.gmtime())
        run_dir=ROOT/f'data/aop/runs/continuous-{stamp}-{game}-action{action}'
        raw=run_dir/'raw'; raw.mkdir(parents=True)
        cmd=['python3','tools/aop/counterfactual_collect.py','--out',str(raw),'--games',game,'--groups-per-game',os.environ.get('AOP_GROUPS_PER_RUN','2'),'--focus-actions',f'ACTION{action}','--prefix-lengths',os.environ.get('AOP_PREFIX_LENGTHS','20,30')]
        if action==6: cmd += ['--action6-x','15','--action6-y','31']
        result=run(cmd,540)
        (run_dir/'collection.log').write_text(result.stdout)
        labels=raw/'counterfactual_labels.jsonl'
        added=sum(1 for x in labels.read_text().splitlines() if x.strip()) if labels.exists() else 0
        canonical_result = None
        if added:
            with RAW_CUM.open('ab') as dst, labels.open('rb') as src: shutil.copyfileobj(src,dst)
            source_manifest = run_dir/'source-manifest.json'
            source_manifest.write_text(json.dumps({
                'schema_version': 'aop-counterfactual-shard-v1',
                'thread': 'model', 'shard_id': run_dir.name,
                'source': 'remote_counterfactual_v1',
                'source_sha256': sha(labels),
                'labels_sha256': sha(labels),
                'rows': added,
                'collector_exit': result.returncode,
                'created_at': time.time(),
                'provenance_complete': True,
                'training_performed': False,
            }, indent=2) + '\n')
            canonical_dir = CANONICAL_SHARDS/run_dir.name
            canonical_result = ingest_counterfactual_shard(
                labels, canonical_dir/'counterfactual_labels.jsonl', source_manifest, 'model'
            )
            canonical_result['shard_dir'] = str(canonical_dir)
        state.update(next_target=(state['next_target'] + (1 if added else 0)) % len(TARGETS), cycles=state['cycles']+1, rows_added=state['rows_added']+added, last_target=f'{game}:ACTION{action}', last_collection_exit=result.returncode, last_added=added, heartbeat=time.time(), cumulative_rows=sum(1 for _ in RAW_CUM.open()) if RAW_CUM.exists() else 0, canonical=canonical_result, training_performed=False, status='candidate_requires_freeze' if canonical_result else 'no_rows_retry')
        STATE.write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps({'status':state['status'],'target':f'{game}:ACTION{action}','added_rows':added,'cumulative_rows':state['cumulative_rows'],'collection_exit':result.returncode,'train_exit':None,'state':str(STATE)}))
        return 0 if result.returncode in (0, 1) else result.returncode
    finally:
        LOCK.unlink(missing_ok=True)

if __name__=='__main__': raise SystemExit(main())
