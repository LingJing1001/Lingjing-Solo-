#!/usr/bin/env python3
"""Append-only approval/audit chain and production write-back dry-run."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path

def digest(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def append_event(path:Path,event:dict):
    prev='GENESIS'
    if path.exists() and path.read_text().strip(): prev=json.loads(path.read_text().splitlines()[-1])['event_hash']
    event=dict(event); event['prev_hash']=prev; event['event_hash']=digest(event); event['created_at']=time.time()
    with path.open('a',encoding='utf-8') as f: f.write(json.dumps(event,ensure_ascii=False,sort_keys=True)+'\n')
    return event

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--log',type=Path,required=True); ap.add_argument('--candidate-sha',required=True); ap.add_argument('--rollback-sha',required=True); ap.add_argument('--mode',choices=['dry-run','approval-template'],default='dry-run'); a=ap.parse_args()
    a.log.parent.mkdir(parents=True,exist_ok=True)
    ev=append_event(a.log,{'schema':'appendix_a.audit.v1','event':'production_writeback_dry_run','candidate_sha256':a.candidate_sha,'rollback_manifest_sha256':a.rollback_sha,'production_writes':0,'runtime_authority_change':False,'external_ugc':'disabled','mode':a.mode})
    print(json.dumps({'status':'pass','production_writes':0,'runtime_authority_change':False,'event_hash':ev['event_hash'],'log':str(a.log)},ensure_ascii=False))
if __name__=='__main__': main()
