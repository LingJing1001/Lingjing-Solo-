#!/usr/bin/env python3
"""Bind existing canonical-v2 shards to the producing source revision.

This is a provenance migration only: row payloads and source hashes are not
changed. It writes a backup before changing manifests/rows and records the
binding revision in each manifest.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--source-commit',required=True); ap.add_argument('--backup',type=Path,required=True); a=ap.parse_args()
    shards=sorted(p.parent for p in a.root.glob('*/counterfactual_labels.jsonl'))
    if not shards: raise SystemExit('no canonical shards')
    if a.backup.exists(): raise SystemExit(f'backup exists: {a.backup}')
    shutil.copytree(a.root,a.backup)
    changed=0; rows=0
    for shard in shards:
        event=shard/'counterfactual_labels.jsonl'; manifest=shard/'manifest.json'
        data=[json.loads(x) for x in event.read_text().splitlines() if x.strip()]
        for row in data:
            prov=row.setdefault('provenance',{})
            prov['source_commit']=a.source_commit
        event.write_text(''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in data))
        meta=json.loads(manifest.read_text())
        meta['source_commit']=a.source_commit
        meta['provenance_binding']='source_commit_bound_by_rebind_canonical_provenance.py'
        meta['labels_sha256']=sha(event)
        manifest.write_text(json.dumps(meta,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
        changed+=1; rows+=len(data)
    print(json.dumps({'schema':'aop.canonical.provenance_rebind.v1','shards':changed,'rows':rows,'source_commit':a.source_commit,'backup':str(a.backup)},ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
