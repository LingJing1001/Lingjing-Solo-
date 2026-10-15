#!/usr/bin/env python3
"""Bounded compatibility watch for canonical shards across repeated scans."""
from __future__ import annotations
import argparse, hashlib, json, time
from collections import Counter
from pathlib import Path
from typing import Any

def sha(path: Path) -> str:
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def scan(root: Path) -> dict[str, Any]:
    files=sorted(root.glob("*/counterfactual_labels.jsonl")); rows=0; bad=0; groups=set(); schemas=Counter(); hashes=[]
    for path in files:
        file_rows=0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            row=json.loads(line); file_rows+=1; rows+=1; groups.add(row.get("counterfactual_group_id")); schemas[row.get("schema_version")]+=1
            prov=row.get("provenance") or {}; delta=row.get("state_delta") or {}
            if row.get("schema_version") != "aop-canonical-v2" or not prov.get("sha256") or not prov.get("shard_id") or not row.get("replay_state_hash") or not delta:
                bad += 1
        if file_rows == 0: bad += 1
        hashes.append((str(path),file_rows,sha(path)))
    return {"files":len(files),"rows":rows,"groups":len(groups),"schemas":dict(schemas),"invalid_rows_or_files":bad,"snapshot_sha256":sha_bytes(json.dumps(hashes,sort_keys=True).encode())}

def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,required=True); ap.add_argument("--cycles",type=int,default=3); ap.add_argument("--interval",type=float,default=1.0); ap.add_argument("--out",type=Path,required=True); args=ap.parse_args()
    if args.cycles < 2: raise ValueError("cycles must be >= 2")
    snapshots=[]
    for index in range(args.cycles):
        snapshots.append({"cycle":index+1,"observed_at":time.time(),"scan":scan(args.root)})
        if index+1 < args.cycles: time.sleep(args.interval)
    stable=all(s["scan"]==snapshots[0]["scan"] for s in snapshots[1:])
    result={"schema":"aop.cross_group_compatibility_watch.v1","root":str(args.root),"cycles":snapshots,"stable":stable,"passed":stable and all(s["scan"]["invalid_rows_or_files"]==0 for s in snapshots) and all(s["scan"]["schemas"]=={"aop-canonical-v2":s["scan"]["rows"]} for s in snapshots)}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"COMPAT cycles={args.cycles} stable={stable} passed={result['passed']} rows={snapshots[-1]['scan']['rows']} groups={snapshots[-1]['scan']['groups']} invalid={snapshots[-1]['scan']['invalid_rows_or_files']} output={args.out}")
    return 0 if result["passed"] else 1
if __name__=="__main__": raise SystemExit(main())
