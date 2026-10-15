#!/usr/bin/env python3
"""Create and verify a hash-bound, executable Appendix A rollback bundle."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def create(args):
    args.out.mkdir(parents=True,exist_ok=True); entries=[]
    for src in args.artifact:
        src=src.resolve()
        if not src.is_file(): raise ValueError(f'missing artifact: {src}')
        dst=args.out/'artifacts'/src.name; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
        entries.append({'source':str(src),'file':str(dst.relative_to(args.out)),'sha256':sha(dst),'size':dst.stat().st_size})
    manifest={'schema':'appendix_a.rollback.v1','mode':'restore-only','artifacts':entries,'authority_status':'shadow','runtime_authority':False,'external_ugc':'disabled'}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
    print(json.dumps({'status':'created','bundle':str(args.out),'artifacts':len(entries)},ensure_ascii=False))

def verify(args):
    m=json.loads((args.bundle/'manifest.json').read_text()); bad=[]
    for e in m.get('artifacts',[]):
        p=args.bundle/e['file']
        if not p.is_file() or sha(p)!=e['sha256'] or p.stat().st_size!=e['size']: bad.append(e['file'])
    result={'schema':m.get('schema'),'status':'pass' if not bad else 'blocked','artifacts':len(m.get('artifacts',[])),'bad':bad,'runtime_authority':m.get('runtime_authority')}
    print(json.dumps(result,ensure_ascii=False)); return 0 if result['status']=='pass' else 2

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    c=sub.add_parser('create'); c.add_argument('--out',type=Path,required=True); c.add_argument('--artifact',type=Path,action='append',required=True); c.set_defaults(fn=create)
    v=sub.add_parser('verify'); v.add_argument('--bundle',type=Path,required=True); v.set_defaults(fn=verify)
    a=ap.parse_args(); result=a.fn(a); return result if isinstance(result,int) else 0
if __name__=='__main__': raise SystemExit(main())
