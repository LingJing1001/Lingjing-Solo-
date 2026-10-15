#!/usr/bin/env python3
"""Emit keyed candidate/baseline predictions and an independent counterfactual holdout report."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from collections import Counter
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from lingjing_solo.neural.aop_encoder import encode_label, profile_key
from lingjing_solo.neural.aop_model import load_checkpoint
from tools.aop.train import read_labels

def sha(path: Path) -> str:
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def predict(model, config: dict[str,Any], row: dict[str,Any]) -> tuple[str|None,bool]:
    names=list(config["action_names"]); ids={n:i for i,n in enumerate(names)}
    legal=[n for n in names if n in (row.get("legal_actions") or [])]
    if not legal: return None, False
    encoded=[]
    for name in names:
        r=dict(row); a=dict(row["requested_action"]); a.update(name=name,id=ids[name]+1,payload={}); r["requested_action"]=a
        encoded.append(encode_label(r, action_id=ids[name], action_count=len(names)))
    with torch.no_grad():
        scores=model(torch.stack(encoded))["action"][torch.arange(len(names)),torch.arange(len(names))]
        alpha=float(config.get("profile_prior_alpha",0.0)); priors=config.get("profile_log_priors") or {}
        if alpha:
            scores=scores+alpha*torch.tensor(priors.get(profile_key(row),[0.0]*len(names)),dtype=scores.dtype)
    legal_ids=[ids[n] for n in legal]; pred=names[legal_ids[int(scores[legal_ids].argmax().item())]]
    return pred, True

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--candidate",type=Path,required=True); ap.add_argument("--baseline",type=Path,required=True); ap.add_argument("--labels",type=Path,required=True); ap.add_argument("--predictions",type=Path,required=True); ap.add_argument("--report",type=Path,required=True); args=ap.parse_args()
    labels=[row for row in read_labels(args.labels) if row.get("counterfactual_group_id") and row.get("replay_verified") is True]
    if not labels:
        raise ValueError("no replay-verified counterfactual holdout rows")
    cand,cc=load_checkpoint(args.candidate); base,bc=load_checkpoint(args.baseline)
    rows=[]; c_ok=b_ok=0; valid=0; replay=0; groups=set()
    for row in labels:
        cp,cv=predict(cand,cc,row); bp,bv=predict(base,bc,row); actual=row["requested_action"]["name"]
        valid += int(cv and bv); replay += int(row.get("replay_verified") is True); groups.add(row["counterfactual_group_id"])
        c_ok += int(cp==actual); b_ok += int(bp==actual)
        rows.append({"prediction_key":{"episode_id":row["episode_id"],"step_id":row["step_id"],"counterfactual_group_id":row["counterfactual_group_id"],"replay_state_hash":row.get("replay_state_hash")},"actual_action":actual,"legal_actions":row.get("legal_actions",[]),"candidate":{"model_version":cc.get("version"),"action":cp,"correct":cp==actual},"baseline":{"model_version":bc.get("version"),"action":bp,"correct":bp==actual}})
    args.predictions.parent.mkdir(parents=True,exist_ok=True); args.predictions.write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in rows),encoding="utf-8")
    report={"schema":"aop.counterfactual_holdout_ab.v1","labels":str(args.labels),"labels_sha256":sha(args.labels),"candidate_checkpoint":str(args.candidate),"candidate_sha256":sha(args.candidate),"baseline_checkpoint":str(args.baseline),"baseline_sha256":sha(args.baseline),"rows":len(rows),"groups":len(groups),"replay_verified_rows":replay,"valid_prediction_rows":valid,"baseline_accuracy":b_ok/len(rows) if rows else 0.0,"candidate_accuracy":c_ok/len(rows) if rows else 0.0,"delta":(c_ok-b_ok)/len(rows) if rows else 0.0,"passed":bool(rows and valid==len(rows) and replay==len(rows) and c_ok>=b_ok),"prediction_file":str(args.predictions)}
    args.report.parent.mkdir(parents=True,exist_ok=True); args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"HOLDOUT rows={len(rows)} groups={len(groups)} replay_verified={replay} baseline={report['baseline_accuracy']:.4f} candidate={report['candidate_accuracy']:.4f} delta={report['delta']:.4f} passed={report['passed']}")
    return 0 if report["passed"] else 1
if __name__=="__main__": raise SystemExit(main())
