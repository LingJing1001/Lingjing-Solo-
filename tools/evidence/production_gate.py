#!/usr/bin/env python3
"""Fail-closed production-admission checklist for Appendix A evidence."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any

def check(report: dict[str,Any], compatibility: dict[str,Any], rollback: Path|None, audit: Path|None, activation: Path|None = None) -> dict[str,Any]:
    failures=[]; checks={}
    def require(name: str, ok: bool, reason: str) -> None:
        checks[name]={"passed":bool(ok),"reason":reason}
        if not ok: failures.append({"check":name,"reason":reason})
    policy=report.get("policy",{}); canon=report.get("canonical",{}); proj=report.get("projection",{})
    hold=report.get("counterfactual_verified_holdout",{}); gate=report.get("gatebench",{}).get("metrics",{})
    approval=report.get("independent_approval",{})
    require("authority_not_shadow", policy.get("authority_status")=="production", "必须经过显式审批后将 authority_status 从 shadow 切换为 production；本次未自动切换")
    require("source_commit_bound", canon.get("source_commits",{}).get("null",0)==0, "全部 canonical rows 必须绑定 source commit；当前存在未绑定 rows")
    require("holdout_positive_delta", float(hold.get("delta",-1))>0 and hold.get("passed") is True, "verified counterfactual holdout 必须通过且相对 baseline 有正 delta")
    require("holdout_binding", hold.get("rows",0)>0 and hold.get("valid_prediction_rows")==hold.get("rows") and hold.get("replay_verified_rows")==hold.get("rows"), "holdout rows、keyed predictions、replay verification 必须全量匹配")
    require("gatebench_clean", float(gate.get("false_accept",1))==0 and float(gate.get("false_reject",1))==0 and float(gate.get("replay_determinism",0))==1, "GateBench false accept/reject 必须为 0 且 replay determinism 必须为 1")
    require("projection_clean", proj.get("status")=="pass" and proj.get("invalid_rows",1)==0, "projection 必须无 invalid rows 且 status=pass")
    require("compatibility_stable", compatibility.get("passed") is True and compatibility.get("stable") is True, "必须有绑定同一 candidate/source 的稳定 compatibility report")
    rollback_ok=False
    if rollback and rollback.is_dir() and (rollback/"manifest.json").is_file():
        try:
            m=json.loads((rollback/"manifest.json").read_text())
            rollback_ok=m.get("schema")=="appendix_a.rollback.v1" and all((rollback/e["file"]).is_file() for e in m.get("artifacts",[])) and m.get("runtime_authority") is False
        except (OSError,KeyError,json.JSONDecodeError): rollback_ok=False
    require("rollback_artifact", rollback_ok, "必须存在可读、hash-bound、runtime_authority=false 的 rollback bundle")
    approval_ok=(
        approval.get("schema")=="appendix_a.independent_approval.v1"
        and approval.get("verdict")=="passed"
        and approval.get("approval_granted") is True
        and bool(approval.get("verifier_agent"))
        and bool(approval.get("task_id"))
        and bool(approval.get("verified_at"))
        and bool(approval.get("report_sha256"))
        and isinstance(approval.get("artifact_hashes"),dict)
        and len(approval.get("artifact_hashes",{})) >= 3
    )
    require("independent_approval", approval_ok, "必须有独立 verifier report、artifact hashes、审批身份和时间戳")
    audit_ok=False
    if audit and audit.is_file():
        try:
            rows=[json.loads(x) for x in audit.read_text().splitlines() if x.strip()]
            audit_ok=bool(rows) and all(r.get("production_writes")==0 and (r.get("runtime_authority_change") is False or r.get("event")=="runtime_activation") for r in rows)
        except (OSError,TypeError,json.JSONDecodeError): audit_ok=False
    require("audit_write_path", audit_ok, "必须有 append-only audit log 且 dry-run production_writes=0")
    require("external_ugc_boundary", policy.get("external_ugc")=="disabled", "首次 production 仍应保持 external UGC disabled，待单独 UGC gate")
    activation_ok=False
    if activation and activation.is_file():
        try:
            a=json.loads(activation.read_text())
            activation_ok=a.get("candidate_id") and a.get("authority_status")=="production" and a.get("active") is True and a.get("approval",{}).get("verdict")=="passed"
        except (OSError,TypeError,json.JSONDecodeError): activation_ok=False
    if activation is not None:
        require("runtime_activation", bool(activation_ok), "必须读回 hash-bound activation record 且 authority_status=production")
    return {"schema":"appendix_a.production_gate.v1","status":"ready" if not failures else "blocked","checks":checks,"failures":failures,"policy":{"runtime_authority_change":"performed" if activation_ok else "not_performed","external_ugc":"disabled","writeback":"not_enabled"}}

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--readiness",type=Path,required=True); ap.add_argument("--compatibility",type=Path,required=True); ap.add_argument("--rollback",type=Path); ap.add_argument("--audit",type=Path); ap.add_argument("--activation",type=Path); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args()
    result=check(json.loads(a.readiness.read_text()),json.loads(a.compatibility.read_text()),a.rollback,a.audit,a.activation)
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True)+"\n")
    print(f"PRODUCTION status={result['status']} failed_checks={len(result['failures'])} output={a.out}")
    return 0 if result["status"]=="ready" else 2
if __name__=="__main__": raise SystemExit(main())
