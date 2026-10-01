"""注册现有 skill 产物到 state/skills/manifest.json（AutoResearch 式循环的账本底座）。

skill 的统一形态：{kind, path, version, gate:{metric,line}, status}。
status ∈ active（运行时消费）/ shadow（产出未接线）/ discarded（验证不过，留档）。
gate 字段直接引用实测数字（来源见各条 notes/commit），循环据此判 keep/discard。

用法：python scripts/build_skill_manifest.py   （幂等，可反复运行）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lingjing_solo.transfer.ledger import (  # noqa: E402
    STATUS_ACTIVE, STATUS_DISCARDED, STATUS_SHADOW, SkillManifest,
)
from lingjing_solo.transfer.miner import ROUTES_DIR  # noqa: E402

MANIFEST = ROOT / "state" / "skills" / "manifest.json"

# gate 数字来源：state/r3_prune_bench.json 与 2026-09-30 双 GT 对比（5 局入口态，
# precision@10，同 harness）；routes 的 gate 是 tests/test_route_replay.py 的 3×WIN。
REGISTRY = [
    {
        "name": "route:bp35_l1", "kind": "route",
        "path": str(ROUTES_DIR / "bp35_l1.json"), "version": "2026-09",
        "gate": {"replay": "3xWIN", "levels": 1}, "status": STATUS_ACTIVE,
    },
    {
        "name": "route:cd82_all6", "kind": "route",
        "path": str(ROUTES_DIR / "cd82_all6.json"), "version": "2026-09",
        "gate": {"replay": "3xWIN", "levels": 6}, "status": STATUS_ACTIVE,
    },
    {
        "name": "route:tn36_l1l7", "kind": "route",
        "path": str(ROUTES_DIR / "tn36_l1l7.json"), "version": "2026-09-30",
        "gate": {"replay": "3xWIN", "levels": 7}, "status": STATUS_ACTIVE,
    },
    {
        "name": "route:vc33_l3", "kind": "route",
        "path": str(ROUTES_DIR / "vc33_l3.json"), "version": "2026-09",
        "gate": {"replay": "3xWIN", "levels": 3}, "status": STATUS_ACTIVE,
    },
    {
        "name": "click_heatmap_v3", "kind": "click_heatmap",
        "path": "F:/pro2/state/click_heatmap_v3.npz", "version": "v3+identity-fix",
        "gate": {"precision@10_newGT": 0.225, "precision@10_oldGT": 0.320,
                 "line": 0.5}, "status": STATUS_ACTIVE,
    },
    {
        "name": "click_heatmap_v4_entry", "kind": "click_heatmap",
        "path": "F:/pro2/state/click_heatmap_v4_entry.npz", "version": "v4b",
        "gate": {"precision@10_newGT": 0.300, "precision@10_oldGT": 0.300,
                 "line": 0.5}, "status": STATUS_SHADOW,
    },
    {
        "name": "click_heatmap_v5", "kind": "click_heatmap",
        "path": "F:/pro2/state/click_heatmap_v5.npz",
        "version": "v5-engine-truth", "gate": {
            "precision@10_newGT": 0.150, "precision@10_oldGT": 0.260, "line": 0.5},
        "status": STATUS_DISCARDED,
    },
    {
        "name": "affordance_ranker_weights", "kind": "affordance_ranker",
        "path": "F:/pro2/state/affordance_ranker_weights.json", "version": "v1+rank+kshot",
        "gate": {"logo_auc": 0.734, "logo_top1": 0.910}, "status": STATUS_SHADOW,
    },
]


def main() -> int:
    m = SkillManifest(MANIFEST)
    for e in REGISTRY:
        path_exists = Path(e["path"]).is_file()
        entry = m.register(e["name"], kind=e["kind"], path=e["path"],
                           version=e["version"], gate=e["gate"], status=e["status"])
        entry["artifact_exists"] = path_exists
        if not path_exists:
            entry.setdefault("notes", "产物缺失——登记时点检查")
    m.save()
    by = {"active": 0, "shadow": 0, "discarded": 0}
    for e in m.skills.values():
        by[e.get("status", "?")] = by.get(e.get("status", "?"), 0) + 1
    print(f"manifest → {MANIFEST}")
    print(f"skills={len(m.skills)}  active={by['active']}  shadow={by['shadow']}  "
          f"discarded={by['discarded']}")
    for name, e in sorted(m.skills.items()):
        gate = json.dumps(e.get("gate", {}), ensure_ascii=False)
        print(f"  {name:28s} {e['kind']:18s} {e['status']:9s} {gate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
