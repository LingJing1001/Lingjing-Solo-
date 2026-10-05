"""统一判决账本 + skill manifest——在线 CEAX verdict 与离线实验共用同一 schema。

一个 jsonl 流服务两端：
  在线  CEAX.observe_outcome 每个 verdict 追加一行（env CEAX_LEDGER=1 启用；
        best-effort——写失败绝不影响 agent，所有异常吞掉）。
  离线  训练/评测实验每次运行写一条 experiment 行（train_v5、benchmark 等）。
overnight 循环、skill manifest 与回放分析都消费同一条流。

行格式（schema 1）：
  verdict 行     {"schema":1, "type":"verdict", "ts":..., "game_id":...,
                  "block":..., "step":..., "action":..., "x":..., "y":...,
                  "delta_pixels":..., "progressed":..., "levels":...,
                  "effect":..., "grid_hash":..., "extra":{}}
  experiment 行  {"schema":1, "type":"experiment", "ts":..., "name":...,
                  "kind":"train|bench|validate", "config":{}, "artifact":"...",
                  "metric":{...}, "gate":{"metric":...,"line":...},
                  "verdict":"keep|discard|pending", "notes":"..."}

SkillManifest（state/skills/manifest.json）：
  name → {kind, path, version, gate:{...}, status: active|shadow|discarded,
          updated_at}。status 语义：active=运行时消费；shadow=产出但未接线；
  discarded=验证不过、留档。循环的 Keep/Discard 就是改这个 status。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1
STATUS_ACTIVE = "active"
STATUS_SHADOW = "shadow"
STATUS_DISCARDED = "discarded"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


@dataclass
class VerdictRecord:
    """CEAX 一步的判决记录（observe_outcome 的落盘形态）。"""

    game_id: str = ""
    block: int = 0                 # step // 100，实验块编号
    step: int = 0
    action: str = ""
    x: Optional[int] = None
    y: Optional[int] = None
    delta_pixels: int = 0
    progressed: bool = False
    levels: int = 0
    effect: float = 0.0
    grid_hash: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {"schema": SCHEMA_VERSION, "type": "verdict", "ts": _now()}
        d.update({k: v for k, v in asdict(self).items()})
        return d


def append_record(path: str | Path, record: Dict[str, Any]) -> bool:
    """追加一行 JSON；目录不存在则创建；任何 I/O 失败返回 False（绝不抛出）。"""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        rec = {"schema": SCHEMA_VERSION, "ts": _now()}
        rec.update(record)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
        return True
    except Exception:
        return False


def read_records(path: str | Path, kind: Optional[str] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if kind is None or r.get("type") == kind:
                    out.append(r)
    except OSError:
        return []
    return out


def experiment_row(*, name: str, kind: str, config: Dict[str, Any],
                   artifact: str = "", metric: Optional[Dict[str, float]] = None,
                   gate: Optional[Dict[str, Any]] = None,
                   verdict: str = "pending", notes: str = "") -> Dict[str, Any]:
    """离线实验一行（train/bench/validate 共用）。verdict ∈ keep|discard|pending。"""
    return {"schema": SCHEMA_VERSION, "type": "experiment", "ts": _now(),
            "name": name, "kind": kind, "config": config, "artifact": artifact,
            "metric": metric or {}, "gate": gate or {}, "verdict": verdict,
            "notes": notes}


class SkillManifest:
    """skill 产物登记簿：routes/npz/weights 等统一注册，Keep/Discard 改 status。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.skills: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> "SkillManifest":
        try:
            if self.path.is_file():
                d = json.loads(self.path.read_text(encoding="utf-8"))
                self.skills = d.get("skills", {})
        except Exception:
            self.skills = {}
        return self

    def register(self, name: str, *, kind: str, path: str, version: str = "",
                 gate: Optional[Dict[str, Any]] = None,
                 status: str = STATUS_ACTIVE) -> Dict[str, Any]:
        entry = {
            "kind": kind, "path": path, "version": version,
            "gate": gate or {}, "status": status, "updated_at": _now(),
        }
        old = self.skills.get(name) or {}
        old_status = old.get("status")
        # 已有 entry 只更新产物字段；status 已被人工/循环定为 discarded 时
        # 不被 silent 覆盖回 active（循环不得绕过闸门复活被弃技能）。
        if old_status == STATUS_DISCARDED and status == STATUS_ACTIVE:
            entry["status"] = STATUS_DISCARDED
        self.skills[name] = entry
        return entry

    def set_status(self, name: str, status: str) -> bool:
        if name not in self.skills:
            return False
        self.skills[name]["status"] = status
        self.skills[name]["updated_at"] = _now()
        return True

    def by_status(self, status: str) -> Dict[str, Dict[str, Any]]:
        return {n: e for n, e in self.skills.items() if e.get("status") == status}

    def save(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps({"schema": SCHEMA_VERSION, "updated_at": _now(),
                            "skills": self.skills},
                           ensure_ascii=False, indent=1),
                encoding="utf-8")
            return True
        except Exception:
            return False
